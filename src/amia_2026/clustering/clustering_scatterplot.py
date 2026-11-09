"""Render an interactive 2D UMAP scatterplot of a reducer+HDBSCAN method's trial.

Steps 4/5 (`umap_hdbscan_clustering.py`, `pacmap_hdbscan_clustering.py`)
write only a metrics JSON — no cluster-assignments Parquet (see
`reduction_hdbscan_pipeline.py`'s own docstring). So rather than reading a
persisted per-row assignments file, this module re-fits the trial directly
from the already-resumable Optuna study that run left behind — the same
re-fit-on-demand approach `verify_trial_reproducibility.py`/
`generate_cluster_labels.py` already use. By default that's the trial
identified by `--metrics-path`'s own `optuna_study_summary.best_trial_number`
(the winner); `--trial-number` substitutes any other recorded trial number
for it instead, to inspect a trial that never won. `refit_winning_trial`
does the re-fit itself: hyperparameters are re-derived via
`optuna_search.resolve_trial_hyperparams` (not `trial.params` directly — an
`"int_ratio"`-kind hyperparameter holds its raw sampled ratio there, not
the resolved integer the original fit used), then re-fit via
`reduction_sweep.evaluate_config`'s own seed/repeat sweep, picking the same
most-reproducible seed's representative repeat the original run would
have. This never touches the Optuna study's own journal file or writes any
new output besides the HTML below.

Before rendering the legend, this module also runs the same two-stage
Anthropic-LLM relabeling pass (`llm_labeling.relabel_topics_with_llm`) that
`reduction_hdbscan_pipeline.run` used to run itself, over that trial's own
topics -- the labeling logic itself is unchanged, only *where* it runs
moved. For the winning trial (the default), those topics are the real
zero-shot ones recorded in the metrics JSON's own `topics` field
(reconstructed via `build_topics_from_metrics`); for a `--trial-number`
override, the metrics file's `topics` don't apply (they describe the
winner's BERTopic result, not an arbitrary trial's), so fresh placeholder
topics are built from that trial's own cluster sizes instead, the same way
`generate_cluster_labels.py` does. Both re-fits are purely for this render:
nothing is ever written back to `--metrics-path`, and a fresh LLM call/
HDBSCAN re-fit happens every time this script runs. A failure in either
(e.g. no `ANTHROPIC_API_KEY`, or the replayed trial failing to re-evaluate
to status `"ok"`) is handled the same way `reduction_hdbscan_pipeline.run`
used to: the LLM failure falls back to the zero-shot/placeholder labels and
continues; a re-fit status mismatch raises, since there would be no valid
cluster assignments left to plot.

The 2-dimensional projection always reuses that method's own "optimal run"
seed (`winning_seed`) rather than introducing new randomness, and is fit
from the same already-loaded embedding matrix the trial re-fit uses — both
read `--embeddings-path` exactly once.

Every per-method difference above is captured once in `VISUALIZATION_SPECS`;
no function below branches on which method it's looking at.

Writes one HTML file, at the path given via `--output-path`. Re-running this
step against the same output path overwrites the existing file wholesale.
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import optuna
import plotly.express as px
import plotly.graph_objects as go
import pyarrow as pa
import umap

from amia_2026.logging_utils import configure_logging, log_memory_usage
from amia_2026.paths import label_from_slug, slug_from_filename
from amia_2026.clustering import clustering_io
from amia_2026.clustering import pacmap_hdbscan_clustering
from amia_2026.clustering import umap_hdbscan_clustering
from amia_2026.clustering.algorithms import llm_labeling
from amia_2026.clustering.algorithms.bertopic_model import TopicRepresentation
from amia_2026.clustering.algorithms.hdbscan_model import HDBSCANClusterer
from amia_2026.clustering.optimization import optuna_search
from amia_2026.clustering.optimization import reduction_sweep
from amia_2026.clustering.reduction_hdbscan_pipeline import ReducerPlugin

logger = logging.getLogger(__name__)

# Used only when a method has no recorded clustering-time seed at all. No
# currently-registered method lacks one (every `VisualizationSpec` below
# resolves a concrete seed), but this stays as a safety net for a future
# method that might not.
DEFAULT_FALLBACK_SEED = 42


def _parse_rgb(rgb_str: str) -> tuple[int, int, int]:
    """Parse a plotly `"rgb(r, g, b)"` string into an `(r, g, b)` int tuple."""
    r, g, b = (int(part.strip()) for part in rgb_str.split("(")[1].split(")")[0].split(","))
    return r, g, b


def _pastelize(rgb_str: str, white_mix: float) -> str:
    """Lighten one `"rgb(r, g, b)"` color by blending it toward white, preserving its hue.

    Hue (not saturation/lightness) is what makes a categorical palette
    colorblind-distinguishable, so blending toward white keeps that
    distinguishability while producing a pastel look.

    Args:
        rgb_str: A plotly-style `"rgb(r, g, b)"` color string.
        white_mix: Fraction of white to blend in, `0.0` (unchanged) to `1.0` (pure white).

    Returns:
        A new `"rgb(r, g, b)"` string, each channel moved `white_mix` of the way to 255.
    """
    r, g, b = _parse_rgb(rgb_str)
    blended = tuple(round(c * (1 - white_mix) + 255 * white_mix) for c in (r, g, b))
    return f"rgb({blended[0]}, {blended[1]}, {blended[2]})"


# Plotly's own `Safe` qualitative palette (from CARTOColors) is explicitly
# designed to stay distinguishable under the common forms of color vision
# deficiency. Its own trailing entry is a neutral gray, intended for
# "not applicable" categories -- reserved here as `NOISE_COLOR` rather than
# pastelized, so noise points stay visually distinct from every pastelized
# real cluster. Every other entry is lightened toward white (see
# `_pastelize`) so the palette reads as pastel without losing the hue
# separation that makes it colorblind-safe in the first place.
_PASTEL_WHITE_MIX = 0.45
_SAFE_PALETTE = px.colors.qualitative.Safe
NOISE_COLOR = _SAFE_PALETTE[-1]
PASTEL_COLORBLIND_PALETTE = [_pastelize(color, _PASTEL_WHITE_MIX) for color in _SAFE_PALETTE[:-1]]


@dataclass(frozen=True)
class VisualizationSpec:
    """Everything this module needs to know about one clustering method.

    Attributes:
        key: Short identifier, also the CLI's `--method` value.
        plugin: This method's reducer `ReducerPlugin` (same one steps 4/5
            use), needed to re-derive the study name and re-fit the winning
            trial (see `refit_winning_trial`).
        default_config_path: This method's own reducer's default Optuna
            search config YAML path.
        extract_seed_fn: `(metrics_dict) -> seed | None`. `None` means this
            method has no recorded clustering-time seed at all.
    """

    key: str
    plugin: ReducerPlugin
    default_config_path: Path
    extract_seed_fn: Callable[[dict], int | None]


VISUALIZATION_SPECS: dict[str, VisualizationSpec] = {
    "umap_hdbscan": VisualizationSpec(
        key="umap_hdbscan",
        plugin=umap_hdbscan_clustering.PLUGIN,
        default_config_path=umap_hdbscan_clustering.DEFAULT_CONFIG_PATH,
        extract_seed_fn=lambda m: m.get("winning_seed"),
    ),
    "pacmap_hdbscan": VisualizationSpec(
        key="pacmap_hdbscan",
        plugin=pacmap_hdbscan_clustering.PLUGIN,
        default_config_path=pacmap_hdbscan_clustering.DEFAULT_CONFIG_PATH,
        extract_seed_fn=lambda m: m.get("winning_seed"),
    ),
}


def load_metrics(spec: VisualizationSpec, metrics_path: Path) -> dict:
    """Read a method's metrics JSON.

    Args:
        spec: The method's `VisualizationSpec`.
        metrics_path: Path to that method's metrics JSON.

    Returns:
        The parsed metrics dict.

    Raises:
        FileNotFoundError: If `metrics_path` doesn't exist, naming which
            clustering module to run first.
    """
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"No {spec.key} metrics found at {metrics_path}; run the {spec.key} clustering step "
            "first and pass its metrics path via --metrics-path"
        )
    with open(metrics_path, encoding="utf-8") as f:
        return json.load(f)


def refit_winning_trial(
    spec: VisualizationSpec,
    metrics: dict,
    embeddings_path: Path,
    embedding_matrix: np.ndarray,
    config_path: Path,
    trial_number_override: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Re-fit one of this method's Optuna trials directly, for display-time cluster assignments.

    Locates the same resumable Optuna study `reduction_hdbscan_pipeline.run`
    would have (derived from `embeddings_path`'s filename and `spec.plugin`,
    under `config_path`'s own `study_dir`), loads the trial identified below,
    and re-fits its hyperparameters via the same seed/repeat reproducibility
    sweep every trial ran internally (`reduction_sweep.evaluate_config`,
    using `config_path`'s own `reproducibility.seeds`/`repeats_per_seed`).

    Args:
        spec: The method's `VisualizationSpec`.
        metrics: That method's parsed metrics dict; its own
            `optuna_study_summary.best_trial_number` identifies which trial
            to replay, unless overridden by `trial_number_override`.
        embeddings_path: Embeddings Parquet path, used only to re-derive this
            run's study name (must be the same file steps 4/5 were
            originally pointed at).
        embedding_matrix: The already-loaded, non-reduced embedding matrix
            (see `clustering_io.extract_embedding_matrix`).
        config_path: Optuna search config YAML (search space, fixed params,
            reproducibility seeds/repeats, `study_dir`) — see
            `optuna_search.load_optuna_config`.
        trial_number_override: If given, re-fits this trial instead of
            `metrics["optuna_study_summary"]["best_trial_number"]` — lets a
            caller inspect any recorded trial's clustering, not just the
            study's current winner, the same way `--trial-number` already
            does for `generate_cluster_labels.py`/`verify_trial_reproducibility.py`.

    Returns:
        `(labels, probabilities)` — the replayed trial's representative
        repeat's plain-HDBSCAN cluster assignments, row-aligned with
        `embedding_matrix`.

    Raises:
        FileNotFoundError: If no Optuna study journal is found for this method/input.
        RuntimeError: If the replayed trial's hyperparameters don't
            re-evaluate to status `"ok"` (mirrors `generate_cluster_labels.py`'s
            own guard -- there would be no valid cluster assignments left to plot).
    """
    config = optuna_search.load_optuna_config(config_path, require_reproducibility=True)
    slug = slug_from_filename(embeddings_path.name, "_trials_embeddings.parquet")
    study_name = f"{slug}_{spec.plugin.name}_hdbscan"
    storage_path = config.study_dir / f"{study_name}_study.journal"
    if not storage_path.exists():
        raise FileNotFoundError(f"No Optuna study journal found at {storage_path}")
    storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(str(storage_path)))
    study = optuna.load_study(study_name=study_name, storage=storage)

    trial_number = (
        trial_number_override if trial_number_override is not None
        else metrics["optuna_study_summary"]["best_trial_number"]
    )
    logger.info(
        "refit_winning_trial: re-fitting trial #%d for %s (%s)",
        trial_number, spec.key,
        "explicit --trial-number override" if trial_number_override is not None else "metrics' own best_trial_number",
    )
    trial = optuna_search.find_trial_by_number(study, trial_number)
    hyperparams = optuna_search.resolve_trial_hyperparams(trial, config.search_space)

    fixed_metric = config.fixed[spec.plugin.fixed_metric_config_key]
    fit_reduced = spec.plugin.make_fit_reduced(fixed_metric)

    def build_hdbscan_clusterer(params: dict) -> HDBSCANClusterer:
        return HDBSCANClusterer(
            min_cluster_size=params["min_cluster_size"],
            min_samples=params["min_samples"],
            metric=fixed_metric,
            cluster_selection_epsilon=config.fixed["cluster_selection_epsilon"],
            cluster_selection_method=params["cluster_selection_method"],
            alpha=config.fixed["alpha"],
        )

    candidate = reduction_sweep.evaluate_config(
        fit_reduced, hyperparams, config.reproducibility.seeds, config.reproducibility.repeats_per_seed,
        spec.plugin.validate_hyperparameters, build_hdbscan_clusterer, embedding_matrix, embedding_matrix.shape[0],
    )
    if candidate.status != "ok":
        raise RuntimeError(
            f"Trial #{trial.number}'s configuration re-evaluated to status={candidate.status!r} instead "
            f"of 'ok'; cannot render a scatterplot for {spec.key}"
        )
    representative = reduction_sweep.representative_repeat(candidate)
    return representative.labels, representative.probabilities


def resolve_seed(spec: VisualizationSpec, metrics: dict) -> tuple[int, bool]:
    """Resolve the seed to use for this method's 2D UMAP projection.

    Args:
        spec: The method's `VisualizationSpec`.
        metrics: That method's parsed metrics dict for this condition.

    Returns:
        `(seed, explicit)` — `explicit` is `False` only when the method has
        no recorded clustering-time seed at all and `DEFAULT_FALLBACK_SEED`
        was used instead (logged as a warning, never silently).
    """
    seed = spec.extract_seed_fn(metrics)
    if seed is not None:
        return seed, True
    logger.warning(
        "resolve_seed: %s has no recorded clustering-time seed for this condition; falling back "
        "to DEFAULT_FALLBACK_SEED=%d (this is not the actual seed used during clustering, since "
        "none was recorded)",
        spec.key, DEFAULT_FALLBACK_SEED,
    )
    return DEFAULT_FALLBACK_SEED, False


def resolve_points(embedding_matrix: np.ndarray, seed: int) -> np.ndarray:
    """Fit a fresh 2D UMAP projection over an already-loaded embedding matrix.

    Args:
        embedding_matrix: The original, non-reduced embedding matrix (the
            same one `refit_winning_trial` re-fits against, so the result is
            row-aligned with it by construction).
        seed: The seed resolved by `resolve_seed` (or a CLI override).

    Returns:
        An `(n_rows, 2)` array of 2D coordinates.
    """
    logger.info("resolve_points: computing a fresh 2D UMAP projection (seed=%d)", seed)
    reducer = umap.UMAP(n_components=2, random_state=seed)
    return reducer.fit_transform(embedding_matrix)


def build_topics_from_metrics(metrics: dict) -> list[TopicRepresentation]:
    """Reconstruct a method's zero-shot `TopicRepresentation` list from its metrics JSON.

    Args:
        metrics: As returned by `load_metrics` -- its own `topics` field is
            written verbatim from `bertopic_model.BERTopicFitResult.topics`
            (see `reduction_hdbscan_pipeline.build_metrics_dict`), so this is
            a lossless round-trip back into `TopicRepresentation` objects.

    Returns:
        One `TopicRepresentation` per entry in `metrics["topics"]`,
        including the noise (`topic_id == -1`) entry.
    """
    return [
        TopicRepresentation(
            topic_id=entry["topic_id"], keywords=entry["keywords"], label=entry["label"], count=entry["count"]
        )
        for entry in metrics["topics"]
    ]


def resolve_topic_labels(
    topics: list[TopicRepresentation],
    guide_labels: list[str],
    assignments_table: pa.Table,
    model: str,
    primary_prompt_path: Path,
    secondary_prompt_path: Path,
) -> dict[int, str]:
    """Run the two-stage LLM relabeling pass over a trial's own topics, for display only.

    Mirrors what `reduction_hdbscan_pipeline.run` used to do for the winning
    trial, with the same fallback behavior: a failure here (LLM error,
    missing `ANTHROPIC_API_KEY`, etc.) is logged and `topics`' own original
    labels are used for the plot instead of aborting the render. Never
    writes back to `assignments_table` -- the caller only gets the
    resulting `{topic_id: label}` lookup to color/caption the plot.

    Args:
        topics: The replayed trial's own topics -- either `metrics["topics"]`
            reconstructed via `build_topics_from_metrics` (the winning
            trial's real BERTopic/zero-shot result), or placeholder topics
            built from the refit labels directly when `--trial-number`
            overrides to a non-winning trial (see `run`), since the
            metrics file's own `topics` only describes the winner.
        guide_labels: The `--candidate-labels` list recorded in the metrics
            JSON, reused here as soft guidance for both LLM calls.
        assignments_table: The in-memory table built by `run` from
            `refit_winning_trial`'s own output; its `criterion_text`/
            `cluster_id`/`cluster_probability` columns are the inputs to the
            LLM pass.
        model: Anthropic model id (see `llm_labeling.DEFAULT_LLM_MODEL`).
        primary_prompt_path: `--primary-summary-prompt-path` file.
        secondary_prompt_path: `--secondary-summary-prompt-path` file.

    Returns:
        `{topic_id: label}`, one entry per entry of `topics` (including
        noise) -- LLM-relabeled where that succeeded, `topics`' own
        original label otherwise.
    """
    labels = np.array(assignments_table.column("cluster_id").to_pylist())
    probabilities = np.array(assignments_table.column("cluster_probability").to_pylist())
    criterion_texts = assignments_table.column("criterion_text").to_pylist()
    try:
        topics = llm_labeling.relabel_topics_with_llm(
            criterion_texts, labels, probabilities, topics, guide_labels, model,
            primary_prompt_path, secondary_prompt_path,
        )
    except MemoryError:
        logger.error(
            "resolve_topic_labels: LLM-based cluster relabeling raised MemoryError; keeping zero-shot labels"
        )
        log_memory_usage(
            logger, "resolve_topic_labels: after MemoryError during LLM relabeling", level=logging.ERROR
        )
    except Exception:
        logger.exception("resolve_topic_labels: LLM-based cluster relabeling failed; keeping zero-shot labels")
    return {topic.topic_id: topic.label for topic in topics}


def build_plot_frame(
    assignments_table: pa.Table, points_2d: np.ndarray, topic_labels_by_id: dict[int, str]
) -> dict[str, list]:
    """Zip cluster assignments with their 2D coordinates into a plotting-ready frame.

    Args:
        assignments_table: The in-memory table built by `run` from
            `refit_winning_trial`'s own output.
        points_2d: As returned by `resolve_points`, row-aligned with `assignments_table`.
        topic_labels_by_id: As returned by `resolve_topic_labels` -- the
            (possibly LLM-relabeled) label for each non-noise `cluster_id`.

    Returns:
        A dict with keys `x`, `y`, `cluster_label` (`"Noise"` for
        `cluster_id == -1`; `"<cluster_id>: <topic_label>"` otherwise, so
        the legend names each cluster by its own topic label rather than a
        bare id), `criterion_text`, `nct_id`.
    """
    cluster_ids = assignments_table.column("cluster_id").to_pylist()
    cluster_labels = [
        "Noise" if cluster_id == -1 else f"{cluster_id}: {topic_labels_by_id[cluster_id]}"
        for cluster_id in cluster_ids
    ]
    return {
        "x": points_2d[:, 0].tolist(),
        "y": points_2d[:, 1].tolist(),
        "cluster_label": cluster_labels,
        "criterion_text": assignments_table.column("criterion_text").to_pylist(),
        "nct_id": assignments_table.column("nct_id").to_pylist(),
    }


def resolve_legend_order(cluster_labels: list[str]) -> list[str]:
    """Order `cluster_label` values for consistent legend display: ascending cluster id, `"Noise"` last.

    Args:
        cluster_labels: The `cluster_label` list from `build_plot_frame`.

    Returns:
        Every distinct value in `cluster_labels`, real clusters sorted
        ascending by their numeric id prefix, with `"Noise"` (if present)
        placed last.
    """
    distinct = set(cluster_labels)
    non_noise = sorted(
        (label for label in distinct if label != "Noise"), key=lambda label: int(label.split(":", 1)[0])
    )
    return non_noise + (["Noise"] if "Noise" in distinct else [])


def build_color_discrete_map(legend_order: list[str]) -> dict[str, str]:
    """Assign each legend entry a fixed color: pastel colorblind-safe hues for clusters, gray for noise.

    Args:
        legend_order: As returned by `resolve_legend_order`. Colors are
            assigned by position, cycling through `PASTEL_COLORBLIND_PALETTE`
            if there are more clusters than palette colors, so the same
            cluster id always gets the same color across repeated runs.

    Returns:
        A dict suitable for `plotly.express.scatter`'s `color_discrete_map`.
    """
    non_noise = [label for label in legend_order if label != "Noise"]
    color_map = {
        label: PASTEL_COLORBLIND_PALETTE[i % len(PASTEL_COLORBLIND_PALETTE)] for i, label in enumerate(non_noise)
    }
    if "Noise" in legend_order:
        color_map["Noise"] = NOISE_COLOR
    return color_map


PLOT_TITLE = "Breast Cancer Trials' Eligibility Criteria Clusters"


def render_scatterplot(frame: dict[str, list]) -> go.Figure:
    """Build the interactive Plotly scatterplot.

    Args:
        frame: As returned by `build_plot_frame`.

    Returns:
        A `plotly.graph_objects.Figure` with cluster-colored points (pastel,
        colorblind-safe hues — see `PASTEL_COLORBLIND_PALETTE`), a legend
        naming each cluster by its own topic label, a criterion-text hover
        tooltip, and a centered, fixed title (`PLOT_TITLE`). The x/y axes
        carry no ticks, lines, or gridlines, since the UMAP projection's
        coordinate scale has no intrinsic meaning.
    """
    legend_order = resolve_legend_order(frame["cluster_label"])
    fig = px.scatter(
        frame,
        x="x",
        y="y",
        color="cluster_label",
        category_orders={"cluster_label": legend_order},
        color_discrete_map=build_color_discrete_map(legend_order),
        hover_data={"criterion_text": True, "nct_id": True, "x": False, "y": False},
        title=PLOT_TITLE,
        labels={"cluster_label": "Cluster"},
    )
    fig.update_layout(title_x=0.5)
    axis_style = dict(
        showticklabels=False, ticks="", showline=False, showgrid=False, zeroline=False, title=None
    )
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    return fig


def run(
    method: str,
    metrics_path: Path,
    embeddings_path: Path,
    output_path: Path,
    llm_model: str,
    primary_summary_prompt_path: Path,
    secondary_summary_prompt_path: Path,
    config_path: Path | None = None,
    seed_override: int | None = None,
    trial_number: int | None = None,
) -> Path:
    """Render and write a 2D UMAP scatterplot for one of a method's trials.

    Args:
        method: Which clustering method's output to visualize — one of
            `VISUALIZATION_SPECS`'s keys.
        metrics_path: That method's metrics JSON path.
        embeddings_path: The embeddings Parquet originally passed to that
            method's own `--input-path` — read once here, both to re-fit
            the replayed trial (see `refit_winning_trial`) and to fit the
            fresh 2D projection (see `resolve_points`).
        output_path: Where to write the HTML.
        llm_model: Anthropic model id for the two-stage LLM relabeling pass
            run here, for display only (see `resolve_topic_labels`).
        primary_summary_prompt_path: File containing the primary-summarization
            system prompt (see `llm_labeling.primary_summarize`).
        secondary_summary_prompt_path: File containing the secondary-summarization
            system prompt (see `llm_labeling.secondary_summarize`).
        config_path: Optuna search config YAML for `refit_winning_trial`;
            defaults to that method's own reducer's default config.
        seed_override: If given, used instead of that method's own
            recorded/fallback seed.
        trial_number: If given, re-fits and plots this specific Optuna trial
            instead of `metrics_path`'s own recorded winner (see
            `refit_winning_trial`'s `trial_number_override`). Since the
            metrics file's own `topics` field only describes the winning
            trial's real BERTopic/zero-shot result, a non-`None` value here
            plots placeholder topics (built fresh from the replayed trial's
            own cluster sizes) through the same LLM relabeling pass instead
            of reusing the metrics file's topics.

    Returns:
        The path the scatterplot was written to.

    Raises:
        ValueError: If `method` isn't a recognized key in `VISUALIZATION_SPECS`.
        FileNotFoundError: If `metrics_path` or the replayed trial's Optuna
            study journal doesn't exist.
        RuntimeError: If the replayed trial fails to re-evaluate to status
            `"ok"` (see `refit_winning_trial`).
    """
    if method not in VISUALIZATION_SPECS:
        raise ValueError(f"Unknown method {method!r}; expected one of {sorted(VISUALIZATION_SPECS)}")
    spec = VISUALIZATION_SPECS[method]
    label = label_from_slug(slug_from_filename(embeddings_path.name, "_trials_embeddings.parquet"))
    resolved_config_path = config_path if config_path is not None else spec.default_config_path

    metrics = load_metrics(spec, metrics_path)

    table = clustering_io.read_embeddings_parquet(embeddings_path)
    filtered = clustering_io.filter_successful_rows(table)
    embedding_matrix = clustering_io.extract_embedding_matrix(filtered)
    traceability = clustering_io.extract_traceability_columns(filtered)

    labels, probabilities = refit_winning_trial(
        spec, metrics, embeddings_path, embedding_matrix, resolved_config_path, trial_number_override=trial_number
    )
    assignments_table = pa.table(
        {
            "nct_id": traceability["nct_id"],
            "criterion_text": traceability["criterion_text"],
            "cluster_id": [int(x) for x in labels],
            "cluster_probability": [float(x) for x in probabilities],
        }
    )

    if seed_override is not None:
        seed, seed_explicit = seed_override, True
        logger.info("run: using CLI-provided --seed=%d override for %s", seed_override, method)
    else:
        seed, seed_explicit = resolve_seed(spec, metrics)

    if trial_number is None:
        topics = build_topics_from_metrics(metrics)
    else:
        # The metrics file's own `topics` describes the *winning* trial's
        # BERTopic result, which doesn't apply to an arbitrary overridden
        # trial -- build fresh placeholder topics from this trial's own
        # cluster sizes instead, the same way `generate_cluster_labels.py`
        # does for an explicit --trial-number.
        cluster_sizes = clustering_io.compute_cluster_sizes(labels)
        topics = [
            TopicRepresentation(
                topic_id=topic_id, keywords=[], label="Noise" if topic_id == -1 else "",
                count=cluster_sizes[str(topic_id)],
            )
            for topic_id in sorted({int(x) for x in labels})
        ]
    guide_labels = metrics.get("candidate_labels", [])
    topic_labels_by_id = resolve_topic_labels(
        topics, guide_labels, assignments_table, llm_model, primary_summary_prompt_path, secondary_summary_prompt_path
    )
    points_2d = resolve_points(embedding_matrix, seed)
    frame = build_plot_frame(assignments_table, points_2d, topic_labels_by_id)
    fig = render_scatterplot(frame)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)

    logger.info(
        "run: wrote %s scatterplot for %r (n_points=%d, seed=%d, seed_explicit=%s) -> %s",
        method, label, assignments_table.num_rows, seed, seed_explicit, output_path,
    )
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for this module.

    Returns:
        An ``argparse.ArgumentParser`` accepting a required ``--method``
        choice, required metrics/embeddings/output path flags, an optional
        ``--config-path``/seed/``--trial-number`` override, and the
        two-stage LLM relabeling pass's own flags (``--llm-model``/
        ``--primary-summary-prompt-path``/``--secondary-summary-prompt-path``).
    """
    parser = argparse.ArgumentParser(
        description="Render an interactive 2D UMAP scatterplot of a reducer+HDBSCAN method's trial."
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=sorted(VISUALIZATION_SPECS),
        help="Which clustering method's output to visualize",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        required=True,
        help="That method's metrics JSON path",
    )
    parser.add_argument(
        "--embeddings-path",
        type=Path,
        required=True,
        help="The same embeddings Parquet originally passed to that method's own --input-path",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Output HTML path",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=None,
        help="Optuna search config YAML used to re-fit the replayed trial; defaults to that method's "
        "own reducer's default config",
    )
    parser.add_argument(
        "--trial-number",
        type=int,
        default=None,
        help="Re-fit and plot this specific Optuna trial instead of --metrics-path's own recorded "
        "winner (see verify_trial_reproducibility.py for how to find one); its clusters are LLM-labeled "
        "fresh rather than reusing --metrics-path's own (winner-only) topics",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the 2D UMAP projection's seed (default: reuse that method's own optimal-run seed)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=llm_labeling.DEFAULT_LLM_MODEL,
        help="Anthropic model id for this method's two-stage LLM cluster relabeling pass, run here "
        f"for display only (default: {llm_labeling.DEFAULT_LLM_MODEL}); requires ANTHROPIC_API_KEY "
        "in .env",
    )
    parser.add_argument(
        "--primary-summary-prompt-path",
        type=Path,
        default=llm_labeling.DEFAULT_PRIMARY_PROMPT_PATH,
        help="Text file containing the system prompt for primary summarization (one LLM call per "
        f"cluster, over its own top-{llm_labeling.TOP_N_CRITERIA_PER_CLUSTER} criteria by "
        f"cluster_probability); default: {llm_labeling.DEFAULT_PRIMARY_PROMPT_PATH}",
    )
    parser.add_argument(
        "--secondary-summary-prompt-path",
        type=Path,
        default=llm_labeling.DEFAULT_SECONDARY_PROMPT_PATH,
        help="Text file containing the system prompt for secondary summarization (one LLM call "
        f"across every cluster's primary label, for semantic distinctness); default: "
        f"{llm_labeling.DEFAULT_SECONDARY_PROMPT_PATH}",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="If given, also append logs to this file (in addition to the console)",
    )
    return parser


def main() -> None:
    """CLI entry point: render and write a scatterplot for one of a method's trials."""
    args = build_arg_parser().parse_args()
    configure_logging(args.log_file)
    run(
        args.method,
        args.metrics_path,
        args.embeddings_path,
        args.output_path,
        args.llm_model,
        args.primary_summary_prompt_path,
        args.secondary_summary_prompt_path,
        config_path=args.config_path,
        seed_override=args.seed,
        trial_number=args.trial_number,
    )


if __name__ == "__main__":
    main()
