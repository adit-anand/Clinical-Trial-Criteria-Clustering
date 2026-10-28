"""Generic "reduce, then cluster with BERTopic-wrapped HDBSCAN, tuned by Optuna" orchestration.

`umap_hdbscan_clustering.py` and `pacmap_hdbscan_clustering.py` both reduce a
condition's embeddings with their own reducer before clustering the reduced
vectors, tuning both jointly via one Optuna search. Everything about that
orchestration — reading/filtering input, building the Optuna objective,
resolving the winning trial, writing the three output artifacts — is
identical between them; the only real differences are reducer-specific
(hyperparameter names, the fit callable, the fixed-metric config key, and a
couple of output field names). This module holds that shared orchestration
once, parametrized by a `ReducerPlugin` describing exactly those differences,
so each reducer module is a thin shell: its own hyperparameter
validation/fit function plus a `ReducerPlugin` instance and CLI entry point.

Each Optuna trial's seed/repeat reproducibility sweep (`reduction_sweep.py`)
is unchanged — it still picks the most reproducible seed and a representative
repeat via a plain HDBSCAN fit. BERTopic is layered on *after* that
selection, fit exactly once per trial on the representative repeat's already-
reduced embeddings, reusing that trial's own HDBSCAN hyperparameters (see
`bertopic_model.fit_bertopic_topics`). BERTopic's own resulting cluster
assignments — not the sweep's earlier plain-HDBSCAN fit — are what the
Optuna objective scores and what gets written for the winning trial, per
`_specs/bertopic-clustering.md`.

Nothing here branches on which reducer is in use outside of reading from
`ReducerPlugin` — adding a further reducer means adding one new sibling
module with its own plugin, not touching this one.
"""

import argparse
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import optuna
import pyarrow as pa

from src.logging_utils import configure_logging
from src.paths import label_from_slug, slug_from_filename
from src.clustering import bertopic_model
from src.clustering import clustering_io
from src.clustering import optuna_search
from src.clustering import reduction_sweep
from src.clustering.hdbscan_model import (
    HDBSCAN_CLUSTER_PARQUET_SCHEMA,
    HDBSCAN_PARAM_NAMES,
    HDBSCANClusterer,
    compute_relative_validity_index,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReducerPlugin:
    """Everything that varies between one reducer's pipeline and another's.

    Attributes:
        name: Short identifier, folded into the Optuna study name (e.g.
            `"umap"` -> study name suffix `"_umap_hdbscan"`).
        display_name: Human-readable name used in log messages and the
            dimensionality-reduction note (e.g. `"UMAP"`).
        reduced_embedding_schema: This reducer's output Parquet schema (same
            shape as every reducer's, just defined in its own module).
        fixed_metric_config_key: The key into `OptunaSearchConfig.fixed` this
            reducer's own distance metric is configured under (e.g.
            `"metric"` for UMAP, `"distance"` for PaCMAP).
        winning_config_field: The metrics-JSON field name this reducer's
            winning hyperparameters are reported under (e.g.
            `"winning_umap_config"`).
        fixed_metric_field: The metrics-JSON field name this reducer's fixed
            metric value is reported under (e.g. `"umap_fixed_metric"`).
        make_fit_reduced: `(metric) -> fit_reduced`, binding this reducer's
            fixed distance metric into a `reduction_sweep`-compatible
            callable (see each reducer module's own `make_fit_*_reduced`).
        validate_hyperparameters: `(hyperparams, n_samples) -> None`, this
            reducer's own precondition check (see each reducer module's own
            `validate_*_hyperparameters`).
    """

    name: str
    display_name: str
    reduced_embedding_schema: pa.Schema
    fixed_metric_config_key: str
    winning_config_field: str
    fixed_metric_field: str
    make_fit_reduced: Callable[[str], Callable[..., np.ndarray]]
    validate_hyperparameters: Callable[[dict, int], None]


def build_reduced_embedding_rows(table: pa.Table, reduced_matrix: np.ndarray, reduced_at: datetime) -> list[dict]:
    """Zip each surviving row's traceability columns with its reduced vector.

    Args:
        table: The table `reduced_matrix` was produced from (row order must match).
        reduced_matrix: The reducer's output, shape `(table.num_rows, n_components)`.
        reduced_at: Wall-clock time this reduction run completed.

    Returns:
        One output dict per row, matching a `ReducerPlugin.reduced_embedding_schema`'s columns.
    """
    columns = clustering_io.extract_traceability_columns(table)
    rows = []
    for i in range(table.num_rows):
        rows.append(
            {
                "nct_id": columns["nct_id"][i],
                "study_first_posted_date": columns["study_first_posted_date"][i],
                "criterion_text": columns["criterion_text"][i],
                "criterion_type": columns["criterion_type"][i],
                "reduced_embedding": [float(x) for x in reduced_matrix[i]],
                "embedding_backend": columns["embedding_backend"][i],
                "embedding_model": columns["embedding_model"][i],
                "reduced_at": reduced_at,
            }
        )
    return rows


def annotate_dimensionality_note(
    candidate: reduction_sweep.ReducerConfigCandidate, embedding_dim: int, display_name: str
) -> None:
    """Append a non-fatal note to the winning candidate if `n_components` doesn't meaningfully reduce dimensionality.

    Per the spec, `n_components >= embedding_dim` is not an error — just
    something to surface.

    Args:
        candidate: The winning trial's candidate, mutated in place.
        embedding_dim: The original, non-reduced embedding dimensionality.
        display_name: The reducer's `ReducerPlugin.display_name`, for the note's text.
    """
    n_components = candidate.hyperparams["n_components"]
    if n_components >= embedding_dim:
        candidate.notes.append(
            f"n_components={n_components} is >= the original embedding_dim={embedding_dim}; this is "
            f"not an error, but {display_name}'s dimensionality reduction is not meaningfully reducing "
            "dimensionality in this configuration"
        )


def build_metrics_dict(
    plugin: ReducerPlugin,
    label: str,
    criterion_type: str,
    input_path: Path,
    reduced_embeddings_output_path: Path,
    assignments_output_path: Path,
    n_samples: int,
    n_samples_excluded: int,
    embedding_dim: int,
    embedding_backend: str | None,
    embedding_model: str | None,
    candidate: reduction_sweep.ReducerConfigCandidate,
    metrics_result: dict,
    relative_validity: float | None,
    bertopic_result: bertopic_model.BERTopicFitResult,
    candidate_labels: list[str],
    zeroshot_model: str,
    seeds: list[int],
    repeats_per_seed: int,
    fixed_metric: str,
    clustered_at: datetime,
) -> dict:
    """Assemble the full metrics JSON payload for one reducer+BERTopic run.

    Args:
        plugin: This reducer's `ReducerPlugin`.
        label: Human-readable label for this run, derived from `input_path`'s filename.
        criterion_type: Which side of the inclusion/exclusion split this run covers.
        input_path: The embeddings Parquet path that was read.
        reduced_embeddings_output_path: Where the reduced embeddings were written.
        assignments_output_path: Where cluster assignments were written.
        n_samples: Number of successfully-embedded rows reduced/clustered.
        n_samples_excluded: Number of input rows dropped for having failed embeddings.
        embedding_dim: Original (non-reduced) embedding dimensionality.
        embedding_backend: Backend recorded on the clustered rows.
        embedding_model: Model recorded on the clustered rows.
        candidate: The winning trial's `ReducerConfigCandidate` (from the
            seed/repeat reproducibility sweep — its own `.notes` are
            provenance about the reducer's chosen seed, independent of BERTopic).
        metrics_result: `HDBSCANClusterer.compute_metrics`'s return value,
            computed from `bertopic_result.labels` (not the sweep's own labels).
        relative_validity: The DBCV score, computed from `bertopic_result.labels`.
        bertopic_result: The winning trial's `bertopic_model.BERTopicFitResult`.
        candidate_labels: The `--candidate-labels` this run was configured with.
        zeroshot_model: The `--zeroshot-model` this run was configured with.
        seeds: The configured reproducibility seed list.
        repeats_per_seed: Number of repeats run per seed.
        fixed_metric: This reducer's fixed distance metric value.
        clustered_at: Wall-clock time this run completed.

    Returns:
        A JSON-serializable dict.
    """
    base = clustering_io.build_base_metrics_dict(
        label=label,
        criterion_type=criterion_type,
        input_path=input_path,
        assignments_output_path=assignments_output_path,
        n_samples=n_samples,
        n_samples_excluded=n_samples_excluded,
        embedding_dim=embedding_dim,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
        cluster_sizes=clustering_io.compute_cluster_sizes(bertopic_result.labels),
        clustered_at=clustered_at,
    )
    return {
        **base,
        "reduced_embeddings_output_path": clustering_io.relative_to_repo_root(reduced_embeddings_output_path),
        "n_clusters_actual": metrics_result["n_clusters_actual"],
        "noise_count": metrics_result["noise_count"],
        "noise_fraction": metrics_result["noise_fraction"],
        "metrics": metrics_result["metrics"],
        "metrics_valid": metrics_result["metrics_valid"],
        "relative_validity_index": relative_validity,
        "notes": metrics_result["notes"] + candidate.notes,
        "topics": [
            {"topic_id": topic.topic_id, "keywords": topic.keywords, "label": topic.label, "count": topic.count}
            for topic in bertopic_result.topics
        ],
        "candidate_labels": candidate_labels,
        "zeroshot_model": zeroshot_model,
        plugin.winning_config_field: {k: v for k, v in candidate.hyperparams.items() if k not in HDBSCAN_PARAM_NAMES},
        "winning_seed": candidate.winning_seed.seed,
        "winning_seed_noise_fraction_std": candidate.winning_seed.noise_fraction_std,
        plugin.fixed_metric_field: fixed_metric,
        "reproducibility_seeds": seeds,
        "reproducibility_repeats_per_seed": repeats_per_seed,
    }


def run(
    plugin: ReducerPlugin,
    input_path: Path,
    criterion_type: str,
    reduced_embeddings_output_path: Path,
    assignments_output_path: Path,
    metrics_output_path: Path,
    config_path: Path,
    candidate_labels: list[str],
    zeroshot_model: str,
) -> tuple[Path, Path, Path] | None:
    """Run the full read-filter-search-write pipeline for one reducer.

    Args:
        plugin: This reducer's `ReducerPlugin`.
        input_path: Embeddings Parquet to read.
        criterion_type: `"inclusion"` or `"exclusion"` — which side of the
            inclusion/exclusion split to cluster (see
            `clustering_io.filter_by_criterion_type`); also folded into this
            run's slug/label so its Optuna study and trial-history export
            never collide with the other side's run over the same input file.
        reduced_embeddings_output_path: Where to write reduced embeddings.
        assignments_output_path: Where to write cluster assignments.
        metrics_output_path: Where to write metrics.
        config_path: YAML config (combined reducer + HDBSCAN search space,
            objective weights, trial budget, reproducibility seeds/repeats,
            study/report directories) — see `optuna_search.load_optuna_config`.
        candidate_labels: Candidate topic labels for BERTopic's zero-shot
            classification layer (see `bertopic_model.parse_candidate_labels`).
        zeroshot_model: HuggingFace zero-shot-classification model id for
            that same layer.

    Returns:
        `(reduced_embeddings_output_path, assignments_output_path,
        metrics_output_path)`, or `None` if every trial in the study was
        invalid (nothing ever fit — nothing is written in that case).

    Raises:
        FileNotFoundError: If the input Parquet or `config_path` does not exist.
        ValueError: If there are no successfully-embedded rows to cluster,
            if embedding dimensionality is inconsistent, or if `config_path`
            fails validation (see `optuna_search.load_optuna_config`).
    """
    slug = f"{slug_from_filename(input_path.name, '_trials_embeddings.parquet')}_{criterion_type}"
    label = label_from_slug(slug)

    table = clustering_io.read_embeddings_parquet(input_path)
    table = clustering_io.filter_by_criterion_type(table, criterion_type)
    total_rows = table.num_rows
    filtered = clustering_io.filter_successful_rows(table)
    n_samples = filtered.num_rows
    n_excluded = total_rows - n_samples

    embedding_dim = clustering_io.validate_embedding_dimensions(filtered)
    embedding_matrix = clustering_io.extract_embedding_matrix(filtered)
    criterion_texts = clustering_io.extract_traceability_columns(filtered)["criterion_text"]
    # Built once per run (not once per trial): criterion_texts never changes
    # between trials, so this is `evaluate_trial`'s shared, loop-invariant
    # input to the BERTopic topic-coherence term (see
    # optuna_search.compute_bertopic_quality_metrics).
    document_term_index = optuna_search.build_document_term_index(criterion_texts)

    config = optuna_search.load_optuna_config(config_path, require_reproducibility=True)
    bounds = optuna_search.EligibilityBounds(
        min_n_clusters=HDBSCANClusterer.MIN_N_CLUSTERS,
        max_n_clusters=HDBSCANClusterer.MAX_N_CLUSTERS,
        max_cluster_share=HDBSCANClusterer.MAX_CLUSTER_SHARE,
    )
    fixed_metric = config.fixed[plugin.fixed_metric_config_key]
    fit_reduced = plugin.make_fit_reduced(fixed_metric)

    reducer_space = {name: spec for name, spec in config.search_space.items() if name not in HDBSCAN_PARAM_NAMES}
    hdbscan_space = {name: spec for name, spec in config.search_space.items() if name in HDBSCAN_PARAM_NAMES}

    def build_hdbscan_clusterer(hyperparams: dict) -> HDBSCANClusterer:
        return HDBSCANClusterer(
            min_cluster_size=hyperparams["min_cluster_size"],
            min_samples=hyperparams["min_samples"],
            metric=fixed_metric,
            cluster_selection_epsilon=config.fixed["cluster_selection_epsilon"],
            cluster_selection_method=hyperparams["cluster_selection_method"],
            alpha=config.fixed["alpha"],
        )

    def evaluate_trial(hyperparams: dict) -> dict:
        """Evaluate one combined reducer+HDBSCAN hyperparameter combination into a trial-cache-shaped record.

        Pulled out of the objective closure so the post-`study.optimize` winner
        resolution can call it too: a resumed study's `study.best_trial` may be
        from a *previous* process invocation, never present in this process's
        own `trial_cache` (Optuna's storage persists only hyperparameters/score,
        not our fit artifacts) — re-evaluating is the only way to recover its
        reduced matrix/labels/probabilities/topics/dr_metrics/bertopic_quality,
        and is safe since the reducer (given the same `random_state`),
        HDBSCAN, and BERTopic's own refit are all fully deterministic given
        the same inputs.

        The seed/repeat reproducibility sweep below still picks a winning
        seed/representative repeat via a plain HDBSCAN fit, unchanged; once
        that representative repeat is chosen, BERTopic is fit exactly once
        on it, and its own cluster assignments (not the sweep's) are what
        get scored and (for the eventual winning trial) written.
        """
        candidate = reduction_sweep.evaluate_config(
            fit_reduced,
            hyperparams,
            config.reproducibility.seeds,
            config.reproducibility.repeats_per_seed,
            plugin.validate_hyperparameters,
            build_hdbscan_clusterer,
            embedding_matrix,
            n_samples,
        )
        if candidate.status == "invalid":
            return {"status": "invalid", "candidate": candidate, "hyperparams": hyperparams}

        representative = reduction_sweep.representative_repeat(candidate)
        dr_metrics = optuna_search.compute_dr_metrics(
            embedding_matrix, representative.reduced_matrix, representative.labels, fixed_metric
        )

        hdbscan_clusterer = build_hdbscan_clusterer(hyperparams)
        try:
            bertopic_result = bertopic_model.fit_bertopic_topics(
                criterion_texts,
                representative.reduced_matrix,
                hdbscan_clusterer.build_estimator(),
                candidate_labels,
                zeroshot_model,
            )
        except Exception:
            logger.exception("evaluate_trial: BERTopic fit failed for hyperparams=%s", hyperparams)
            return {"status": "invalid", "candidate": candidate, "hyperparams": hyperparams}

        if not bertopic_model.same_partition(representative.labels, bertopic_result.labels):
            logger.warning(
                "evaluate_trial: BERTopic's own refit clustering disagrees with the seed/repeat sweep's "
                "plain-HDBSCAN fit for hyperparams=%s (expected a relabeling of the same partition) -- "
                "scoring and writing use BERTopic's own labels regardless, per design",
                hyperparams,
            )

        bertopic_quality = optuna_search.compute_bertopic_quality_metrics(bertopic_result, document_term_index)

        metrics_result = hdbscan_clusterer.compute_metrics(representative.reduced_matrix, bertopic_result.labels)
        relative_validity = compute_relative_validity_index(
            representative.reduced_matrix, bertopic_result.labels, fixed_metric
        )
        status = "ok" if metrics_result["metrics_valid"] else "degenerate"

        return {
            "status": status,
            "candidate": candidate,
            "dr_metrics": dr_metrics,
            "bertopic_quality": bertopic_quality,
            "hyperparams": hyperparams,
            "metrics_result": metrics_result,
            "relative_validity": relative_validity,
            "bertopic_result": bertopic_result,
        }

    study_name = f"{slug}_{plugin.name}_hdbscan"
    storage_path = config.study_dir / f"{study_name}_study.journal"
    study = optuna_search.build_study(storage_path, study_name, config.sampler_seed)

    trial_cache: dict[int, dict] = {}

    def objective(trial: optuna.Trial) -> float:
        hyperparams = {
            **optuna_search.suggest_hyperparams(trial, reducer_space),
            **optuna_search.suggest_hyperparams(trial, hdbscan_space),
        }
        record = evaluate_trial(hyperparams)
        reduction_sweep.log_config_candidate(trial.number, record["candidate"])
        trial_cache[trial.number] = record
        # Persisted to the study's storage (unlike trial_cache, which is this
        # process's own memory) so count_trial_statuses reflects every trial
        # across every resumed invocation, not just this one.
        trial.set_user_attr("status", record["status"])

        if record["status"] == "invalid":
            return optuna_search.INVALID_TRIAL_SCORE

        return optuna_search.score_reduction_candidate(
            record["metrics_result"],
            record["relative_validity"],
            record["bertopic_result"].labels,
            hyperparams["min_cluster_size"],
            config.objective_weights,
            record["dr_metrics"],
            record["bertopic_quality"],
            bounds,
        )

    n_remaining = optuna_search.remaining_trials(study, config.n_trials)
    if n_remaining > 0:
        logger.info("run: running %d new trial(s) for %r (total budget=%d)", n_remaining, label, config.n_trials)
        study.optimize(objective, n_trials=n_remaining)
    else:
        logger.info("run: study %r already has >= %d trial(s); skipping new trials", study_name, config.n_trials)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trial_history_path = config.results_dir / f"{study_name}_optuna_trials_{timestamp}.csv"
    optuna_search.export_trial_history(study, trial_history_path)

    status_counts = optuna_search.count_trial_statuses(study)

    # `best_ok_trial_or_none` (not `best_trial_or_none`) deliberately: a
    # degenerate trial (dominant cluster / excess noise / cluster-count out
    # of range) must never be written out as this run's result just because
    # it out-scored every eligible trial -- see that function's docstring.
    best_trial = optuna_search.best_ok_trial_or_none(study)
    if best_trial is None:
        if not study.get_trials(deepcopy=False):
            logger.warning("run: study has no trials for %r; writing nothing", label)
        else:
            logger.warning(
                "run: no trial produced a valid (non-degenerate) clustering for %r (%d trial(s) attempted: "
                "%d ok, %d degenerate, %d invalid); writing nothing",
                label, len(study.get_trials(deepcopy=False)),
                status_counts["ok"], status_counts["degenerate"], status_counts["invalid"],
            )
        return None
    winner = trial_cache.get(best_trial.number)
    if winner is None:
        logger.info(
            "run: best trial #%d for %r predates this process; re-evaluating its hyperparameters",
            best_trial.number, label,
        )
        winner = evaluate_trial(best_trial.params)
    if winner["status"] != "ok":
        # Defensive only: `evaluate_trial` is deterministic given the same
        # hyperparameters, so a trial persisted with status "ok" should
        # always re-evaluate to "ok" too. Guards against ever writing a
        # degenerate/invalid result if that assumption is ever violated.
        logger.warning(
            "run: best 'ok' trial #%d for %r re-evaluated to status=%r instead; writing nothing",
            best_trial.number, label, winner["status"],
        )
        return None

    candidate: reduction_sweep.ReducerConfigCandidate = winner["candidate"]
    annotate_dimensionality_note(candidate, embedding_dim, plugin.display_name)
    representative = reduction_sweep.representative_repeat(candidate)
    bertopic_result: bertopic_model.BERTopicFitResult = winner["bertopic_result"]
    hdbscan_clusterer = build_hdbscan_clusterer(winner["hyperparams"])

    clustered_at = datetime.now(timezone.utc)
    reduced_rows = build_reduced_embedding_rows(filtered, representative.reduced_matrix, clustered_at)
    clustering_io.write_parquet(reduced_rows, plugin.reduced_embedding_schema, reduced_embeddings_output_path)

    assignment_rows = hdbscan_clusterer.build_assignment_rows(
        filtered, bertopic_result.labels, bertopic_result.probabilities, bertopic_result.topics, clustered_at
    )
    clustering_io.write_parquet(assignment_rows, HDBSCAN_CLUSTER_PARQUET_SCHEMA, assignments_output_path)

    embedding_backend = filtered.column("embedding_backend")[0].as_py()
    embedding_model = filtered.column("embedding_model")[0].as_py()
    metrics_dict = build_metrics_dict(
        plugin,
        label=label,
        criterion_type=criterion_type,
        input_path=input_path,
        reduced_embeddings_output_path=reduced_embeddings_output_path,
        assignments_output_path=assignments_output_path,
        n_samples=n_samples,
        n_samples_excluded=n_excluded,
        embedding_dim=embedding_dim,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
        candidate=candidate,
        metrics_result=winner["metrics_result"],
        relative_validity=winner["relative_validity"],
        bertopic_result=bertopic_result,
        candidate_labels=candidate_labels,
        zeroshot_model=zeroshot_model,
        seeds=config.reproducibility.seeds,
        repeats_per_seed=config.reproducibility.repeats_per_seed,
        fixed_metric=fixed_metric,
        clustered_at=clustered_at,
    )
    dr_metrics = winner["dr_metrics"]
    metrics_dict["dr_metrics"] = asdict(dr_metrics) if dr_metrics is not None else None
    bertopic_quality = winner["bertopic_quality"]
    metrics_dict["bertopic_quality_metrics"] = asdict(bertopic_quality) if bertopic_quality is not None else None
    metrics_dict["optuna_study_summary"] = {
        "study_name": study_name,
        "storage_path": clustering_io.relative_to_repo_root(storage_path),
        "trial_history_path": clustering_io.relative_to_repo_root(trial_history_path),
        "n_trials_total_in_study": len(study.get_trials(deepcopy=False)),
        "n_trials_ok": status_counts["ok"],
        "n_trials_degenerate": status_counts["degenerate"],
        "n_trials_invalid": status_counts["invalid"],
        "best_trial_number": best_trial.number,
        "best_score": best_trial.value,
    }
    metrics_dict["objective_weights"] = asdict(config.objective_weights)
    clustering_io.write_json(metrics_dict, metrics_output_path)

    logger.info(
        "run: best trial #%d %s config %s (seed=%d) clustered %d row(s) into %d topic(s) for %r -> %s, %s, %s "
        "(noise=%d/%d (%.1f%%), silhouette=%s, excluded=%d failed-embedding row(s), "
        "trials: %d ok, %d degenerate, %d invalid)",
        best_trial.number, plugin.display_name, candidate.hyperparams, candidate.winning_seed.seed, n_samples,
        winner["metrics_result"]["n_clusters_actual"], label, reduced_embeddings_output_path,
        assignments_output_path, metrics_output_path, winner["metrics_result"]["noise_count"], n_samples,
        winner["metrics_result"]["noise_fraction"] * 100, winner["metrics_result"]["metrics"]["silhouette_score"],
        n_excluded, status_counts["ok"], status_counts["degenerate"], status_counts["invalid"],
    )
    return reduced_embeddings_output_path, assignments_output_path, metrics_output_path


def build_arg_parser(description: str, default_config_path: Path) -> argparse.ArgumentParser:
    """Build the shared CLI argument parser used by every reducer's pipeline module.

    Args:
        description: This reducer module's own CLI `--help` description.
        default_config_path: This reducer's own default `--config-path`.

    Returns:
        An ``argparse.ArgumentParser`` accepting required input/output path
        flags, a `--config-path` override for the Optuna search config, and
        the BERTopic zero-shot-classification layer's own flags.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Input embeddings Parquet path (as written by criteria_embedding.py)",
    )
    parser.add_argument(
        "--criterion-type",
        type=str,
        choices=["inclusion", "exclusion"],
        required=True,
        help=(
            "Which side of eligibility criteria to cluster. Rows with criterion_type='both' "
            "(unstructured eligibility text with no inclusion/exclusion headers to split on) are "
            "included on both sides. Give each side's outputs distinct paths/filenames."
        ),
    )
    parser.add_argument(
        "--reduced-embeddings-output-path",
        type=Path,
        required=True,
        help="Output reduced-embeddings Parquet path",
    )
    parser.add_argument(
        "--assignments-output-path",
        type=Path,
        required=True,
        help="Output cluster-assignments Parquet path",
    )
    parser.add_argument(
        "--metrics-output-path",
        type=Path,
        required=True,
        help="Output metrics JSON path",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=default_config_path,
        help=f"Optuna search config YAML; default: {default_config_path}",
    )
    parser.add_argument(
        "--candidate-labels",
        type=str,
        required=True,
        help="Comma-separated candidate topic labels for BERTopic's zero-shot classification layer "
        "(e.g. 'oncology eligibility,treatment history,disease measurement')",
    )
    parser.add_argument(
        "--zeroshot-model",
        type=str,
        default=bertopic_model.DEFAULT_ZEROSHOT_MODEL,
        help="HuggingFace zero-shot-classification model id for BERTopic's topic-labeling layer "
        f"(default: {bertopic_model.DEFAULT_ZEROSHOT_MODEL})",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="If given, also append logs to this file (in addition to the console)",
    )
    return parser


def main(plugin: ReducerPlugin, description: str, default_config_path: Path) -> None:
    """Shared CLI entry point: run the Optuna search and write reduced embeddings/cluster assignments/metrics."""
    args = build_arg_parser(description, default_config_path).parse_args()
    configure_logging(args.log_file)
    candidate_labels = bertopic_model.parse_candidate_labels(args.candidate_labels)

    run(
        plugin,
        args.input_path,
        args.criterion_type,
        args.reduced_embeddings_output_path,
        args.assignments_output_path,
        args.metrics_output_path,
        args.config_path,
        candidate_labels,
        args.zeroshot_model,
    )
