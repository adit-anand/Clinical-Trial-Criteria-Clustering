"""Reduce criterion embeddings with PaCMAP, then cluster via HDBSCAN, tuned by an Optuna search.

Mirrors `umap_hdbscan_clustering.py` exactly, swapping UMAP for PaCMAP as the
reducer. Reads the same embeddings Parquet file format `criteria_embedding.py`
writes, drops rows whose embedding failed, then runs an Optuna study whose
every trial suggests a combined hyperparameter set — PaCMAP's own
(`n_components`/`n_neighbors`/`MN_ratio`/`FP_ratio`) *and* HDBSCAN's own —
from the search space defined in a `--config-path` YAML file. PaCMAP has no
`min_dist` equivalent to UMAP's — its neighborhood/distance behavior is
instead shaped by `MN_ratio`/`FP_ratio`, which weight mid-near vs. further
point pairs during optimization. Each trial internally runs the configured
`reproducibility.seeds` x `repeats_per_seed` grid via `reduction_sweep.
evaluate_config` (unchanged reducer-agnostic machinery) so the most
*reproducible* seed can be selected before the trial is scored.

Each trial's composite objective score (`optuna_search.score_reduction_candidate`:
DBCV plus trustworthiness/continuity/steadiness/cohesiveness — computed via
`zadu`, using cosine distance per the fixed `distance` config value — folded
with the same noise/dominant-cluster/cluster-count/micro-cluster penalties
this module shares with `umap_hdbscan_clustering.py` via `hdbscan_model.py`) both drives Optuna's own trial selection and
determines which trial's result gets written. The study is resumable, and a
full per-trial history CSV is written under `data/clustering/reports/` on
every run — see `optuna_search.py`.

The run writes three output artifacts, at the paths given via
`--reduced-embeddings-output-path`/`--assignments-output-path`/
`--metrics-output-path`.

Re-running this step against the same output paths overwrites all three
existing output files (and the trial-history CSV) wholesale; only the
Optuna study's own file-based storage accumulates across runs.
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import optuna
import pacmap
import pyarrow as pa

# Allow running this file directly (e.g. `uv run src/clustering/pacmap_hdbscan_clustering.py`)
# as well as via `-m` (e.g. `uv run python -m src.clustering.pacmap_hdbscan_clustering`).
# Direct execution leaves __package__ unset and never puts the repo root on
# sys.path, so the src.* imports below would otherwise fail with
# "ModuleNotFoundError: No module named 'src'"; -m already sets both up
# correctly, so this is a no-op in that case.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataclasses import asdict

from src.logging_utils import configure_logging
from src.paths import REPO_ROOT, label_from_slug, slug_from_filename
from src.clustering import clustering_io
from src.clustering import optuna_search
from src.clustering import reduction_sweep
from src.clustering.hdbscan_model import HDBSCAN_CLUSTER_PARQUET_SCHEMA, HDBSCAN_PARAM_NAMES, HDBSCANClusterer

logger = logging.getLogger(__name__)

PACMAP_REDUCED_EMBEDDING_PARQUET_SCHEMA = pa.schema(
    [
        ("nct_id", pa.string()),
        ("study_first_posted_date", pa.string()),
        ("criterion_text", pa.string()),
        ("criterion_type", pa.string()),
        ("reduced_embedding", pa.list_(pa.float32())),
        ("embedding_backend", pa.string()),
        ("embedding_model", pa.string()),
        ("reduced_at", pa.timestamp("us", tz="UTC")),
    ]
)


# PaCMAP's own `distance` parameter only accepts {"euclidean", "manhattan",
# "angular", "dot", "hamming"} — it has no "cosine" spelling of its own,
# calling the same cosine-like metric "angular" instead. That "angular" path
# (faiss.IndexHNSWFlat with METRIC_INNER_PRODUCT, inside pacmap's own
# compute_nearest_neighbors) has a confirmed numerical bug in the installed
# pacmap version: at n_neighbors >= 30 it reliably raises "RuntimeWarning:
# overflow encountered in multiply" in its `sqrt(2 * (1 - D))` cosine-
# distance conversion, and can crash a run outright depending on the data —
# reproduced directly against this pipeline's own embeddings. Since the
# embedding backends this pipeline reads from (`bert`/`huggingface`) already
# emit L2-normalized (unit-norm) vectors, Euclidean distance between them is
# a monotonic function of cosine distance (`||a-b||^2 = 2*(1-cos_sim(a,b))`)
# — mathematically equivalent for ranking/reduction purposes, not an
# approximation. So instead of translating "cosine" to pacmap's "angular",
# this module normalizes the input itself and fits with "euclidean",
# sidestepping the buggy "angular" code path entirely. `metric`/`distance`
# stays "cosine" everywhere else in this module (HDBSCAN's own `metric`,
# DBCV, and `optuna_search.compute_dr_metrics`'s `metric == "cosine"` check
# all accept that spelling natively, and have no equivalent bug) — only the
# call into `pacmap.PaCMAP` itself is affected.
_PACMAP_NATIVE_DISTANCES = {"euclidean", "manhattan", "angular", "dot", "hamming"}


def make_fit_pacmap_reduced(distance: str) -> Callable[..., np.ndarray]:
    """Bind PaCMAP's fixed (non-swept) distance metric into a `reduction_sweep`-compatible callable.

    Args:
        distance: The distance metric held fixed across the whole search.
            `"cosine"` normalizes the input and fits with `"euclidean"`
            instead of PaCMAP's own buggy `"angular"` (see
            `_PACMAP_NATIVE_DISTANCES` above); any of PaCMAP's own native
            distance names is passed through unchanged.

    Returns:
        A callable `(embedding_matrix, seed, n_components, n_neighbors,
        mn_ratio, fp_ratio) -> reduced_matrix`, matching the shape
        `reduction_sweep.evaluate_repeat` expects for `fit_reduced`.

    Raises:
        ValueError: If `distance` is neither `"cosine"` nor one of PaCMAP's
            own native distance names.
    """
    if distance == "cosine":
        normalize_before_fit = True
        pacmap_distance = "euclidean"
    elif distance in _PACMAP_NATIVE_DISTANCES:
        normalize_before_fit = False
        pacmap_distance = distance
    else:
        raise ValueError(
            f"Unknown PaCMAP distance {distance!r}; expected 'cosine' or one of {sorted(_PACMAP_NATIVE_DISTANCES)}"
        )

    def fit_reduced(
        embedding_matrix: np.ndarray, seed: int, n_components: int, n_neighbors: int, mn_ratio: float, fp_ratio: float
    ) -> np.ndarray:
        matrix = optuna_search.l2_normalize_rows(embedding_matrix) if normalize_before_fit else embedding_matrix
        reducer = pacmap.PaCMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            MN_ratio=mn_ratio,
            FP_ratio=fp_ratio,
            distance=pacmap_distance,
            random_state=seed,
        )
        return reducer.fit_transform(matrix)

    return fit_reduced


def validate_pacmap_hyperparameters(hyperparams: dict, n_samples: int) -> None:
    """Confirm one PaCMAP hyperparameter combination is usable for this dataset.

    Args:
        hyperparams: This trial's `n_components`/`n_neighbors`/`mn_ratio`/`fp_ratio`.
        n_samples: Number of embeddings available to reduce.

    Raises:
        ValueError: If `n_components` is less than 1, if `mn_ratio` or
            `fp_ratio` is not greater than 0, or if `n_neighbors` is not less
            than `n_samples` (PaCMAP's own neighborhood graph is degenerate
            otherwise) — naming both numbers, pre-empting pacmap's own
            less-contextualized failure for the same condition.
    """
    n_components = hyperparams["n_components"]
    n_neighbors = hyperparams["n_neighbors"]
    mn_ratio = hyperparams["mn_ratio"]
    fp_ratio = hyperparams["fp_ratio"]
    if n_components < 1:
        raise ValueError(f"n_components must be at least 1, got {n_components}")
    if mn_ratio <= 0:
        raise ValueError(f"mn_ratio must be greater than 0, got {mn_ratio}")
    if fp_ratio <= 0:
        raise ValueError(f"fp_ratio must be greater than 0, got {fp_ratio}")
    if n_neighbors >= n_samples:
        raise ValueError(
            f"n_neighbors={n_neighbors} must be less than the number of available embeddings "
            f"({n_samples}); reduce --n-neighbors or provide more data"
        )


def annotate_dimensionality_note(candidate: reduction_sweep.ReducerConfigCandidate, embedding_dim: int) -> None:
    """Append a non-fatal note to the winning candidate if `n_components` doesn't meaningfully reduce dimensionality.

    Per the spec, `n_components >= embedding_dim` is not an error — just
    something to surface. See `umap_hdbscan_clustering.annotate_dimensionality_note`
    for the identical UMAP-side version of this note.

    Args:
        candidate: The winning trial's candidate, mutated in place.
        embedding_dim: The original, non-reduced embedding dimensionality.
    """
    n_components = candidate.hyperparams["n_components"]
    if n_components >= embedding_dim:
        candidate.notes.append(
            f"n_components={n_components} is >= the original embedding_dim={embedding_dim}; this is "
            "not an error, but PaCMAP's dimensionality reduction is not meaningfully reducing "
            "dimensionality in this configuration"
        )


def build_reduced_embedding_rows(table: pa.Table, reduced_matrix: np.ndarray, reduced_at: datetime) -> list[dict]:
    """Zip each surviving row's traceability columns with its PaCMAP-reduced vector.

    Args:
        table: The table `reduced_matrix` was produced from (row order must match).
        reduced_matrix: PaCMAP's output, shape `(table.num_rows, n_components)`.
        reduced_at: Wall-clock time this reduction run completed.

    Returns:
        One output dict per row, matching `PACMAP_REDUCED_EMBEDDING_PARQUET_SCHEMA`'s columns.
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


def build_metrics_dict(
    label: str,
    input_path: Path,
    reduced_embeddings_output_path: Path,
    assignments_output_path: Path,
    n_samples: int,
    n_samples_excluded: int,
    embedding_dim: int,
    embedding_backend: str | None,
    embedding_model: str | None,
    hdbscan_clusterer: HDBSCANClusterer,
    candidate: reduction_sweep.ReducerConfigCandidate,
    seeds: list[int],
    repeats_per_seed: int,
    pacmap_distance: str,
    clustered_at: datetime,
) -> dict:
    """Assemble the full metrics JSON payload for one PaCMAP+HDBSCAN run.

    See `umap_hdbscan_clustering.build_metrics_dict` for the identical
    UMAP-side version of this function; the only difference is the
    `winning_pacmap_config` field name and its `mn_ratio`/`fp_ratio` naming.
    The caller (`run`) merges in `optuna_study_summary`/`objective_weights`/
    `dr_metrics` afterward.

    Args:
        label: Human-readable label for this run, derived from `input_path`'s filename.
        input_path: The embeddings Parquet path that was read.
        reduced_embeddings_output_path: Where the reduced embeddings were written.
        assignments_output_path: Where cluster assignments were written.
        n_samples: Number of successfully-embedded rows reduced/clustered.
        n_samples_excluded: Number of input rows dropped for having failed embeddings.
        embedding_dim: Original (non-reduced) embedding dimensionality.
        embedding_backend: Backend recorded on the clustered rows.
        embedding_model: Model recorded on the clustered rows.
        hdbscan_clusterer: The winning trial's `HDBSCANClusterer`.
        candidate: The winning trial's `ReducerConfigCandidate`.
        seeds: The configured reproducibility seed list.
        repeats_per_seed: Number of repeats run per seed.
        pacmap_distance: PaCMAP's fixed distance metric.
        clustered_at: Wall-clock time this run completed.

    Returns:
        A JSON-serializable dict.
    """
    representative = reduction_sweep.representative_repeat(candidate)
    base = clustering_io.build_base_metrics_dict(
        label=label,
        input_path=input_path,
        assignments_output_path=assignments_output_path,
        n_samples=n_samples,
        n_samples_excluded=n_samples_excluded,
        embedding_dim=embedding_dim,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
        cluster_sizes=clustering_io.compute_cluster_sizes(representative.labels),
        clustered_at=clustered_at,
    )
    return {
        **base,
        "reduced_embeddings_output_path": clustering_io.relative_to_repo_root(reduced_embeddings_output_path),
        "n_clusters_actual": representative.metrics_result["n_clusters_actual"],
        "noise_count": representative.metrics_result["noise_count"],
        "noise_fraction": representative.metrics_result["noise_fraction"],
        "hdbscan_params": {
            "min_cluster_size": hdbscan_clusterer.min_cluster_size,
            "min_samples": hdbscan_clusterer.min_samples,
            "metric": hdbscan_clusterer.metric,
            "cluster_selection_epsilon": hdbscan_clusterer.cluster_selection_epsilon,
            "cluster_selection_method": hdbscan_clusterer.cluster_selection_method,
            "alpha": hdbscan_clusterer.alpha,
        },
        "metrics": representative.metrics_result["metrics"],
        "metrics_valid": representative.metrics_result["metrics_valid"],
        "relative_validity_index": representative.relative_validity,
        "notes": representative.metrics_result["notes"] + candidate.notes,
        "winning_pacmap_config": {k: v for k, v in candidate.hyperparams.items() if k not in HDBSCAN_PARAM_NAMES},
        "winning_seed": candidate.winning_seed.seed,
        "winning_seed_noise_fraction_std": candidate.winning_seed.noise_fraction_std,
        "pacmap_fixed_distance": pacmap_distance,
        "reproducibility_seeds": seeds,
        "reproducibility_repeats_per_seed": repeats_per_seed,
    }


def run(
    input_path: Path,
    reduced_embeddings_output_path: Path,
    assignments_output_path: Path,
    metrics_output_path: Path,
    config_path: Path,
) -> tuple[Path, Path, Path] | None:
    """Run the full read-filter-search-write pipeline.

    See `umap_hdbscan_clustering.run` for the identical UMAP-side version of
    this function; the only differences are PaCMAP's hyperparameter names
    (`mn_ratio`/`fp_ratio` instead of `min_dist`) and output paths.

    Args:
        input_path: Embeddings Parquet to read.
        reduced_embeddings_output_path: Where to write reduced embeddings.
        assignments_output_path: Where to write cluster assignments.
        metrics_output_path: Where to write metrics.
        config_path: YAML config (combined PaCMAP + HDBSCAN search space,
            objective weights, trial budget, reproducibility seeds/repeats,
            study/report directories) — see `optuna_search.load_optuna_config`.

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
    slug = slug_from_filename(input_path.name, "_trials_embeddings.parquet")
    label = label_from_slug(slug)

    table = clustering_io.read_embeddings_parquet(input_path)
    total_rows = table.num_rows
    filtered = clustering_io.filter_successful_rows(table)
    n_samples = filtered.num_rows
    n_excluded = total_rows - n_samples

    embedding_dim = clustering_io.validate_embedding_dimensions(filtered)
    embedding_matrix = clustering_io.extract_embedding_matrix(filtered)

    config = optuna_search.load_optuna_config(config_path, require_reproducibility=True)
    bounds = optuna_search.EligibilityBounds(
        min_n_clusters=HDBSCANClusterer.MIN_N_CLUSTERS,
        max_n_clusters=HDBSCANClusterer.MAX_N_CLUSTERS,
        max_cluster_share=HDBSCANClusterer.MAX_CLUSTER_SHARE,
    )
    pacmap_distance = config.fixed["distance"]
    fit_reduced = make_fit_pacmap_reduced(pacmap_distance)

    reducer_space = {name: spec for name, spec in config.search_space.items() if name not in HDBSCAN_PARAM_NAMES}
    hdbscan_space = {name: spec for name, spec in config.search_space.items() if name in HDBSCAN_PARAM_NAMES}

    def build_hdbscan_clusterer(hyperparams: dict) -> HDBSCANClusterer:
        return HDBSCANClusterer(
            min_cluster_size=hyperparams["min_cluster_size"],
            min_samples=hyperparams["min_samples"],
            metric=pacmap_distance,
            cluster_selection_epsilon=hyperparams["cluster_selection_epsilon"],
            cluster_selection_method=hyperparams["cluster_selection_method"],
            alpha=hyperparams["alpha"],
        )

    def evaluate_trial(hyperparams: dict) -> dict:
        """Evaluate one combined PaCMAP+HDBSCAN hyperparameter combination into a trial-cache-shaped record.

        Pulled out of the objective closure so the post-`study.optimize` winner
        resolution can call it too: a resumed study's `study.best_trial` may be
        from a *previous* process invocation, never present in this process's
        own `trial_cache` (Optuna's storage persists only hyperparameters/score,
        not our fit artifacts) — re-evaluating is the only way to recover its
        reduced matrix/labels/probabilities, and is safe since both PaCMAP
        (given the same `random_state`) and HDBSCAN are fully deterministic.
        """
        candidate = reduction_sweep.evaluate_config(
            fit_reduced,
            hyperparams,
            config.reproducibility.seeds,
            config.reproducibility.repeats_per_seed,
            validate_pacmap_hyperparameters,
            build_hdbscan_clusterer,
            embedding_matrix,
            n_samples,
        )
        if candidate.status == "invalid":
            return {"status": "invalid", "candidate": candidate, "hyperparams": hyperparams}

        representative = reduction_sweep.representative_repeat(candidate)
        dr_metrics = optuna_search.compute_dr_metrics(
            embedding_matrix, representative.reduced_matrix, representative.labels, pacmap_distance
        )
        return {
            "status": candidate.status,
            "candidate": candidate,
            "dr_metrics": dr_metrics,
            "hyperparams": hyperparams,
        }

    study_name = f"{slug}_pacmap_hdbscan"
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

        representative = reduction_sweep.representative_repeat(record["candidate"])
        return optuna_search.score_reduction_candidate(
            representative.metrics_result,
            representative.relative_validity,
            representative.labels,
            hyperparams["min_cluster_size"],
            config.objective_weights,
            record["dr_metrics"],
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

    best_trial = optuna_search.best_trial_or_none(study)
    if best_trial is None:
        logger.warning("run: study has no trials for %r; writing nothing", label)
        return None
    winner = trial_cache.get(best_trial.number)
    if winner is None:
        logger.info(
            "run: best trial #%d for %r predates this process; re-evaluating its hyperparameters",
            best_trial.number, label,
        )
        winner = evaluate_trial(best_trial.params)
    if winner["status"] == "invalid":
        logger.warning("run: no trial ever fit for %r; writing nothing", label)
        return None

    candidate: reduction_sweep.ReducerConfigCandidate = winner["candidate"]
    annotate_dimensionality_note(candidate, embedding_dim)
    representative = reduction_sweep.representative_repeat(candidate)
    hdbscan_clusterer = build_hdbscan_clusterer(winner["hyperparams"])

    clustered_at = datetime.now(timezone.utc)
    reduced_rows = build_reduced_embedding_rows(filtered, representative.reduced_matrix, clustered_at)
    clustering_io.write_parquet(reduced_rows, PACMAP_REDUCED_EMBEDDING_PARQUET_SCHEMA, reduced_embeddings_output_path)

    assignment_rows = hdbscan_clusterer.build_assignment_rows(
        filtered, representative.labels, representative.probabilities, clustered_at
    )
    clustering_io.write_parquet(assignment_rows, HDBSCAN_CLUSTER_PARQUET_SCHEMA, assignments_output_path)

    embedding_backend = filtered.column("embedding_backend")[0].as_py()
    embedding_model = filtered.column("embedding_model")[0].as_py()
    metrics_dict = build_metrics_dict(
        label=label,
        input_path=input_path,
        reduced_embeddings_output_path=reduced_embeddings_output_path,
        assignments_output_path=assignments_output_path,
        n_samples=n_samples,
        n_samples_excluded=n_excluded,
        embedding_dim=embedding_dim,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
        hdbscan_clusterer=hdbscan_clusterer,
        candidate=candidate,
        seeds=config.reproducibility.seeds,
        repeats_per_seed=config.reproducibility.repeats_per_seed,
        pacmap_distance=pacmap_distance,
        clustered_at=clustered_at,
    )
    dr_metrics = winner["dr_metrics"]
    metrics_dict["dr_metrics"] = asdict(dr_metrics) if dr_metrics is not None else None
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
        "run: best trial #%d PaCMAP config %s (seed=%d) clustered %d row(s) into %d cluster(s) for %r -> %s, %s, %s "
        "(noise=%d/%d (%.1f%%), silhouette=%s, excluded=%d failed-embedding row(s), "
        "trials: %d ok, %d degenerate, %d invalid)",
        best_trial.number, candidate.hyperparams, candidate.winning_seed.seed, n_samples,
        representative.metrics_result["n_clusters_actual"], label, reduced_embeddings_output_path,
        assignments_output_path, metrics_output_path, representative.metrics_result["noise_count"], n_samples,
        representative.metrics_result["noise_fraction"] * 100, representative.metrics_result["metrics"]["silhouette_score"],
        n_excluded, status_counts["ok"], status_counts["degenerate"], status_counts["invalid"],
    )
    return reduced_embeddings_output_path, assignments_output_path, metrics_output_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for this module.

    Returns:
        An ``argparse.ArgumentParser`` accepting required input/output path
        flags plus a `--config-path` override for the Optuna search config.
    """
    parser = argparse.ArgumentParser(
        description="Reduce criterion embeddings via PaCMAP, then cluster via HDBSCAN, tuned by an Optuna search."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Input embeddings Parquet path (as written by criteria_embedding.py)",
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
        default=REPO_ROOT / "config" / "pacmap_hdbscan_optuna.yaml",
        help="Optuna search config YAML (combined PaCMAP + HDBSCAN search space, objective "
        "weights, trial budget, reproducibility seeds/repeats); default: "
        "config/pacmap_hdbscan_optuna.yaml",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="If given, also append logs to this file (in addition to the console)",
    )
    return parser


def main() -> None:
    """CLI entry point: run the Optuna search and write reduced embeddings/cluster assignments/metrics."""
    args = build_arg_parser().parse_args()
    configure_logging(args.log_file)

    run(
        args.input_path,
        args.reduced_embeddings_output_path,
        args.assignments_output_path,
        args.metrics_output_path,
        args.config_path,
    )


if __name__ == "__main__":
    main()
