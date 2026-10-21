"""Reduce criterion embeddings with UMAP, then cluster via HDBSCAN.

Reads the same embeddings Parquet file format `hdbscan_clustering.py` reads,
drops rows whose embedding failed, then sweeps a grid of UMAP hyperparameter combinations
(``n_components``/``n_neighbors``/``min_dist``). For each combination, every
seed in a user-supplied seed list is fit ``repeats_per_seed`` times (default
10) so the most *reproducible* seed — the one whose repeats agree most with
each other, not the one that scores best — can be selected; HDBSCAN's own
hyperparameters are held fixed across the whole sweep rather than swept
(``hdbscan_clustering.py`` already covers sweeping those alone, including
against this step's own reduced-embeddings output). The config/seed/repeat sweep
mechanics — reducer-agnostic — live in ``reduction_sweep.py``; this module
only supplies UMAP's own fit/validate functions and this step's I/O.

The run writes three output artifacts, at the paths given via
`--reduced-embeddings-output-path`/`--assignments-output-path`/
`--metrics-output-path`:

    Reduced embeddings: one row per successfully-embedded criterion,
        carrying its UMAP-reduced vector, independently joinable back to the
        source criteria via ``nct_id`` without needing the cluster assignments.

    Cluster assignments: one row per successfully-embedded criterion,
        carrying its assigned ``cluster_id`` (``-1`` for noise) and
        ``cluster_probability`` — same schema `hdbscan_clustering.py` itself
        writes.

    Cluster metrics: clustering performance metrics for the winning
        configuration, plus the winning UMAP config/seed and the full
        config/seed/repeat sweep audit trail, so the choice is traceable
        rather than a black box.

Re-running this step against the same output paths overwrites all three
existing output files wholesale; there is no append/skip-if-exists/merge
behavior.
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pyarrow as pa
import umap

# Allow running this file directly (e.g. `uv run src/clustering/umap_hdbscan_clustering.py`)
# as well as via `-m` (e.g. `uv run python -m src.clustering.umap_hdbscan_clustering`).
# Direct execution leaves __package__ unset and never puts the repo root on
# sys.path, so the src.* imports below would otherwise fail with
# "ModuleNotFoundError: No module named 'src'"; -m already sets both up
# correctly, so this is a no-op in that case.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.paths import label_from_slug, slug_from_filename
from src.clustering import clustering_io
from src.clustering import reduction_sweep
from src.clustering.hdbscan_clustering import (
    HDBSCAN_CLUSTER_PARQUET_SCHEMA,
    HDBSCANClusterer,
    parse_float_list,
    parse_int_list,
)

logger = logging.getLogger(__name__)

DEFAULT_N_COMPONENTS = [2]
DEFAULT_N_NEIGHBORS = [15]
DEFAULT_MIN_DIST = [0.1]
DEFAULT_METRIC = "euclidean"
DEFAULT_SEEDS = [42]
DEFAULT_REPEATS_PER_SEED = 10
DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE_VALUES = [HDBSCANClusterer.DEFAULT_MIN_CLUSTER_SIZE]

UMAP_REDUCED_EMBEDDING_PARQUET_SCHEMA = pa.schema(
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


def make_fit_umap_reduced(metric: str) -> Callable[..., np.ndarray]:
    """Bind UMAP's fixed (non-swept) distance metric into a `reduction_sweep`-compatible callable.

    Args:
        metric: UMAP's own distance metric, held fixed across the whole sweep.

    Returns:
        A callable `(embedding_matrix, seed, n_components, n_neighbors,
        min_dist) -> reduced_matrix`, matching the shape
        `reduction_sweep.evaluate_repeat` expects for `fit_reduced`.
    """

    def fit_reduced(
        embedding_matrix: np.ndarray, seed: int, n_components: int, n_neighbors: int, min_dist: float
    ) -> np.ndarray:
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=seed,
        )
        return reducer.fit_transform(embedding_matrix)

    return fit_reduced


def validate_umap_hyperparameters(hyperparams: dict, n_samples: int) -> None:
    """Confirm one UMAP hyperparameter combination is usable for this dataset.

    Args:
        hyperparams: This config's `n_components`/`n_neighbors`/`min_dist`.
        n_samples: Number of embeddings available to reduce.

    Raises:
        ValueError: If `n_components` is less than 1, if `min_dist` is not in
            `[0.0, 1.0]`, or if `n_neighbors` is not less than `n_samples`
            (UMAP's own neighborhood graph is degenerate otherwise) — naming
            both numbers, pre-empting umap-learn's own less-contextualized
            failure for the same condition.
    """
    n_components = hyperparams["n_components"]
    n_neighbors = hyperparams["n_neighbors"]
    min_dist = hyperparams["min_dist"]
    if n_components < 1:
        raise ValueError(f"n_components must be at least 1, got {n_components}")
    if not (0.0 <= min_dist <= 1.0):
        raise ValueError(f"min_dist must be between 0.0 and 1.0, got {min_dist}")
    if n_neighbors >= n_samples:
        raise ValueError(
            f"n_neighbors={n_neighbors} must be less than the number of available embeddings "
            f"({n_samples}); reduce --n-neighbors or provide more data"
        )


def annotate_dimensionality_notes(candidates: list[reduction_sweep.ReducerConfigCandidate], embedding_dim: int) -> None:
    """Append a non-fatal note to any candidate whose `n_components` doesn't meaningfully reduce dimensionality.

    Per the spec, `n_components >= embedding_dim` is not an error — just
    something to surface. `ReducerConfigCandidate.notes` is a plain list, so
    appending to it in place is safe even though the dataclass is frozen
    (frozen only blocks reassigning the `notes` attribute itself).

    Args:
        candidates: Every candidate from this sweep, mutated in place.
        embedding_dim: The original, non-reduced embedding dimensionality.
    """
    for candidate in candidates:
        n_components = candidate.hyperparams["n_components"]
        if n_components >= embedding_dim:
            candidate.notes.append(
                f"n_components={n_components} is >= the original embedding_dim={embedding_dim}; this is "
                "not an error, but UMAP's dimensionality reduction is not meaningfully reducing "
                "dimensionality in this configuration"
            )


def build_reduced_embedding_rows(table: pa.Table, reduced_matrix: np.ndarray, reduced_at: datetime) -> list[dict]:
    """Zip each surviving row's traceability columns with its UMAP-reduced vector.

    Args:
        table: The table `reduced_matrix` was produced from (row order must match).
        reduced_matrix: UMAP's output, shape `(table.num_rows, n_components)`.
        reduced_at: Wall-clock time this reduction run completed.

    Returns:
        One output dict per row, matching `UMAP_REDUCED_EMBEDDING_PARQUET_SCHEMA`'s columns.
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
    best: reduction_sweep.ReducerConfigCandidate,
    all_candidates: list[reduction_sweep.ReducerConfigCandidate],
    seeds: list[int],
    repeats_per_seed: int,
    umap_metric: str,
    clustered_at: datetime,
) -> dict:
    """Assemble the full metrics JSON payload for one UMAP+HDBSCAN run.

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
        hdbscan_clusterer: The winning combination's `HDBSCANClusterer`
            (built from `best.hyperparams["min_cluster_size"]` plus this
            run's other, fixed HDBSCAN hyperparameters).
        best: The winning `ReducerConfigCandidate`.
        all_candidates: Every candidate tried, for the audit trail.
        seeds: The full seed list that was swept.
        repeats_per_seed: Number of repeats run per seed.
        umap_metric: UMAP's fixed distance metric.
        clustered_at: Wall-clock time this run completed.

    Returns:
        A JSON-serializable dict.
    """
    representative = reduction_sweep.representative_repeat(best)
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
        "notes": representative.metrics_result["notes"] + best.notes,
        "winning_umap_config": {k: v for k, v in best.hyperparams.items() if k != "min_cluster_size"},
        "winning_seed": best.winning_seed.seed,
        "winning_seed_noise_fraction_std": best.winning_seed.noise_fraction_std,
        "umap_config_search": {
            "n_components_values": sorted({c.hyperparams["n_components"] for c in all_candidates}),
            "n_neighbors_values": sorted({c.hyperparams["n_neighbors"] for c in all_candidates}),
            "min_dist_values": sorted({c.hyperparams["min_dist"] for c in all_candidates}),
            "min_cluster_size_values": sorted({c.hyperparams["min_cluster_size"] for c in all_candidates}),
            "metric": umap_metric,
            "seeds": seeds,
            "repeats_per_seed": repeats_per_seed,
            "candidates": reduction_sweep.serialize_candidates(all_candidates),
        },
    }


def run(
    input_path: Path,
    reduced_embeddings_output_path: Path,
    assignments_output_path: Path,
    metrics_output_path: Path,
    n_components_values: list[int] = DEFAULT_N_COMPONENTS,
    n_neighbors_values: list[int] = DEFAULT_N_NEIGHBORS,
    min_dist_values: list[float] = DEFAULT_MIN_DIST,
    umap_metric: str = DEFAULT_METRIC,
    seeds: list[int] = DEFAULT_SEEDS,
    repeats_per_seed: int = DEFAULT_REPEATS_PER_SEED,
    hdbscan_min_cluster_size_values: list[int] = DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE_VALUES,
    hdbscan_min_samples: int = HDBSCANClusterer.DEFAULT_MIN_CLUSTER_SIZE,
    hdbscan_metric: str = HDBSCANClusterer.DEFAULT_METRIC,
    hdbscan_cluster_selection_epsilon: float = HDBSCANClusterer.DEFAULT_CLUSTER_SELECTION_EPSILON,
    hdbscan_cluster_selection_method: str = HDBSCANClusterer.DEFAULT_CLUSTER_SELECTION_METHOD,
    hdbscan_alpha: float = HDBSCANClusterer.DEFAULT_ALPHA,
) -> tuple[Path, Path, Path] | None:
    """Run the full read-filter-sweep-write pipeline.

    Args:
        input_path: Embeddings Parquet to read.
        reduced_embeddings_output_path: Where to write reduced embeddings.
        assignments_output_path: Where to write cluster assignments.
        metrics_output_path: Where to write metrics.
        n_components_values: UMAP `n_components` values to sweep.
        n_neighbors_values: UMAP `n_neighbors` values to sweep.
        min_dist_values: UMAP `min_dist` values to sweep.
        umap_metric: UMAP's distance metric, held fixed across the sweep.
        seeds: UMAP `random_state` values to sweep for reproducibility.
        repeats_per_seed: Number of repeated fits per (config, seed) pair.
        hdbscan_min_cluster_size_values: HDBSCAN `min_cluster_size` values to
            sweep, folded into the same config grid as the UMAP
            hyperparameters above (so a config now means a UMAP
            hyperparameter combination *and* a `min_cluster_size` value).
        hdbscan_min_samples: Fixed (non-swept) HDBSCAN hyperparameter; always
            an explicit value (never `None`), so the effective `min_samples`
            HDBSCAN fits with is never left to sklearn's own implicit fallback.
        hdbscan_metric: Fixed (non-swept) HDBSCAN hyperparameter.
        hdbscan_cluster_selection_epsilon: Fixed (non-swept) HDBSCAN hyperparameter.
        hdbscan_cluster_selection_method: Fixed (non-swept) HDBSCAN hyperparameter.
        hdbscan_alpha: Fixed (non-swept) HDBSCAN hyperparameter.

    Returns:
        `(reduced_embeddings_output_path, assignments_output_path,
        metrics_output_path)`, or `None` if no UMAP configuration produced a
        usable clustering result (in which case nothing is written).

    Raises:
        FileNotFoundError: If the input Parquet does not exist.
        ValueError: If there are no successfully-embedded rows to cluster,
            if embedding dimensionality is inconsistent, or if the sweep
            grid/seed list/repeat count exceed `reduction_sweep`'s caps.
    """
    label = label_from_slug(slug_from_filename(input_path.name, "_trials_embeddings.parquet"))

    table = clustering_io.read_embeddings_parquet(input_path)
    total_rows = table.num_rows
    filtered = clustering_io.filter_successful_rows(table)
    n_samples = filtered.num_rows
    n_excluded = total_rows - n_samples

    embedding_dim = clustering_io.validate_embedding_dimensions(filtered)
    embedding_matrix = clustering_io.extract_embedding_matrix(filtered)

    def build_hdbscan_clusterer(hyperparams: dict) -> HDBSCANClusterer:
        return HDBSCANClusterer(
            min_cluster_size=hyperparams["min_cluster_size"],
            min_samples=hdbscan_min_samples,
            metric=hdbscan_metric,
            cluster_selection_epsilon=hdbscan_cluster_selection_epsilon,
            cluster_selection_method=hdbscan_cluster_selection_method,
            alpha=hdbscan_alpha,
        )

    fit_reduced = make_fit_umap_reduced(umap_metric)
    best, all_candidates = reduction_sweep.run_reduction_sweep(
        fit_reduced=fit_reduced,
        validate_fn=validate_umap_hyperparameters,
        hyperparameter_values={
            "n_components": n_components_values,
            "n_neighbors": n_neighbors_values,
            "min_dist": min_dist_values,
            "min_cluster_size": hdbscan_min_cluster_size_values,
        },
        seeds=seeds,
        repeats_per_seed=repeats_per_seed,
        build_hdbscan_clusterer=build_hdbscan_clusterer,
        embedding_matrix=embedding_matrix,
        n_samples=n_samples,
    )
    annotate_dimensionality_notes(all_candidates, embedding_dim)

    if best is None:
        logger.warning(
            "run: no UMAP configuration produced a usable relative validity index for %r; writing nothing",
            label,
        )
        return None

    representative = reduction_sweep.representative_repeat(best)
    clustered_at = datetime.now(timezone.utc)
    hdbscan_clusterer = build_hdbscan_clusterer(best.hyperparams)

    reduced_rows = build_reduced_embedding_rows(filtered, representative.reduced_matrix, clustered_at)
    clustering_io.write_parquet(reduced_rows, UMAP_REDUCED_EMBEDDING_PARQUET_SCHEMA, reduced_embeddings_output_path)

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
        best=best,
        all_candidates=all_candidates,
        seeds=seeds,
        repeats_per_seed=repeats_per_seed,
        umap_metric=umap_metric,
        clustered_at=clustered_at,
    )
    clustering_io.write_json(metrics_dict, metrics_output_path)

    logger.info(
        "run: winning UMAP config %s (seed=%d) clustered %d row(s) into %d cluster(s) for %r -> %s, %s, %s "
        "(noise=%d/%d (%.1f%%), silhouette=%s, excluded=%d failed-embedding row(s))",
        best.hyperparams, best.winning_seed.seed, n_samples, representative.metrics_result["n_clusters_actual"],
        label, reduced_embeddings_output_path, assignments_output_path, metrics_output_path,
        representative.metrics_result["noise_count"], n_samples, representative.metrics_result["noise_fraction"] * 100,
        representative.metrics_result["metrics"]["silhouette_score"], n_excluded,
    )
    return reduced_embeddings_output_path, assignments_output_path, metrics_output_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for this module.

    Returns:
        An ``argparse.ArgumentParser`` accepting UMAP sweep options, fixed
        HDBSCAN hyperparameter flags, and required input/output path flags.
        Flag names are agnostic to which algorithm they configure (e.g.
        ``--n-components``, ``--cluster-selection-method``); the one
        exception is ``--distance-metric``, used for UMAP's own distance
        metric since a bare ``--metric`` would be ambiguous with HDBSCAN's
        (mirrored by `pacmap_hdbscan_clustering.build_arg_parser`, which
        uses the same ``--distance-metric`` name for PaCMAP's).
    """
    parser = argparse.ArgumentParser(
        description="Reduce criterion embeddings via UMAP, then cluster via HDBSCAN."
    )
    parser.add_argument(
        "--n-components",
        type=parse_int_list,
        default=DEFAULT_N_COMPONENTS,
        help="Comma-separated list of UMAP n_components values to sweep (default: 2)",
    )
    parser.add_argument(
        "--n-neighbors",
        type=parse_int_list,
        default=DEFAULT_N_NEIGHBORS,
        help="Comma-separated list of UMAP n_neighbors values to sweep (default: 15)",
    )
    parser.add_argument(
        "--min-dist",
        type=parse_float_list,
        default=DEFAULT_MIN_DIST,
        help="Comma-separated list of UMAP min_dist values to sweep (default: 0.1)",
    )
    parser.add_argument(
        "--distance-metric",
        type=str,
        default=DEFAULT_METRIC,
        help="UMAP distance metric, held fixed across the sweep (default: euclidean)",
    )
    parser.add_argument(
        "--seeds",
        type=parse_int_list,
        default=DEFAULT_SEEDS,
        help="Comma-separated list of seeds to sweep for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--repeats-per-seed",
        type=int,
        default=DEFAULT_REPEATS_PER_SEED,
        help="Number of repeated fits per (config, seed) pair, used to measure reproducibility (default: 10)",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=parse_int_list,
        default=DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE_VALUES,
        help="Comma-separated list of HDBSCAN min_cluster_size values to sweep, folded into the "
        "same config grid as the UMAP hyperparameters above (default: 200)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=HDBSCANClusterer.DEFAULT_MIN_CLUSTER_SIZE,
        help="Fixed HDBSCAN min_samples, not swept (default: 200)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=HDBSCANClusterer.DEFAULT_METRIC,
        help="Fixed HDBSCAN distance metric, not swept (default: euclidean)",
    )
    parser.add_argument(
        "--cluster-selection-epsilon",
        type=float,
        default=HDBSCANClusterer.DEFAULT_CLUSTER_SELECTION_EPSILON,
        help="Fixed HDBSCAN cluster_selection_epsilon, not swept (default: 0.0)",
    )
    parser.add_argument(
        "--cluster-selection-method",
        type=str,
        choices=["eom", "leaf"],
        default=HDBSCANClusterer.DEFAULT_CLUSTER_SELECTION_METHOD,
        help="Fixed HDBSCAN cluster_selection_method, not swept (default: eom)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=HDBSCANClusterer.DEFAULT_ALPHA,
        help="Fixed HDBSCAN alpha, not swept (default: 1.0)",
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
    return parser


def main() -> None:
    """CLI entry point: sweep, cluster, and write reduced embeddings/cluster assignments/metrics."""
    logging.basicConfig(level=logging.INFO)

    args = build_arg_parser().parse_args()

    run(
        args.input_path,
        args.reduced_embeddings_output_path,
        args.assignments_output_path,
        args.metrics_output_path,
        n_components_values=args.n_components,
        n_neighbors_values=args.n_neighbors,
        min_dist_values=args.min_dist,
        umap_metric=args.distance_metric,
        seeds=args.seeds,
        repeats_per_seed=args.repeats_per_seed,
        hdbscan_min_cluster_size_values=args.min_cluster_size,
        hdbscan_min_samples=args.min_samples,
        hdbscan_metric=args.metric,
        hdbscan_cluster_selection_epsilon=args.cluster_selection_epsilon,
        hdbscan_cluster_selection_method=args.cluster_selection_method,
        hdbscan_alpha=args.alpha,
    )


if __name__ == "__main__":
    main()
