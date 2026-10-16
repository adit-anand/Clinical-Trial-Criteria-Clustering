"""Reduce per-condition criterion embeddings with PaCMAP, then cluster via HDBSCAN.

Mirrors `umap_hdbscan_clustering.py` exactly, swapping UMAP for PaCMAP as the
reducer. Reads the same embeddings Parquet file `hdbscan_clustering.py`
reads (``data/embeddings/<condition>_trials_embeddings.parquet``), drops
rows whose embedding failed, then sweeps a grid of PaCMAP hyperparameter
combinations (``n_components``/``n_neighbors``/``MN_ratio``/``FP_ratio``).
PaCMAP has no ``min_dist`` equivalent to UMAP's — its neighborhood/distance
behavior is instead shaped by ``MN_ratio``/``FP_ratio``, which weight
mid-near vs. further point pairs during optimization. As with the UMAP
module, HDBSCAN's own hyperparameters are held fixed across the sweep, and
every seed in a user-supplied seed list is fit ``repeats_per_seed`` times so
the most *reproducible* seed can be selected; the reducer-agnostic
config/seed/repeat sweep mechanics live in ``reduction_sweep.py``.

The run writes three output artifacts:

    data/clustering/pacmap_hdbscan/<condition>_trials_pacmap_reduced_embeddings.parquet
    data/clustering/pacmap_hdbscan/<condition>_trials_pacmap_hdbscan_clusters.parquet
    data/clustering/pacmap_hdbscan/<condition>_trials_pacmap_hdbscan_cluster_metrics.json

Re-running this step on the same condition overwrites all three existing
output files wholesale; there is no append/skip-if-exists/merge behavior.
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
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

from src.paths import REPO_ROOT
from src.preprocessing.criteria_retrieval import EligibilityCriteriaFetcher
from src.clustering import clustering_io
from src.clustering import reduction_sweep
from src.clustering.hdbscan_clustering import (
    HDBSCAN_CLUSTER_PARQUET_SCHEMA,
    HDBSCANClusterer,
    parse_float_list,
    parse_int_list,
)

OUTPUT_DIR = REPO_ROOT / "data" / "clustering" / "pacmap_hdbscan"

logger = logging.getLogger(__name__)

DEFAULT_N_COMPONENTS = [2]
DEFAULT_N_NEIGHBORS = [10]
DEFAULT_MN_RATIO = [0.5]
DEFAULT_FP_RATIO = [2.0]
DEFAULT_DISTANCE = "euclidean"
DEFAULT_SEEDS = [42]
DEFAULT_REPEATS_PER_SEED = 10
DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE_VALUES = [HDBSCANClusterer.DEFAULT_MIN_CLUSTER_SIZE]

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


def make_fit_pacmap_reduced(distance: str) -> Callable[..., np.ndarray]:
    """Bind PaCMAP's fixed (non-swept) distance metric into a `reduction_sweep`-compatible callable.

    Args:
        distance: PaCMAP's own distance metric, held fixed across the whole sweep.

    Returns:
        A callable `(embedding_matrix, seed, n_components, n_neighbors,
        mn_ratio, fp_ratio) -> reduced_matrix`, matching the shape
        `reduction_sweep.evaluate_repeat` expects for `fit_reduced`.
    """

    def fit_reduced(
        embedding_matrix: np.ndarray, seed: int, n_components: int, n_neighbors: int, mn_ratio: float, fp_ratio: float
    ) -> np.ndarray:
        reducer = pacmap.PaCMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            MN_ratio=mn_ratio,
            FP_ratio=fp_ratio,
            distance=distance,
            random_state=seed,
        )
        return reducer.fit_transform(embedding_matrix)

    return fit_reduced


def validate_pacmap_hyperparameters(hyperparams: dict, n_samples: int) -> None:
    """Confirm one PaCMAP hyperparameter combination is usable for this dataset.

    Args:
        hyperparams: This config's `n_components`/`n_neighbors`/`mn_ratio`/`fp_ratio`.
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
            f"({n_samples}); reduce --pacmap-n-neighbors or provide more data"
        )


def annotate_dimensionality_notes(candidates: list[reduction_sweep.ReducerConfigCandidate], embedding_dim: int) -> None:
    """Append a non-fatal note to any candidate whose `n_components` doesn't meaningfully reduce dimensionality.

    Per the spec, `n_components >= embedding_dim` is not an error — just
    something to surface. See `umap_hdbscan_clustering.annotate_dimensionality_notes`
    for the identical UMAP-side version of this note.

    Args:
        candidates: Every candidate from this sweep, mutated in place.
        embedding_dim: The original, non-reduced embedding dimensionality.
    """
    for candidate in candidates:
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
    condition: str,
    slug: str,
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
    pacmap_distance: str,
    clustered_at: datetime,
) -> dict:
    """Assemble the full metrics JSON payload for one PaCMAP+HDBSCAN run.

    See `umap_hdbscan_clustering.build_metrics_dict` for the identical
    UMAP-side version of this function; the only difference is the
    `pacmap_config_search` field name and its `metric`/`distance` naming.

    Args:
        condition: Condition string passed to `run`.
        slug: `condition`'s filesystem-safe slug.
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
        pacmap_distance: PaCMAP's fixed distance metric.
        clustered_at: Wall-clock time this run completed.

    Returns:
        A JSON-serializable dict.
    """
    representative = reduction_sweep.representative_repeat(best)
    base = clustering_io.build_base_metrics_dict(
        condition=condition,
        slug=slug,
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
        "winning_pacmap_config": {k: v for k, v in best.hyperparams.items() if k != "min_cluster_size"},
        "winning_seed": best.winning_seed.seed,
        "winning_seed_noise_fraction_std": best.winning_seed.noise_fraction_std,
        "pacmap_config_search": {
            "n_components_values": sorted({c.hyperparams["n_components"] for c in all_candidates}),
            "n_neighbors_values": sorted({c.hyperparams["n_neighbors"] for c in all_candidates}),
            "mn_ratio_values": sorted({c.hyperparams["mn_ratio"] for c in all_candidates}),
            "fp_ratio_values": sorted({c.hyperparams["fp_ratio"] for c in all_candidates}),
            "min_cluster_size_values": sorted({c.hyperparams["min_cluster_size"] for c in all_candidates}),
            "distance": pacmap_distance,
            "seeds": seeds,
            "repeats_per_seed": repeats_per_seed,
            "candidates": reduction_sweep.serialize_candidates(all_candidates),
        },
    }


def run(
    condition: str,
    n_components_values: list[int] = DEFAULT_N_COMPONENTS,
    n_neighbors_values: list[int] = DEFAULT_N_NEIGHBORS,
    mn_ratio_values: list[float] = DEFAULT_MN_RATIO,
    fp_ratio_values: list[float] = DEFAULT_FP_RATIO,
    pacmap_distance: str = DEFAULT_DISTANCE,
    seeds: list[int] = DEFAULT_SEEDS,
    repeats_per_seed: int = DEFAULT_REPEATS_PER_SEED,
    hdbscan_min_cluster_size_values: list[int] = DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE_VALUES,
    hdbscan_min_samples: int = HDBSCANClusterer.DEFAULT_MIN_CLUSTER_SIZE,
    hdbscan_metric: str = HDBSCANClusterer.DEFAULT_METRIC,
    hdbscan_cluster_selection_epsilon: float = HDBSCANClusterer.DEFAULT_CLUSTER_SELECTION_EPSILON,
    hdbscan_cluster_selection_method: str = HDBSCANClusterer.DEFAULT_CLUSTER_SELECTION_METHOD,
    hdbscan_alpha: float = HDBSCANClusterer.DEFAULT_ALPHA,
    input_path: Path | None = None,
    reduced_embeddings_output_path: Path | None = None,
    assignments_output_path: Path | None = None,
    metrics_output_path: Path | None = None,
) -> tuple[Path, Path, Path] | None:
    """Run the full read-filter-sweep-write pipeline for one condition.

    See `umap_hdbscan_clustering.run` for the identical UMAP-side version of
    this function; the only differences are PaCMAP's hyperparameter names
    (`mn_ratio`/`fp_ratio` instead of `min_dist`) and output paths.

    Args:
        condition: Condition string, e.g. "breast cancer".
        n_components_values: PaCMAP `n_components` values to sweep.
        n_neighbors_values: PaCMAP `n_neighbors` values to sweep.
        mn_ratio_values: PaCMAP `MN_ratio` values to sweep.
        fp_ratio_values: PaCMAP `FP_ratio` values to sweep.
        pacmap_distance: PaCMAP's distance metric, held fixed across the sweep.
        seeds: PaCMAP `random_state` values to sweep for reproducibility.
        repeats_per_seed: Number of repeated fits per (config, seed) pair.
        hdbscan_min_cluster_size_values: HDBSCAN `min_cluster_size` values to
            sweep, folded into the same config grid as the PaCMAP
            hyperparameters above (so a config now means a PaCMAP
            hyperparameter combination *and* a `min_cluster_size` value).
        hdbscan_min_samples: Fixed (non-swept) HDBSCAN hyperparameter; always
            an explicit value (never `None`), so the effective `min_samples`
            HDBSCAN fits with is never left to sklearn's own implicit fallback.
        hdbscan_metric: Fixed (non-swept) HDBSCAN hyperparameter.
        hdbscan_cluster_selection_epsilon: Fixed (non-swept) HDBSCAN hyperparameter.
        hdbscan_cluster_selection_method: Fixed (non-swept) HDBSCAN hyperparameter.
        hdbscan_alpha: Fixed (non-swept) HDBSCAN hyperparameter.
        input_path: Embeddings Parquet to read. If omitted, defaults as
            `hdbscan_clustering.py` itself does.
        reduced_embeddings_output_path: Where to write reduced embeddings.
            If omitted, defaults to
            ``OUTPUT_DIR / f"{slug}_trials_pacmap_reduced_embeddings.parquet"``.
        assignments_output_path: Where to write cluster assignments. If
            omitted, defaults to
            ``OUTPUT_DIR / f"{slug}_trials_pacmap_hdbscan_clusters.parquet"``.
        metrics_output_path: Where to write metrics. If omitted, defaults to
            ``OUTPUT_DIR / f"{slug}_trials_pacmap_hdbscan_cluster_metrics.json"``.

    Returns:
        `(reduced_embeddings_output_path, assignments_output_path,
        metrics_output_path)`, or `None` if no PaCMAP configuration produced
        a usable clustering result (in which case nothing is written).

    Raises:
        FileNotFoundError: If the resolved input Parquet does not exist.
        ValueError: If there are no successfully-embedded rows to cluster,
            if embedding dimensionality is inconsistent, or if the sweep
            grid/seed list/repeat count exceed `reduction_sweep`'s caps.
    """
    slug = EligibilityCriteriaFetcher.condition_to_slug(condition)
    default_input_path, default_assignments_path, default_metrics_path = clustering_io.resolve_default_paths(
        condition, OUTPUT_DIR, "_trials_pacmap_hdbscan_clusters.parquet", "_trials_pacmap_hdbscan_cluster_metrics.json"
    )
    if input_path is None:
        input_path = default_input_path
    if assignments_output_path is None:
        assignments_output_path = default_assignments_path
    if metrics_output_path is None:
        metrics_output_path = default_metrics_path
    if reduced_embeddings_output_path is None:
        reduced_embeddings_output_path = OUTPUT_DIR / f"{slug}_trials_pacmap_reduced_embeddings.parquet"

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

    fit_reduced = make_fit_pacmap_reduced(pacmap_distance)
    best, all_candidates = reduction_sweep.run_reduction_sweep(
        fit_reduced=fit_reduced,
        validate_fn=validate_pacmap_hyperparameters,
        hyperparameter_values={
            "n_components": n_components_values,
            "n_neighbors": n_neighbors_values,
            "mn_ratio": mn_ratio_values,
            "fp_ratio": fp_ratio_values,
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
            "run: no PaCMAP configuration produced a usable relative validity index for %r; writing nothing",
            condition,
        )
        return None

    representative = reduction_sweep.representative_repeat(best)
    clustered_at = datetime.now(timezone.utc)
    hdbscan_clusterer = build_hdbscan_clusterer(best.hyperparams)

    reduced_rows = build_reduced_embedding_rows(filtered, representative.reduced_matrix, clustered_at)
    clustering_io.write_parquet(reduced_rows, PACMAP_REDUCED_EMBEDDING_PARQUET_SCHEMA, reduced_embeddings_output_path)

    assignment_rows = hdbscan_clusterer.build_assignment_rows(
        filtered, representative.labels, representative.probabilities, clustered_at
    )
    clustering_io.write_parquet(assignment_rows, HDBSCAN_CLUSTER_PARQUET_SCHEMA, assignments_output_path)

    embedding_backend = filtered.column("embedding_backend")[0].as_py()
    embedding_model = filtered.column("embedding_model")[0].as_py()
    metrics_dict = build_metrics_dict(
        condition=condition,
        slug=slug,
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
        pacmap_distance=pacmap_distance,
        clustered_at=clustered_at,
    )
    clustering_io.write_json(metrics_dict, metrics_output_path)

    logger.info(
        "run: winning PaCMAP config %s (seed=%d) clustered %d row(s) into %d cluster(s) for %r -> %s, %s, %s "
        "(noise=%d/%d (%.1f%%), silhouette=%s, excluded=%d failed-embedding row(s))",
        best.hyperparams, best.winning_seed.seed, n_samples, representative.metrics_result["n_clusters_actual"],
        condition, reduced_embeddings_output_path, assignments_output_path, metrics_output_path,
        representative.metrics_result["noise_count"], n_samples, representative.metrics_result["noise_fraction"] * 100,
        representative.metrics_result["metrics"]["silhouette_score"], n_excluded,
    )
    return reduced_embeddings_output_path, assignments_output_path, metrics_output_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for this module.

    Returns:
        An ``argparse.ArgumentParser`` accepting a positional ``condition``
        argument plus PaCMAP sweep options, fixed HDBSCAN hyperparameter
        flags, and path-override options. PaCMAP-side flags are prefixed
        ``--pacmap-`` and HDBSCAN-side flags ``--hdbscan-``, mirroring
        `umap_hdbscan_clustering.build_arg_parser`'s ``--umap-``/``--hdbscan-``
        scheme.
    """
    parser = argparse.ArgumentParser(
        description="Reduce a condition's criterion embeddings via PaCMAP, then cluster via HDBSCAN."
    )
    parser.add_argument(
        "condition",
        help='Condition to cluster, e.g. "breast cancer" (same string passed to criteria_embedding.py)',
    )
    parser.add_argument(
        "--pacmap-n-components",
        type=parse_int_list,
        default=DEFAULT_N_COMPONENTS,
        help="Comma-separated list of PaCMAP n_components values to sweep (default: 2)",
    )
    parser.add_argument(
        "--pacmap-n-neighbors",
        type=parse_int_list,
        default=DEFAULT_N_NEIGHBORS,
        help="Comma-separated list of PaCMAP n_neighbors values to sweep (default: 10); large "
        "values (e.g. 30+) risk over-smoothing the projection into one dominant cluster, which "
        "the MIN_N_CLUSTERS/MAX_CLUSTER_SHARE/MAX_NOISE_FRACTION guards will reject as degenerate "
        "-- pair with a --hdbscan-min-cluster-size sweep instead of relying on very large values here",
    )
    parser.add_argument(
        "--pacmap-mn-ratio",
        type=parse_float_list,
        default=DEFAULT_MN_RATIO,
        help="Comma-separated list of PaCMAP MN_ratio values to sweep (default: 0.5)",
    )
    parser.add_argument(
        "--pacmap-fp-ratio",
        type=parse_float_list,
        default=DEFAULT_FP_RATIO,
        help="Comma-separated list of PaCMAP FP_ratio values to sweep (default: 2.0)",
    )
    parser.add_argument(
        "--pacmap-distance",
        type=str,
        default=DEFAULT_DISTANCE,
        help="PaCMAP distance metric, held fixed across the sweep (default: euclidean)",
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
        "--hdbscan-min-cluster-size",
        type=parse_int_list,
        default=DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE_VALUES,
        help="Comma-separated list of HDBSCAN min_cluster_size values to sweep, folded into the "
        "same config grid as the PaCMAP hyperparameters above (default: 200)",
    )
    parser.add_argument(
        "--hdbscan-min-samples",
        type=int,
        default=HDBSCANClusterer.DEFAULT_MIN_CLUSTER_SIZE,
        help="Fixed HDBSCAN min_samples, not swept (default: 200)",
    )
    parser.add_argument(
        "--hdbscan-metric",
        type=str,
        default=HDBSCANClusterer.DEFAULT_METRIC,
        help="Fixed HDBSCAN distance metric, not swept (default: euclidean)",
    )
    parser.add_argument(
        "--hdbscan-cluster-selection-epsilon",
        type=float,
        default=HDBSCANClusterer.DEFAULT_CLUSTER_SELECTION_EPSILON,
        help="Fixed HDBSCAN cluster_selection_epsilon, not swept (default: 0.0)",
    )
    parser.add_argument(
        "--hdbscan-cluster-selection-method",
        type=str,
        choices=["eom", "leaf"],
        default=HDBSCANClusterer.DEFAULT_CLUSTER_SELECTION_METHOD,
        help="Fixed HDBSCAN cluster_selection_method, not swept (default: eom)",
    )
    parser.add_argument(
        "--hdbscan-alpha",
        type=float,
        default=HDBSCANClusterer.DEFAULT_ALPHA,
        help="Fixed HDBSCAN alpha, not swept (default: 1.0)",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="Override the default input embeddings Parquet path",
    )
    parser.add_argument(
        "--reduced-embeddings-output-path",
        type=Path,
        default=None,
        help="Override the default output reduced-embeddings Parquet path",
    )
    parser.add_argument(
        "--assignments-output-path",
        type=Path,
        default=None,
        help="Override the default output cluster-assignments Parquet path",
    )
    parser.add_argument(
        "--metrics-output-path",
        type=Path,
        default=None,
        help="Override the default output metrics JSON path",
    )
    return parser


def main() -> None:
    """CLI entry point: sweep, cluster, and write reduced embeddings/cluster assignments/metrics for one condition."""
    logging.basicConfig(level=logging.INFO)

    args = build_arg_parser().parse_args()

    run(
        args.condition,
        n_components_values=args.pacmap_n_components,
        n_neighbors_values=args.pacmap_n_neighbors,
        mn_ratio_values=args.pacmap_mn_ratio,
        fp_ratio_values=args.pacmap_fp_ratio,
        pacmap_distance=args.pacmap_distance,
        seeds=args.seeds,
        repeats_per_seed=args.repeats_per_seed,
        hdbscan_min_cluster_size_values=args.hdbscan_min_cluster_size,
        hdbscan_min_samples=args.hdbscan_min_samples,
        hdbscan_metric=args.hdbscan_metric,
        hdbscan_cluster_selection_epsilon=args.hdbscan_cluster_selection_epsilon,
        hdbscan_cluster_selection_method=args.hdbscan_cluster_selection_method,
        hdbscan_alpha=args.hdbscan_alpha,
        input_path=args.input_path,
        reduced_embeddings_output_path=args.reduced_embeddings_output_path,
        assignments_output_path=args.assignments_output_path,
        metrics_output_path=args.metrics_output_path,
    )


if __name__ == "__main__":
    main()
