"""Sweep HDBSCAN hyperparameters for a condition and keep the best result.

Reads the same embeddings Parquet file `hdbscan_clustering.py` reads, then
fits HDBSCAN once per combination of `min_cluster_size`, `min_samples`,
`cluster_selection_epsilon`, and `alpha` drawn from the Cartesian product of
CLI-supplied value lists (`metric`/`cluster_selection_method` are held fixed
across the sweep). Every combination's clustering performance metrics are
logged as the sweep runs; no separate sweep-results file is written. Once
the sweep finishes, the single best combination — ranked by relative
validity index (DBCV, higher is better), tie-broken by noise fraction (lower
is better) — has its cluster assignments and metrics written to the exact
same conventional output location `hdbscan_clustering.py` itself would use,
overwriting any prior output for the condition. The written metrics JSON
carries one extra field beyond `hdbscan_clustering.py`'s own schema,
`relative_validity_index`, recording the DBCV score that decided the winner
(otherwise only visible in the sweep's log output).

A combination that's invalid for the dataset (e.g. `min_samples` exceeding
the sample count) is skipped with a logged reason rather than aborting the
rest of the sweep. A combination that fits but produces a degenerate result
(e.g. all-noise) is logged but never eligible to be the "best." If no
combination qualifies, nothing is written.

Fitting and metrics logic is reused unmodified from `HDBSCANClusterer`
(see `hdbscan_clustering.py`); this module only orchestrates which
combinations to try and picks the winner.
"""

import argparse
import itertools
import logging
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
from hdbscan import validity

# Allow running this file directly (e.g. `uv run src/clustering/hdbscan_sweep.py`)
# as well as via `-m` (e.g. `uv run python -m src.clustering.hdbscan_sweep`).
# Direct execution leaves __package__ unset and never puts the repo root on
# sys.path, so the src.* imports below would otherwise fail with
# "ModuleNotFoundError: No module named 'src'"; -m already sets both up
# correctly, so this is a no-op in that case.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.preprocessing.criteria_retrieval import EligibilityCriteriaFetcher
from src.clustering import clustering_io
from src.clustering.hdbscan_clustering import HDBSCAN_CLUSTER_PARQUET_SCHEMA, OUTPUT_DIR, HDBSCANClusterer

logger = logging.getLogger(__name__)

MAX_SWEEP_COMBINATIONS = 50


@dataclass(frozen=True)
class SweepCandidate:
    """The outcome of evaluating one hyperparameter combination.

    Attributes:
        hyperparams: The swept `min_cluster_size`/`min_samples`/
            `cluster_selection_epsilon`/`alpha` values this candidate used.
        status: `"invalid"` if `HDBSCANClusterer.validate_hyperparameters`
            rejected the combination for this dataset (never fit);
            `"degenerate"` if fitting succeeded but produced no usable
            quality metrics (e.g. all-noise); `"ok"` otherwise.
        detail: Human-readable explanation, set for `"invalid"`/`"degenerate"`.
        metrics_result: `HDBSCANClusterer.compute_metrics(...)`'s return
            value, set whenever fitting was attempted (`"degenerate"`/`"ok"`).
        relative_validity: The DBCV relative validity index, set only for
            `"ok"` candidates where it was itself successfully computed.
        labels: The fitted model's cluster labels, kept so the eventual
            winner never needs to be refit.
        probabilities: The fitted model's cluster membership probabilities.
    """

    hyperparams: dict
    status: Literal["ok", "invalid", "degenerate"]
    detail: str | None = None
    metrics_result: dict | None = None
    relative_validity: float | None = None
    labels: np.ndarray | None = None
    probabilities: np.ndarray | None = None


def parse_int_list(value: str) -> list[int]:
    """Parse a comma-separated CLI value into a list of ints.

    Args:
        value: e.g. `"5,10,15"`.

    Returns:
        `[5, 10, 15]`.

    Raises:
        argparse.ArgumentTypeError: If any comma-separated entry isn't a
            valid int, naming the offending entry.
    """
    parts = [p.strip() for p in value.split(",")]
    try:
        return [int(p) for p in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a comma-separated list of integers, got {value!r}") from exc


def parse_float_list(value: str) -> list[float]:
    """Parse a comma-separated CLI value into a list of floats.

    Args:
        value: e.g. `"0.0,0.1,0.2"`.

    Returns:
        `[0.0, 0.1, 0.2]`.

    Raises:
        argparse.ArgumentTypeError: If any comma-separated entry isn't a
            valid float, naming the offending entry.
    """
    parts = [p.strip() for p in value.split(",")]
    try:
        return [float(p) for p in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a comma-separated list of numbers, got {value!r}") from exc


def generate_combinations(
    min_cluster_size_values: list[int],
    min_samples_values: list[int],
    cluster_selection_epsilon_values: list[float],
    alpha_values: list[float],
) -> list[dict]:
    """Build the Cartesian product of the four swept hyperparameters.

    Args:
        min_cluster_size_values: Candidate `min_cluster_size` values.
        min_samples_values: Candidate `min_samples` values.
        cluster_selection_epsilon_values: Candidate `cluster_selection_epsilon` values.
        alpha_values: Candidate `alpha` values.

    Returns:
        One dict per combination, each with keys `min_cluster_size`,
        `min_samples`, `cluster_selection_epsilon`, `alpha`.

    Raises:
        ValueError: If the product size exceeds `MAX_SWEEP_COMBINATIONS`,
            naming the actual count and the limit.
    """
    combos = [
        {
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "cluster_selection_epsilon": cluster_selection_epsilon,
            "alpha": alpha,
        }
        for min_cluster_size, min_samples, cluster_selection_epsilon, alpha in itertools.product(
            min_cluster_size_values, min_samples_values, cluster_selection_epsilon_values, alpha_values
        )
    ]
    if len(combos) > MAX_SWEEP_COMBINATIONS:
        raise ValueError(
            f"Requested sweep has {len(combos)} combination(s), exceeding the limit of "
            f"{MAX_SWEEP_COMBINATIONS}; narrow one or more of --min-cluster-size/--min-samples/"
            "--cluster-selection-epsilon/--alpha"
        )
    return combos


def evaluate_combination(
    hyperparams: dict,
    metric: str,
    cluster_selection_method: str,
    embedding_matrix: np.ndarray,
    n_samples: int,
) -> SweepCandidate:
    """Fit and score HDBSCAN for one hyperparameter combination.

    Args:
        hyperparams: One combination from `generate_combinations`.
        metric: Distance metric, held fixed across the sweep.
        cluster_selection_method: Held fixed across the sweep.
        embedding_matrix: Array of shape `(n_samples, embedding_dim)`.
        n_samples: `embedding_matrix.shape[0]`.

    Returns:
        A `SweepCandidate` reflecting whether the combination was invalid,
        degenerate, or a usable result.
    """
    clusterer = HDBSCANClusterer(
        min_cluster_size=hyperparams["min_cluster_size"],
        min_samples=hyperparams["min_samples"],
        metric=metric,
        cluster_selection_epsilon=hyperparams["cluster_selection_epsilon"],
        cluster_selection_method=cluster_selection_method,
        alpha=hyperparams["alpha"],
    )

    try:
        clusterer.validate_hyperparameters(n_samples)
    except ValueError as exc:
        return SweepCandidate(hyperparams=hyperparams, status="invalid", detail=str(exc))

    model = clusterer.fit_hdbscan(embedding_matrix)
    labels = model.labels_
    probabilities = model.probabilities_
    metrics_result = clusterer.compute_metrics(embedding_matrix, labels)

    if not metrics_result["metrics_valid"]:
        return SweepCandidate(
            hyperparams=hyperparams,
            status="degenerate",
            detail="; ".join(metrics_result["notes"]) or "No usable quality metrics for this combination",
            metrics_result=metrics_result,
            labels=labels,
            probabilities=probabilities,
        )

    try:
        # hdbscan.validity.validity_index requires float64 input (its Cython
        # core is compiled against double_t); clustering_io's embedding
        # matrices are float32, so cast just for this call.
        relative_validity = float(
            validity.validity_index(embedding_matrix.astype(np.float64), labels, metric=metric)
        )
        if math.isnan(relative_validity):
            relative_validity = None
            metrics_result["notes"].append("Relative validity index (DBCV) computed to NaN for this combination")
    except Exception as exc:  # noqa: BLE001 - DBCV computation failure shouldn't abort the sweep
        relative_validity = None
        metrics_result["notes"].append(f"Could not compute relative validity index (DBCV): {exc}")

    return SweepCandidate(
        hyperparams=hyperparams,
        status="ok",
        metrics_result=metrics_result,
        relative_validity=relative_validity,
        labels=labels,
        probabilities=probabilities,
    )


def select_best(candidates: list[SweepCandidate]) -> SweepCandidate | None:
    """Pick the best candidate by relative validity index, tie-broken by noise fraction.

    Args:
        candidates: As produced by `evaluate_combination`, one per combination.

    Returns:
        The `"ok"` candidate with the highest `relative_validity` (ties
        broken by the lowest `noise_fraction`), or `None` if no candidate
        has a computed relative validity index.
    """
    eligible = [c for c in candidates if c.status == "ok" and c.relative_validity is not None]
    if not eligible:
        return None
    return min(eligible, key=lambda c: (-c.relative_validity, c.metrics_result["noise_fraction"]))


def _log_candidate(index: int, total: int, candidate: SweepCandidate) -> None:
    """Log one INFO line summarizing a single evaluated combination."""
    if candidate.status == "invalid":
        logger.info(
            "evaluate_combination: [%d/%d] %s -> invalid (%s)",
            index, total, candidate.hyperparams, candidate.detail,
        )
        return
    if candidate.status == "degenerate":
        logger.info(
            "evaluate_combination: [%d/%d] %s -> degenerate, noise=%.1f%% (%s)",
            index, total, candidate.hyperparams,
            candidate.metrics_result["noise_fraction"] * 100, candidate.detail,
        )
        return
    logger.info(
        "evaluate_combination: [%d/%d] %s -> ok, clusters=%d, noise=%.1f%%, silhouette=%s, relative_validity=%s",
        index, total, candidate.hyperparams,
        candidate.metrics_result["n_clusters_actual"],
        candidate.metrics_result["noise_fraction"] * 100,
        candidate.metrics_result["metrics"]["silhouette_score"],
        candidate.relative_validity,
    )


def run_sweep(
    condition: str,
    min_cluster_size_values: list[int],
    min_samples_values: list[int],
    cluster_selection_epsilon_values: list[float],
    alpha_values: list[float],
    metric: str = HDBSCANClusterer.DEFAULT_METRIC,
    cluster_selection_method: str = HDBSCANClusterer.DEFAULT_CLUSTER_SELECTION_METHOD,
    input_path: Path | None = None,
    assignments_output_path: Path | None = None,
    metrics_output_path: Path | None = None,
) -> tuple[Path, Path] | None:
    """Sweep HDBSCAN hyperparameters for a condition and write the best result.

    Args:
        condition: Condition string, e.g. "breast cancer" — the same string
            passed to `criteria_embedding.py`/`hdbscan_clustering.py`.
        min_cluster_size_values: Candidate `min_cluster_size` values.
        min_samples_values: Candidate `min_samples` values.
        cluster_selection_epsilon_values: Candidate `cluster_selection_epsilon` values.
        alpha_values: Candidate `alpha` values.
        metric: Distance metric, held fixed across the sweep.
        cluster_selection_method: Held fixed across the sweep.
        input_path: Embeddings Parquet to read. If omitted, defaults as
            `hdbscan_clustering.py` itself does.
        assignments_output_path: Where to write the winning combination's
            cluster assignments. If omitted, defaults to the condition's
            normal HDBSCAN output path.
        metrics_output_path: Where to write the winning combination's
            metrics. If omitted, defaults to the condition's normal HDBSCAN
            metrics path.

    Returns:
        `(assignments_output_path, metrics_output_path)` for the winning
        combination, or `None` if no combination produced a usable result
        (in which case nothing is written).

    Raises:
        FileNotFoundError: If the resolved input Parquet does not exist.
        ValueError: If there are no successfully-embedded rows to cluster,
            if embedding dimensionality is inconsistent, or if the requested
            sweep has more than `MAX_SWEEP_COMBINATIONS` combinations.
    """
    slug = EligibilityCriteriaFetcher.condition_to_slug(condition)
    default_input_path, default_assignments_path, default_metrics_path = clustering_io.resolve_default_paths(
        condition, OUTPUT_DIR, "_trials_hdbscan_clusters.parquet", "_trials_hdbscan_cluster_metrics.json"
    )
    if input_path is None:
        input_path = default_input_path
    if assignments_output_path is None:
        assignments_output_path = default_assignments_path
    if metrics_output_path is None:
        metrics_output_path = default_metrics_path

    table = clustering_io.read_embeddings_parquet(input_path)
    total_rows = table.num_rows
    filtered = clustering_io.filter_successful_rows(table)
    n_samples = filtered.num_rows
    n_excluded = total_rows - n_samples

    embedding_dim = clustering_io.validate_embedding_dimensions(filtered)
    embedding_matrix = clustering_io.extract_embedding_matrix(filtered)

    combos = generate_combinations(
        min_cluster_size_values, min_samples_values, cluster_selection_epsilon_values, alpha_values
    )
    logger.info("run_sweep: evaluating %d combination(s) for %r (n_samples=%d)", len(combos), condition, n_samples)

    candidates = []
    for index, hyperparams in enumerate(combos, start=1):
        candidate = evaluate_combination(hyperparams, metric, cluster_selection_method, embedding_matrix, n_samples)
        _log_candidate(index, len(combos), candidate)
        candidates.append(candidate)

    best = select_best(candidates)
    if best is None:
        logger.warning(
            "run_sweep: no combination produced a usable relative validity index for %r; writing nothing",
            condition,
        )
        return None

    clustered_at = datetime.now(timezone.utc)
    clusterer = HDBSCANClusterer(
        min_cluster_size=best.hyperparams["min_cluster_size"],
        min_samples=best.hyperparams["min_samples"],
        metric=metric,
        cluster_selection_epsilon=best.hyperparams["cluster_selection_epsilon"],
        cluster_selection_method=cluster_selection_method,
        alpha=best.hyperparams["alpha"],
    )
    rows = clusterer.build_assignment_rows(filtered, best.labels, best.probabilities, clustered_at)
    clustering_io.write_parquet(rows, HDBSCAN_CLUSTER_PARQUET_SCHEMA, assignments_output_path)

    embedding_backend = filtered.column("embedding_backend")[0].as_py()
    embedding_model = filtered.column("embedding_model")[0].as_py()
    metrics_dict = clusterer.build_metrics_dict(
        condition=condition,
        slug=slug,
        input_path=input_path,
        assignments_output_path=assignments_output_path,
        n_samples=n_samples,
        n_samples_excluded=n_excluded,
        embedding_dim=embedding_dim,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
        metrics_result=best.metrics_result,
        labels=best.labels,
        clustered_at=clustered_at,
    )
    metrics_dict["relative_validity_index"] = best.relative_validity
    clustering_io.write_json(metrics_dict, metrics_output_path)

    logger.info(
        "run_sweep: best combination for %r is %s (relative_validity=%.4f, noise=%.1f%%) -> %s, %s",
        condition, best.hyperparams, best.relative_validity,
        best.metrics_result["noise_fraction"] * 100, assignments_output_path, metrics_output_path,
    )
    return assignments_output_path, metrics_output_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for this module.

    Returns:
        An ``argparse.ArgumentParser`` accepting a positional ``condition``
        argument plus comma-separated value lists for the swept
        hyperparameters, fixed-value flags for the un-swept ones, and
        path-override options.
    """
    parser = argparse.ArgumentParser(
        description="Sweep HDBSCAN hyperparameters for a condition and keep the best result."
    )
    parser.add_argument(
        "condition",
        help='Condition to sweep, e.g. "breast cancer" (same string passed to criteria_embedding.py)',
    )
    parser.add_argument(
        "--min-cluster-size",
        type=parse_int_list,
        default=[HDBSCANClusterer.DEFAULT_MIN_CLUSTER_SIZE],
        help="Comma-separated list of min_cluster_size values to try (default: 5)",
    )
    parser.add_argument(
        "--min-samples",
        type=parse_int_list,
        default=[HDBSCANClusterer.DEFAULT_MIN_CLUSTER_SIZE],
        help="Comma-separated list of min_samples values to try (default: 5)",
    )
    parser.add_argument(
        "--cluster-selection-epsilon",
        type=parse_float_list,
        default=[HDBSCANClusterer.DEFAULT_CLUSTER_SELECTION_EPSILON],
        help="Comma-separated list of cluster_selection_epsilon values to try (default: 0.0)",
    )
    parser.add_argument(
        "--alpha",
        type=parse_float_list,
        default=[HDBSCANClusterer.DEFAULT_ALPHA],
        help="Comma-separated list of alpha values to try (default: 1.0)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=HDBSCANClusterer.DEFAULT_METRIC,
        help="Distance metric to use, held fixed across the sweep (default: euclidean)",
    )
    parser.add_argument(
        "--cluster-selection-method",
        type=str,
        choices=["eom", "leaf"],
        default=HDBSCANClusterer.DEFAULT_CLUSTER_SELECTION_METHOD,
        help="Cluster selection method, held fixed across the sweep: 'eom' or 'leaf' (default: eom)",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="Override the default input embeddings Parquet path",
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
    logging.basicConfig(level=logging.INFO)
    args = build_arg_parser().parse_args()
    run_sweep(
        args.condition,
        min_cluster_size_values=args.min_cluster_size,
        min_samples_values=args.min_samples,
        cluster_selection_epsilon_values=args.cluster_selection_epsilon,
        alpha_values=args.alpha,
        metric=args.metric,
        cluster_selection_method=args.cluster_selection_method,
        input_path=args.input_path,
        assignments_output_path=args.assignments_output_path,
        metrics_output_path=args.metrics_output_path,
    )


if __name__ == "__main__":
    main()
