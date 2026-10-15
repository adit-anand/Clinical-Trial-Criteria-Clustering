"""Cluster per-condition eligibility criteria embeddings via HDBSCAN, sweeping hyperparameters.

Reads the same embeddings Parquet file as `kmeans_clustering.py`
(``data/embeddings/<condition>_trials_embeddings.parquet``), drops rows whose
embedding failed, and fits `sklearn.cluster.HDBSCAN` once per combination in
the Cartesian product of the supplied `--min-cluster-size`/`--min-samples`/
`--cluster-selection-epsilon`/`--alpha` value lists (`--metric`/
`--cluster-selection-method` are held fixed across the whole sweep, not
swept). Passing a single value for each (the default) is simply a sweep of
size one — the module always goes through the same combination-evaluation
and winner-selection path, whether there's one candidate or fifty. Unlike
k-means, there is no `k` to choose for any individual fit — HDBSCAN discovers
clusters of varying density directly from the data, and explicitly labels
points that don't belong to any cluster as noise (``cluster_id == -1``).

Each combination's clustering performance metrics are logged as the sweep
runs; no separate sweep-results file is written. Once every combination has
been tried, the single best one — ranked by relative validity index (DBCV,
higher is better), tie-broken by noise fraction (lower is better) — has its
cluster assignments and metrics written:

    data/clustering/hdbscan/<condition>_trials_hdbscan_clusters.parquet
        One row per successfully-embedded criterion, carrying its assigned
        ``cluster_id`` (``-1`` for noise), HDBSCAN's cluster membership
        ``cluster_probability``, plus enough source columns (``nct_id``,
        ``criterion_text``, etc.) to join back to the embeddings file.

    data/clustering/hdbscan/<condition>_trials_hdbscan_cluster_metrics.json
        Clustering performance metrics (silhouette score, Davies-Bouldin
        index, Calinski-Harabasz index — computed only over non-noise
        points) plus noise statistics, run provenance, and
        ``relative_validity_index`` (the DBCV score that decided the
        winner), so clustering quality can be reviewed without re-running
        the clustering step.

A combination that's invalid for the dataset (e.g. `min_samples` exceeding
the sample count) is skipped with a logged reason rather than aborting the
rest of the sweep. A combination that fits but produces a degenerate result
— per `HDBSCANClusterer.compute_metrics`'s eligibility constraints (cluster
count outside `MIN_N_CLUSTERS`/`MAX_N_CLUSTERS`, a dominant cluster
exceeding `MAX_CLUSTER_SHARE`, or noise exceeding `MAX_NOISE_FRACTION`) —
is logged but never eligible to be the "best." If no
combination qualifies — including when only the single default combination
was tried and it's degenerate — nothing is written. The total number of
combinations (the product of all four swept lists' lengths) is capped at
`MAX_SWEEP_COMBINATIONS`; exceeding it raises an error before any fitting is
attempted.

Re-running this step on the same condition overwrites both existing output
files wholesale — there is no append/skip-if-exists/merge behavior.

Read/validate/write logic shared with the k-means clustering pipeline (see
`kmeans_clustering.py`) lives in `clustering_io.py`; this module holds only
HDBSCAN-specific logic (fitting, noise-aware metrics, probability output,
and the hyperparameter sweep itself). `hdbscan_sweep.py` is a thin,
backward-compatible alias for this module's CLI — all sweep logic lives
here, not there.
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
import pyarrow as pa
from hdbscan import validity
from sklearn.cluster import HDBSCAN
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

# Allow running this file directly (e.g. `uv run src/clustering/hdbscan_clustering.py`)
# as well as via `-m` (e.g. `uv run python -m src.clustering.hdbscan_clustering`).
# Direct execution leaves __package__ unset and never puts the repo root on
# sys.path, so the src.* imports below would otherwise fail with
# "ModuleNotFoundError: No module named 'src'"; -m already sets both up
# correctly, so this is a no-op in that case.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.paths import REPO_ROOT
from src.preprocessing.criteria_retrieval import EligibilityCriteriaFetcher
from src.clustering import clustering_io

OUTPUT_DIR = REPO_ROOT / "data" / "clustering" / "hdbscan"

logger = logging.getLogger(__name__)

# The total number of combinations (the product of all four swept lists'
# lengths) is capped here; exceeding it raises an error before any fitting
# is attempted.
MAX_SWEEP_COMBINATIONS = 50

HDBSCAN_CLUSTER_PARQUET_SCHEMA = pa.schema(
    [
        ("nct_id", pa.string()),
        ("study_first_posted_date", pa.string()),
        ("criterion_text", pa.string()),
        ("criterion_type", pa.string()),
        ("cluster_id", pa.int32()),
        ("cluster_probability", pa.float32()),
        ("embedding_backend", pa.string()),
        ("embedding_model", pa.string()),
        ("clustered_at", pa.timestamp("us", tz="UTC")),
    ]
)


class HDBSCANClusterer:
    """Fits and scores a single HDBSCAN hyperparameter combination.

    Unlike `KMeansClusterer`, there is no "auto vs. fixed k" branching —
    HDBSCAN always discovers its own cluster count from the data; only its
    density hyperparameters are configurable. This class fits and scores one
    combination; `run_sweep` (module-level, below) constructs one instance
    per combination it tries.

    Typical usage:
        clusterer = HDBSCANClusterer()
        model = clusterer.fit_hdbscan(embedding_matrix)
    """

    DEFAULT_MIN_CLUSTER_SIZE = 200
    DEFAULT_METRIC = "euclidean"
    DEFAULT_CLUSTER_SELECTION_EPSILON = 0.0
    DEFAULT_CLUSTER_SELECTION_METHOD = "eom"
    DEFAULT_ALPHA = 1.0

    # A result's non-noise cluster count must fall in this range to be
    # considered valid/eligible to win a sweep — too few clusters is as
    # undesirable a result as too many, for this pipeline's purposes.
    MIN_N_CLUSTERS = 10
    MAX_N_CLUSTERS = 100

    # No single cluster may hold more than this share of all data points
    # (a lopsided result where one cluster swallows most of the dataset is
    # treated as degenerate, not a meaningful clustering), and the overall
    # noise fraction may not exceed this same limit either.
    MAX_CLUSTER_SHARE = 0.20
    MAX_NOISE_FRACTION = 0.20

    def __init__(
        self,
        min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
        min_samples: int | None = None,
        metric: str = DEFAULT_METRIC,
        cluster_selection_epsilon: float = DEFAULT_CLUSTER_SELECTION_EPSILON,
        cluster_selection_method: str = DEFAULT_CLUSTER_SELECTION_METHOD,
        alpha: float = DEFAULT_ALPHA,
    ):
        """Store HDBSCAN configuration for this clusterer.

        Defaults match `sklearn.cluster.HDBSCAN`'s own defaults, so omitting
        a flag behaves exactly as sklearn's default construction would.

        Args:
            min_cluster_size: The minimum size of a group of points to be
                considered a cluster.
            min_samples: The number of samples in a neighborhood for a point
                to be a core point. If `None`, sklearn defaults this to
                `min_cluster_size`.
            metric: The distance metric to use.
            cluster_selection_epsilon: A distance threshold below which
                clusters are merged (0.0 disables this).
            cluster_selection_method: `"eom"` (excess of mass) or `"leaf"`.
            alpha: Distance scaling parameter used in robust single linkage.
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.cluster_selection_method = cluster_selection_method
        self.alpha = alpha

    def validate_hyperparameters(self, n_samples: int) -> None:
        """Confirm the configured hyperparameters are usable for this dataset.

        Replicates sklearn's own fallback (`min_samples` defaults to
        `min_cluster_size` when unset) so this pre-flight check and
        `HDBSCAN.fit`'s internal behavior never disagree about what
        "effective `min_samples`" means.

        Args:
            n_samples: Number of embeddings available to cluster.

        Raises:
            ValueError: If `min_cluster_size` is less than 1, if `alpha` is
                not greater than 0, or if the effective `min_samples`
                (explicit value, or `min_cluster_size` if unset) exceeds
                `n_samples` — naming both numbers, pre-empting sklearn's own
                less-contextualized `ValueError` for the same condition
                (confirmed empirically to occur otherwise).
        """
        if self.min_cluster_size < 1:
            raise ValueError(f"min_cluster_size must be at least 1, got {self.min_cluster_size}")
        if self.alpha <= 0:
            raise ValueError(f"alpha must be greater than 0, got {self.alpha}")
        effective_min_samples = self.min_samples if self.min_samples is not None else self.min_cluster_size
        if effective_min_samples > n_samples:
            raise ValueError(
                f"min_samples={effective_min_samples} (min_cluster_size={self.min_cluster_size}, "
                f"min_samples={self.min_samples}) exceeds the number of available embeddings "
                f"({n_samples}); reduce --min-cluster-size/--min-samples or provide more data"
            )

    def fit_hdbscan(self, embedding_matrix: np.ndarray) -> HDBSCAN:
        """Fit an HDBSCAN model over an embedding matrix.

        Args:
            embedding_matrix: Array of shape `(n_samples, embedding_dim)`.

        Returns:
            The fitted `HDBSCAN` estimator (`.labels_` and `.probabilities_`
            are available on it).
        """
        model = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            cluster_selection_method=self.cluster_selection_method,
            alpha=self.alpha,
            copy=True,
        )
        model.fit(embedding_matrix)
        return model

    def compute_metrics(self, embedding_matrix: np.ndarray, labels: np.ndarray) -> dict:
        """Compute clustering performance metrics for a fitted HDBSCAN result.

        Silhouette score, the Davies-Bouldin index, and the Calinski-Harabasz
        index are computed only over non-noise points (`label >= 0`), guarded
        the same way the k-means pipeline guards its own metrics (requires
        2 <= n_clusters_actual <= n_non_noise - 1 among non-noise points);
        when that guard fails (e.g. an all-noise or single-cluster result),
        the three metrics are reported as `None` with an explanatory note
        rather than letting sklearn raise. `noise_count`/`noise_fraction` are
        always reported, regardless of whether the metrics could be computed.

        Independently of whether the metrics were computable, a result is
        only considered valid/eligible to win a sweep if it also satisfies
        three additional, business-level constraints — each violated
        independently of the others, and independently of whether the
        metrics above were computable:

        - `n_clusters_actual` falls within `[MIN_N_CLUSTERS, MAX_N_CLUSTERS]`
          — too few clusters is treated as degenerate just as too many are.
        - No single cluster holds more than `MAX_CLUSTER_SHARE` of all
          points — a lopsided result where one cluster swallows most of the
          dataset isn't a meaningful clustering.
        - `noise_fraction` does not exceed `MAX_NOISE_FRACTION`.

        `metrics` are still reported (not `None`) whenever they're
        computable, even if one of the above constraints fails, so a
        rejected result remains inspectable; only `metrics_valid` reflects
        the rejection, with its own explanatory note per violated constraint.

        Args:
            embedding_matrix: The matrix HDBSCAN was fit on.
            labels: `model.labels_` (may include `-1` for noise).

        Returns:
            A dict with keys `metrics` (the 3 metric values, `None` only if
            they couldn't be computed at all), `notes` (list of human-readable
            caveats), `n_clusters_actual` (count of distinct non-negative
            labels), `noise_count`, `noise_fraction`, and `metrics_valid`
            (True iff the metrics were computable *and* `n_clusters_actual`,
            the largest cluster's share of all points, and `noise_fraction`
            all satisfy the constraints described above).
        """
        n_samples = embedding_matrix.shape[0]
        noise_mask = labels == -1
        noise_count = int(noise_mask.sum())
        noise_fraction = noise_count / n_samples

        non_noise_matrix = embedding_matrix[~noise_mask]
        non_noise_labels = labels[~noise_mask]
        n_non_noise = non_noise_matrix.shape[0]
        n_clusters_actual = len(set(non_noise_labels.tolist()))
        notes: list[str] = []

        if n_non_noise > 0:
            _, cluster_counts = np.unique(non_noise_labels, return_counts=True)
            max_cluster_share = float(cluster_counts.max()) / n_samples
        else:
            max_cluster_share = 0.0

        can_compute = 2 <= n_clusters_actual <= n_non_noise - 1
        if can_compute:
            metrics = {
                "silhouette_score": float(silhouette_score(non_noise_matrix, non_noise_labels)),
                "davies_bouldin_index": float(davies_bouldin_score(non_noise_matrix, non_noise_labels)),
                "calinski_harabasz_index": float(calinski_harabasz_score(non_noise_matrix, non_noise_labels)),
            }
        else:
            metrics = {
                "silhouette_score": None,
                "davies_bouldin_index": None,
                "calinski_harabasz_index": None,
            }
            notes.append(
                "Skipped silhouette/Davies-Bouldin/Calinski-Harabasz scores: require "
                f"2 <= n_clusters <= n_non_noise - 1 among non-noise points, but got "
                f"n_clusters_actual={n_clusters_actual} with n_non_noise={n_non_noise} "
                f"(noise_count={noise_count} of {n_samples})"
            )

        in_desired_cluster_count_range = self.MIN_N_CLUSTERS <= n_clusters_actual <= self.MAX_N_CLUSTERS
        if not in_desired_cluster_count_range:
            notes.append(
                f"n_clusters_actual={n_clusters_actual} is outside the desired range "
                f"[{self.MIN_N_CLUSTERS}, {self.MAX_N_CLUSTERS}]; treating this result as invalid/degenerate"
            )

        within_max_cluster_share = max_cluster_share <= self.MAX_CLUSTER_SHARE
        if not within_max_cluster_share:
            notes.append(
                f"the largest cluster holds {max_cluster_share:.1%} of all points, exceeding "
                f"MAX_CLUSTER_SHARE={self.MAX_CLUSTER_SHARE:.0%}; treating this result as invalid/degenerate"
            )

        within_max_noise_fraction = noise_fraction <= self.MAX_NOISE_FRACTION
        if not within_max_noise_fraction:
            notes.append(
                f"noise_fraction={noise_fraction:.1%} exceeds MAX_NOISE_FRACTION="
                f"{self.MAX_NOISE_FRACTION:.0%}; treating this result as invalid/degenerate"
            )

        metrics_valid = (
            can_compute
            and in_desired_cluster_count_range
            and within_max_cluster_share
            and within_max_noise_fraction
        )

        return {
            "metrics": metrics,
            "notes": notes,
            "n_clusters_actual": n_clusters_actual,
            "noise_count": noise_count,
            "noise_fraction": noise_fraction,
            "metrics_valid": metrics_valid,
        }

    def build_assignment_rows(
        self, table: pa.Table, labels: np.ndarray, probabilities: np.ndarray, clustered_at: datetime
    ) -> list[dict]:
        """Zip each surviving row's traceability columns with its cluster label and probability.

        Args:
            table: The table `labels`/`probabilities` were produced from (row
                order must match).
            labels: One cluster id per row of `table` (`-1` for noise).
            probabilities: One cluster membership probability per row
                (`0.0` for noise points).
            clustered_at: Wall-clock time this clustering run completed.

        Returns:
            One output dict per row, matching
            `HDBSCAN_CLUSTER_PARQUET_SCHEMA`'s columns.
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
                    "cluster_id": int(labels[i]),
                    "cluster_probability": float(probabilities[i]),
                    "embedding_backend": columns["embedding_backend"][i],
                    "embedding_model": columns["embedding_model"][i],
                    "clustered_at": clustered_at,
                }
            )
        return rows

    def build_metrics_dict(
        self,
        condition: str,
        slug: str,
        input_path: Path,
        assignments_output_path: Path,
        n_samples: int,
        n_samples_excluded: int,
        embedding_dim: int,
        embedding_backend: str | None,
        embedding_model: str | None,
        metrics_result: dict,
        labels: np.ndarray,
        clustered_at: datetime,
    ) -> dict:
        """Assemble the full metrics JSON payload for one clustering run.

        Args:
            condition: Condition string passed to `run_sweep`.
            slug: `condition`'s filesystem-safe slug.
            input_path: The embeddings Parquet path that was read.
            assignments_output_path: Where cluster assignments were written.
            n_samples: Number of successfully-embedded rows clustered.
            n_samples_excluded: Number of input rows dropped for having
                failed embeddings.
            embedding_dim: Common embedding dimensionality.
            embedding_backend: Backend recorded on the clustered rows.
            embedding_model: Model recorded on the clustered rows.
            metrics_result: The dict returned by `compute_metrics`.
            labels: `model.labels_`, used to compute per-cluster sizes
                (including a `-1` entry for noise, if any).
            clustered_at: Wall-clock time this clustering run completed.

        Returns:
            A JSON-serializable dict matching the documented metrics schema.
        """
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
            cluster_sizes=clustering_io.compute_cluster_sizes(labels),
            clustered_at=clustered_at,
        )
        return {
            **base,
            "n_clusters_actual": metrics_result["n_clusters_actual"],
            "noise_count": metrics_result["noise_count"],
            "noise_fraction": metrics_result["noise_fraction"],
            "hdbscan_params": {
                "min_cluster_size": self.min_cluster_size,
                "min_samples": self.min_samples,
                "metric": self.metric,
                "cluster_selection_epsilon": self.cluster_selection_epsilon,
                "cluster_selection_method": self.cluster_selection_method,
                "alpha": self.alpha,
            },
            "metrics": metrics_result["metrics"],
            "metrics_valid": metrics_result["metrics_valid"],
            "notes": metrics_result["notes"],
        }


@dataclass(frozen=True)
class SweepCandidate:
    """The outcome of evaluating one hyperparameter combination.

    Attributes:
        hyperparams: The swept `min_cluster_size`/`min_samples`/
            `cluster_selection_epsilon`/`alpha` values this candidate used.
        status: `"invalid"` if `HDBSCANClusterer.validate_hyperparameters`
            rejected the combination for this dataset (never fit);
            `"degenerate"` if fitting succeeded but `metrics_valid` came back
            `False` (see `HDBSCANClusterer.compute_metrics`'s eligibility
            constraints); `"ok"` otherwise.
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

    This is the module's only orchestration entry point — a single value per
    list (the CLI defaults) is simply a sweep of size one; the winner-
    selection and output-writing logic is identical regardless of how many
    combinations were tried.

    Args:
        condition: Condition string, e.g. "breast cancer" — the same string
            passed to `criteria_embedding.py`.
        min_cluster_size_values: Candidate `min_cluster_size` values.
        min_samples_values: Candidate `min_samples` values.
        cluster_selection_epsilon_values: Candidate `cluster_selection_epsilon` values.
        alpha_values: Candidate `alpha` values.
        metric: Distance metric to use, held fixed across the sweep.
        cluster_selection_method: Held fixed across the sweep.
        input_path: Embeddings Parquet to read. If omitted, defaults to
            the embeddings directory's `{slug}_trials_embeddings.parquet`.
        assignments_output_path: Where to write the winning combination's
            cluster assignments. If omitted, defaults to
            ``OUTPUT_DIR / f"{slug}_trials_hdbscan_clusters.parquet"``.
        metrics_output_path: Where to write the winning combination's
            metrics. If omitted, defaults to
            ``OUTPUT_DIR / f"{slug}_trials_hdbscan_cluster_metrics.json"``.

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
        description="Cluster a condition's criterion embeddings via HDBSCAN, sweeping hyperparameters."
    )
    parser.add_argument(
        "condition",
        help='Condition to cluster, e.g. "breast cancer" (same string passed to criteria_embedding.py)',
    )
    parser.add_argument(
        "--min-cluster-size",
        type=parse_int_list,
        default=[HDBSCANClusterer.DEFAULT_MIN_CLUSTER_SIZE],
        help="Comma-separated list of min_cluster_size values to try (default: 200)",
    )
    parser.add_argument(
        "--min-samples",
        type=parse_int_list,
        default=[HDBSCANClusterer.DEFAULT_MIN_CLUSTER_SIZE],
        help="Comma-separated list of min_samples values to try (default: 200)",
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
    """CLI entry point: sweep, cluster, and write cluster assignments/metrics for one condition."""
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
