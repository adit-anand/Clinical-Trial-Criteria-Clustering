"""Shared HDBSCAN fitting/metrics library code for the reducer-based clustering pipelines.

Plain, standalone HDBSCAN clustering is not a pipeline in this repo — only
`pacmap_hdbscan_clustering.py` and `umap_hdbscan_clustering.py` (reduce
first, then cluster with HDBSCAN) use HDBSCAN, both via `reduction_sweep.py`.
This module holds the logic those two share: `HDBSCANClusterer` (fitting,
noise-aware metrics, probability output) and `compute_relative_validity_index`
(DBCV). It is a library module, not a CLI entry point.
"""

import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
from hdbscan import validity
from sklearn.cluster import HDBSCAN
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from src.clustering import clustering_io

# Shared with pacmap_hdbscan_clustering.py/umap_hdbscan_clustering.py: splits
# a reducer-pipeline trial's combined search_space dict into HDBSCAN's own
# hyperparameters vs. the reducer's.
HDBSCAN_PARAM_NAMES = frozenset(
    {"min_cluster_size", "min_samples", "cluster_selection_epsilon", "cluster_selection_method", "alpha"}
)

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

    Each reducer pipeline's Optuna objective constructs one instance per
    trial (see `pacmap_hdbscan_clustering.py`/`umap_hdbscan_clustering.py`).

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
    # considered valid/eligible to win a search — too few clusters is as
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
                clusters are merged (0.0 disables this). Forced to `0.0`
                whenever `metric == "cosine"` — see the note below.
            cluster_selection_method: `"eom"` (excess of mass) or `"leaf"`.
            alpha: Distance scaling parameter used in robust single linkage.
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric
        # Workaround for a confirmed bug in the installed scikit-learn
        # (1.9.1): HDBSCAN(metric="cosine", cluster_selection_epsilon=<any
        # nonzero value>).fit(...) raises
        # "TypeError: only 0-dimensional arrays can be converted to Python
        # scalars" inside sklearn's own _tree.pyx epsilon-merge code,
        # regardless of cluster_selection_method or the input data — cosine
        # distance's epsilon-merge path is simply broken in this version.
        # Clamping here (rather than only at the `fit_hdbscan` call site)
        # means every downstream consumer of `self.cluster_selection_epsilon`
        # (metrics-JSON provenance, DBCV, logging) reports the value that
        # was actually used, not what a caller originally asked for.
        self.cluster_selection_epsilon = 0.0 if metric == "cosine" else cluster_selection_epsilon
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
        to require 2 <= n_clusters_actual <= n_non_noise - 1 among non-noise
        points; when that guard fails (e.g. an all-noise or single-cluster result),
        the three metrics are reported as `None` with an explanatory note
        rather than letting sklearn raise. `noise_count`/`noise_fraction` are
        always reported, regardless of whether the metrics could be computed.

        Independently of whether the metrics were computable, a result is
        only considered valid/eligible to win a search if it also satisfies
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
        label: str,
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
            label: Human-readable label for this run, derived from
                `input_path`'s filename.
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
            label=label,
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


def compute_relative_validity_index(embedding_matrix: np.ndarray, labels: np.ndarray, metric: str) -> float | None:
    """Compute the DBCV relative validity index for a fitted HDBSCAN result.

    Shared helper reused by `reduction_sweep.evaluate_repeat` — do not
    duplicate this logic there.

    Args:
        embedding_matrix: The matrix HDBSCAN was fit on.
        labels: `model.labels_` (may include `-1` for noise).
        metric: The distance metric HDBSCAN was fit with.

    Returns:
        The DBCV score, or `None` if it's not computable (a NaN result, or
        any exception from `hdbscan.validity.validity_index`) — never raises.
    """
    try:
        # hdbscan.validity.validity_index requires float64 input (its Cython
        # core is compiled against double_t); embedding matrices here are
        # typically float32, so cast just for this call.
        value = float(validity.validity_index(embedding_matrix.astype(np.float64), labels, metric=metric))
        if math.isnan(value):
            return None
        return value
    except Exception:  # noqa: BLE001 - DBCV computation failure shouldn't abort a trial
        return None
