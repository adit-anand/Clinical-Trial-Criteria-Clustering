"""Cluster per-condition eligibility criteria embeddings via HDBSCAN.

Reads the same embeddings Parquet file as `criteria_clustering.py`
(``data/embeddings/<condition>_trials_embeddings.parquet``), drops rows whose
embedding failed, and fits a `sklearn.cluster.HDBSCAN` model over the
remaining embedding vectors. Unlike k-means, HDBSCAN discovers clusters of
varying density directly from the data — there is no `k` to choose — and
explicitly labels points that don't belong to any cluster as noise
(``cluster_id == -1``). The run writes two output artifacts:

    data/clustering/hdbscan/<condition>_trials_hdbscan_clusters.parquet
        One row per successfully-embedded criterion, carrying its assigned
        ``cluster_id`` (``-1`` for noise), HDBSCAN's cluster membership
        ``cluster_probability``, plus enough source columns (``nct_id``,
        ``criterion_text``, etc.) to join back to the embeddings file.

    data/clustering/hdbscan/<condition>_trials_hdbscan_cluster_metrics.json
        Clustering performance metrics (silhouette score, Davies-Bouldin
        index, Calinski-Harabasz index — computed only over non-noise
        points) plus noise statistics and run provenance, so clustering
        quality can be reviewed without re-running the clustering step.

Re-running this step on the same condition overwrites both existing output
files wholesale — there is no append/skip-if-exists/merge behavior.

Read/validate/write logic shared with the k-means clustering pipeline (see
`criteria_clustering.py`) lives in `clustering_io.py`; this module holds only
HDBSCAN-specific logic (fitting, noise-aware metrics, probability output).
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
from sklearn.cluster import HDBSCAN
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "data" / "clustering" / "hdbscan"

# clustering_io.py lives in a sibling module with no shared package
# namespace (src/clustering/ has no __init__.py, consistent with this
# module being invoked as a standalone script). Add the src/ root to
# sys.path explicitly so this still works when run directly rather than
# via the installed package.
sys.path.insert(0, str(REPO_ROOT / "src"))

from clustering import clustering_io
from preprocessing.criteria_retrieval import EligibilityCriteriaFetcher

logger = logging.getLogger(__name__)

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
    """Orchestrates the read -> filter -> cluster -> score -> write pipeline for one condition.

    Unlike `CriteriaClusterer`, there is no "auto vs. fixed k" branching —
    HDBSCAN always discovers its own cluster count from the data; only its
    density hyperparameters are configurable.

    Typical usage:
        clusterer = HDBSCANClusterer()
        assignments_path, metrics_path = clusterer.run("breast cancer")
    """

    DEFAULT_MIN_CLUSTER_SIZE = 5
    DEFAULT_METRIC = "euclidean"
    DEFAULT_CLUSTER_SELECTION_EPSILON = 0.0
    DEFAULT_CLUSTER_SELECTION_METHOD = "eom"

    def __init__(
        self,
        min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
        min_samples: int | None = None,
        metric: str = DEFAULT_METRIC,
        cluster_selection_epsilon: float = DEFAULT_CLUSTER_SELECTION_EPSILON,
        cluster_selection_method: str = DEFAULT_CLUSTER_SELECTION_METHOD,
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
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.cluster_selection_method = cluster_selection_method

    def validate_hyperparameters(self, n_samples: int) -> None:
        """Confirm the configured hyperparameters are usable for this dataset.

        Replicates sklearn's own fallback (`min_samples` defaults to
        `min_cluster_size` when unset) so this pre-flight check and
        `HDBSCAN.fit`'s internal behavior never disagree about what
        "effective `min_samples`" means.

        Args:
            n_samples: Number of embeddings available to cluster.

        Raises:
            ValueError: If `min_cluster_size` is less than 1, or if the
                effective `min_samples` (explicit value, or `min_cluster_size`
                if unset) exceeds `n_samples` — naming both numbers, pre-empting
                sklearn's own less-contextualized `ValueError` for the same
                condition (confirmed empirically to occur otherwise).
        """
        if self.min_cluster_size < 1:
            raise ValueError(f"min_cluster_size must be at least 1, got {self.min_cluster_size}")
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
        when the guard fails (e.g. an all-noise or single-cluster result),
        the three metrics are reported as `None` with an explanatory note
        rather than letting sklearn raise. `noise_count`/`noise_fraction` are
        always reported, regardless of whether the metrics could be computed.

        Args:
            embedding_matrix: The matrix HDBSCAN was fit on.
            labels: `model.labels_` (may include `-1` for noise).

        Returns:
            A dict with keys `metrics` (the 3 metric values, `None` for any
            skipped), `notes` (list of human-readable caveats),
            `n_clusters_actual` (count of distinct non-negative labels),
            `noise_count`, `noise_fraction`, and `metrics_valid` (True iff
            none of the guarded metrics were skipped).
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

        can_score = 2 <= n_clusters_actual <= n_non_noise - 1
        if can_score:
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

        return {
            "metrics": metrics,
            "notes": notes,
            "n_clusters_actual": n_clusters_actual,
            "noise_count": noise_count,
            "noise_fraction": noise_fraction,
            "metrics_valid": can_score,
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
            condition: Condition string passed to `run`.
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
            },
            "metrics": metrics_result["metrics"],
            "metrics_valid": metrics_result["metrics_valid"],
            "notes": metrics_result["notes"],
        }

    def run(
        self,
        condition: str,
        input_path: Path | None = None,
        assignments_output_path: Path | None = None,
        metrics_output_path: Path | None = None,
    ) -> tuple[Path, Path]:
        """Run the full read-filter-cluster-score-write pipeline for one condition.

        Args:
            condition: Condition string, e.g. "breast cancer" — the same
                string passed to `criteria_embedding.py`. Used to derive
                default input/output paths via `condition_to_slug`.
            input_path: Embeddings Parquet to read. If omitted, defaults to
                the embeddings directory's `{slug}_trials_embeddings.parquet`.
            assignments_output_path: Where to write cluster assignments. If
                omitted, defaults to
                ``OUTPUT_DIR / f"{slug}_trials_hdbscan_clusters.parquet"``.
            metrics_output_path: Where to write metrics. If omitted, defaults
                to ``OUTPUT_DIR / f"{slug}_trials_hdbscan_cluster_metrics.json"``.

        Returns:
            A tuple `(assignments_output_path, metrics_output_path)`.

        Raises:
            FileNotFoundError: If the resolved input Parquet does not exist.
            ValueError: If there are no successfully-embedded rows to
                cluster, if embedding dimensionality is inconsistent, or if
                the configured hyperparameters are invalid for the available
                sample count.
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

        self.validate_hyperparameters(n_samples)
        model = self.fit_hdbscan(embedding_matrix)
        labels = model.labels_
        probabilities = model.probabilities_
        metrics_result = self.compute_metrics(embedding_matrix, labels)

        clustered_at = datetime.now(timezone.utc)
        rows = self.build_assignment_rows(filtered, labels, probabilities, clustered_at)
        clustering_io.write_parquet(rows, HDBSCAN_CLUSTER_PARQUET_SCHEMA, assignments_output_path)

        embedding_backend = filtered.column("embedding_backend")[0].as_py()
        embedding_model = filtered.column("embedding_model")[0].as_py()

        metrics_dict = self.build_metrics_dict(
            condition=condition,
            slug=slug,
            input_path=input_path,
            assignments_output_path=assignments_output_path,
            n_samples=n_samples,
            n_samples_excluded=n_excluded,
            embedding_dim=embedding_dim,
            embedding_backend=embedding_backend,
            embedding_model=embedding_model,
            metrics_result=metrics_result,
            labels=labels,
            clustered_at=clustered_at,
        )
        clustering_io.write_json(metrics_dict, metrics_output_path)

        logger.info(
            "run: clustered %d row(s) into %d cluster(s) for %r -> %s, %s "
            "(embedding_dim=%d, noise=%d/%d (%.1f%%), silhouette=%s, excluded=%d failed-embedding row(s))",
            n_samples,
            metrics_result["n_clusters_actual"],
            condition,
            assignments_output_path,
            metrics_output_path,
            embedding_dim,
            metrics_result["noise_count"],
            n_samples,
            metrics_result["noise_fraction"] * 100,
            metrics_result["metrics"]["silhouette_score"],
            n_excluded,
        )
        return assignments_output_path, metrics_output_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for this module.

    Returns:
        An ``argparse.ArgumentParser`` accepting a positional ``condition``
        argument plus HDBSCAN and path-override options.
    """
    parser = argparse.ArgumentParser(
        description="Cluster a condition's criterion embeddings via HDBSCAN."
    )
    parser.add_argument(
        "condition",
        help='Condition to cluster, e.g. "breast cancer" (same string passed to criteria_embedding.py)',
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=HDBSCANClusterer.DEFAULT_MIN_CLUSTER_SIZE,
        help="Minimum size of a group of points to be considered a cluster (default: 5)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=None,
        help="Number of samples in a neighborhood for a point to be a core point (default: min_cluster_size)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=HDBSCANClusterer.DEFAULT_METRIC,
        help="Distance metric to use (default: euclidean)",
    )
    parser.add_argument(
        "--cluster-selection-epsilon",
        type=float,
        default=HDBSCANClusterer.DEFAULT_CLUSTER_SELECTION_EPSILON,
        help="Distance threshold below which clusters are merged (default: 0.0)",
    )
    parser.add_argument(
        "--cluster-selection-method",
        type=str,
        choices=["eom", "leaf"],
        default=HDBSCANClusterer.DEFAULT_CLUSTER_SELECTION_METHOD,
        help="Cluster selection method: 'eom' (excess of mass) or 'leaf' (default: eom)",
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
    """CLI entry point: cluster and write cluster assignments/metrics for one condition."""
    logging.basicConfig(level=logging.INFO)

    args = build_arg_parser().parse_args()

    clusterer = HDBSCANClusterer(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric=args.metric,
        cluster_selection_epsilon=args.cluster_selection_epsilon,
        cluster_selection_method=args.cluster_selection_method,
    )
    clusterer.run(
        args.condition,
        input_path=args.input_path,
        assignments_output_path=args.assignments_output_path,
        metrics_output_path=args.metrics_output_path,
    )


if __name__ == "__main__":
    main()
