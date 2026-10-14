"""Cluster per-condition eligibility criteria embeddings via k-means.

Reads an embeddings Parquet file produced by
`src/embeddings/criteria_embedding.py`
(``data/embeddings/<condition>_trials_embeddings.parquet``), drops rows whose
embedding failed (``embedding_success`` is False — there is no vector to
cluster for those), and fits a `sklearn.cluster.KMeans` model over the
remaining embedding vectors. By default, the number of clusters (`k`) is
selected automatically: k-means is fit once per candidate `k` in a search
range, and the `k` with the highest silhouette score is kept. Passing
``--n-clusters`` instead skips the search and clusters at exactly that `k`.
Either way, the run writes two output artifacts:

    data/clustering/kmeans/<condition>_trials_clusters.parquet
        One row per successfully-embedded criterion, carrying its assigned
        ``cluster_id`` plus enough source columns (``nct_id``,
        ``criterion_text``, etc.) to join back to the embeddings file.

    data/clustering/kmeans/<condition>_trials_cluster_metrics.json
        Clustering performance metrics (inertia, silhouette score,
        Davies-Bouldin index, Calinski-Harabasz index) plus run provenance
        (sample/cluster counts, k-means parameters), so clustering quality
        can be reviewed without re-running the clustering step.

Re-running this step on the same condition overwrites both existing output
files wholesale — there is no append/skip-if-exists/merge behavior.

Read/validate/write logic shared with the HDBSCAN clustering pipeline (see
`hdbscan_clustering.py`) lives in `clustering_io.py`; this module holds only
k-means-specific logic (fitting, `k` selection, k-means's own metrics guard).
"""

import argparse
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "data" / "clustering" / "kmeans"

# clustering_io.py and criteria_retrieval.py live in sibling directories with
# no shared package namespace (none of src/clustering/, src/preprocessing/
# has an __init__.py, consistent with these modules being invoked as
# standalone scripts). Add the src/ root to sys.path explicitly so this still
# works when the script is run directly rather than via the installed package.
sys.path.insert(0, str(REPO_ROOT / "src"))

from clustering import clustering_io
from preprocessing.criteria_retrieval import EligibilityCriteriaFetcher

logger = logging.getLogger(__name__)

CLUSTER_PARQUET_SCHEMA = pa.schema(
    [
        ("nct_id", pa.string()),
        ("study_first_posted_date", pa.string()),
        ("criterion_text", pa.string()),
        ("criterion_type", pa.string()),
        ("cluster_id", pa.int32()),
        ("embedding_backend", pa.string()),
        ("embedding_model", pa.string()),
        ("clustered_at", pa.timestamp("us", tz="UTC")),
    ]
)


class CriteriaClusterer:
    """Orchestrates the read -> filter -> cluster -> score -> write pipeline for one condition.

    Unlike `CriteriaEmbedder`/`EligibilityCriteriaFetcher`, this class holds
    no long-lived expensive resource (no model/session) — k-means
    configuration is cheap to store, and a fresh `KMeans` estimator is
    constructed and fit per `run()` call since it is stateful per dataset.

    Typical usage:
        clusterer = CriteriaClusterer()  # auto-selects k via silhouette score
        assignments_path, metrics_path = clusterer.run("breast cancer")

        clusterer = CriteriaClusterer(n_clusters=20)  # fixed k, skips the search
        assignments_path, metrics_path = clusterer.run("breast cancer")
    """

    DEFAULT_K_MIN = 2
    DEFAULT_RANDOM_STATE = 42
    DEFAULT_N_INIT = 10
    DEFAULT_MAX_ITER = 300

    def __init__(
        self,
        n_clusters: int | None = None,
        k_min: int = DEFAULT_K_MIN,
        k_max: int | None = None,
        random_state: int = DEFAULT_RANDOM_STATE,
        n_init: int = DEFAULT_N_INIT,
        max_iter: int = DEFAULT_MAX_ITER,
    ):
        """Store k-means configuration for this clusterer.

        Args:
            n_clusters: Fix the number of k-means clusters (`k`) and skip
                automatic selection. If `None` (the default), `k` is
                auto-selected by sweeping `[k_min, k_max]` and keeping the
                `k` with the highest silhouette score.
            k_min: Minimum `k` to search when auto-selecting. Ignored if
                `n_clusters` is set.
            k_max: Maximum `k` to search when auto-selecting. If `None`, a
                default is computed per-dataset by `resolve_k_max`. Ignored
                if `n_clusters` is set.
            random_state: Seed for k-means's centroid initialization, fixed
                by default so repeated runs (and repeated sweeps) are
                reproducible.
            n_init: Number of k-means initializations to run per `k`,
                keeping the best result by inertia.
            max_iter: Maximum number of iterations per initialization.
        """
        self.n_clusters = n_clusters
        self.k_min = k_min
        self.k_max = k_max
        self.random_state = random_state
        self.n_init = n_init
        self.max_iter = max_iter

    def validate_k(self, n_clusters: int, n_samples: int) -> None:
        """Confirm a given `n_clusters` is usable for this dataset.

        Used by the fixed-`k` path (where `n_clusters` comes from the
        caller/CLI).

        Args:
            n_clusters: The `k` to validate.
            n_samples: Number of embeddings available to cluster.

        Raises:
            ValueError: If `n_clusters` is less than 1, or greater than
                `n_samples` — naming both numbers so the fix is obvious.
        """
        if n_clusters < 1:
            raise ValueError(f"n_clusters must be at least 1, got {n_clusters}")
        if n_clusters > n_samples:
            raise ValueError(
                f"Requested n_clusters={n_clusters} exceeds the number of "
                f"available embeddings ({n_samples}); reduce --n-clusters or provide more data"
            )

    def resolve_k_max(self, n_samples: int) -> int:
        """Compute the default search ceiling for auto-selection.

        Args:
            n_samples: Number of embeddings available to cluster.

        Returns:
            `min(100, floor(sqrt(n_samples / 2)))`, clipped to `n_samples - 1`
            (silhouette's hard ceiling on `k`).
        """
        heuristic = int(math.sqrt(n_samples / 2))
        return min(100, heuristic, n_samples - 1)

    def validate_k_range(self, n_samples: int, k_min: int, k_max: int) -> None:
        """Confirm a `[k_min, k_max]` search range is usable for this dataset.

        Args:
            n_samples: Number of embeddings available to cluster.
            k_min: Minimum `k` to search.
            k_max: Maximum `k` to search.

        Raises:
            ValueError: If `k_min` is less than 2 (silhouette needs at least
                2 clusters), if `k_max` exceeds `n_samples - 1` (silhouette's
                hard ceiling), or if `k_max < k_min` (which is also how too
                few samples to search any valid `k` surfaces, e.g.
                `n_samples < 3`) — naming the numbers involved and suggesting
                `--n-clusters` as a fallback for very small datasets.
        """
        if k_min < 2:
            raise ValueError(f"k_min must be at least 2 for silhouette-based auto-selection, got {k_min}")
        if k_max > n_samples - 1:
            raise ValueError(
                f"k_max={k_max} exceeds n_samples - 1 ({n_samples - 1}); "
                "silhouette score cannot be computed for k >= n_samples"
            )
        if k_max < k_min:
            raise ValueError(
                f"No valid k to search: k_max={k_max} is less than k_min={k_min} "
                f"(n_samples={n_samples}); auto-selection needs at least 3 successfully-embedded "
                "rows, pass --n-clusters explicitly for small datasets"
            )

    def fit_kmeans(self, embedding_matrix: np.ndarray, n_clusters: int) -> KMeans:
        """Fit a k-means model over an embedding matrix at a given `k`.

        Args:
            embedding_matrix: Array of shape `(n_samples, embedding_dim)`.
            n_clusters: Number of clusters to fit.

        Returns:
            The fitted `KMeans` estimator (`.labels_` and `.inertia_` are
            available on it).
        """
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
            max_iter=self.max_iter,
        )
        kmeans.fit(embedding_matrix)
        return kmeans

    def select_optimal_k(
        self, embedding_matrix: np.ndarray, k_min: int, k_max: int
    ) -> tuple[int, KMeans, list[dict]]:
        """Sweep `[k_min, k_max]` and pick the `k` with the highest silhouette score.

        Every `k` in the range is fit via `fit_kmeans`, so this shares the
        exact same k-means configuration (`random_state`/`n_init`/`max_iter`)
        as the fixed-`k` path. Since `k_min >= 2` and `k_max <= n_samples - 1`
        are enforced by `validate_k_range` before this is called, all 4
        metrics are always computable for every candidate `k` here (no
        guarding needed, unlike `compute_metrics`'s general case).

        Args:
            embedding_matrix: Array of shape `(n_samples, embedding_dim)`.
            k_min: Minimum `k` to search (inclusive).
            k_max: Maximum `k` to search (inclusive).

        Returns:
            A tuple `(best_k, best_kmeans, sweep_results)`, where
            `sweep_results` has one dict per searched `k`
            (`k`, `inertia`, `silhouette_score`, `davies_bouldin_index`,
            `calinski_harabasz_index`), and `best_kmeans` is the already-fit
            estimator for `best_k` (no redundant refit).
        """
        sweep_results: list[dict] = []
        best_k: int | None = None
        best_kmeans: KMeans | None = None
        best_silhouette = -1.0

        for k in range(k_min, k_max + 1):
            kmeans = self.fit_kmeans(embedding_matrix, k)
            labels = kmeans.labels_
            silhouette = float(silhouette_score(embedding_matrix, labels))
            sweep_results.append(
                {
                    "k": k,
                    "inertia": float(kmeans.inertia_),
                    "silhouette_score": silhouette,
                    "davies_bouldin_index": float(davies_bouldin_score(embedding_matrix, labels)),
                    "calinski_harabasz_index": float(calinski_harabasz_score(embedding_matrix, labels)),
                }
            )
            if silhouette > best_silhouette:
                best_silhouette = silhouette
                best_k = k
                best_kmeans = kmeans

        return best_k, best_kmeans, sweep_results

    def compute_metrics(
        self, embedding_matrix: np.ndarray, labels: np.ndarray, kmeans: KMeans, n_clusters_requested: int
    ) -> dict:
        """Compute clustering performance metrics for a fitted k-means result.

        `inertia` is always computable. Silhouette score, the Davies-Bouldin
        index, and the Calinski-Harabasz index all require at least 2 and at
        most `n_samples - 1` distinct cluster labels — when that guard fails
        (e.g. a degenerate clustering collapsed to 1 effective cluster), the
        three guarded metrics are reported as `None` with an explanatory note
        rather than letting sklearn raise.

        Args:
            embedding_matrix: The matrix `kmeans` was fit on.
            labels: `kmeans.labels_`.
            kmeans: The fitted estimator (for `.inertia_`).
            n_clusters_requested: The `k` `kmeans` was constructed with
                (whether from a fixed `--n-clusters` or the auto-selection
                sweep's chosen `k`), used only to detect/note empty-cluster
                collapse.

        Returns:
            A dict with keys `metrics` (the 4 metric values, `None` for any
            skipped), `notes` (list of human-readable caveats), `n_clusters_actual`
            (distinct label count), and `metrics_valid` (True iff none of the
            guarded metrics were skipped).
        """
        n_samples = embedding_matrix.shape[0]
        n_clusters_actual = len(set(labels.tolist()))
        notes: list[str] = []

        can_score = 2 <= n_clusters_actual <= n_samples - 1
        if can_score:
            metrics = {
                "inertia": float(kmeans.inertia_),
                "silhouette_score": float(silhouette_score(embedding_matrix, labels)),
                "davies_bouldin_index": float(davies_bouldin_score(embedding_matrix, labels)),
                "calinski_harabasz_index": float(calinski_harabasz_score(embedding_matrix, labels)),
            }
        else:
            metrics = {
                "inertia": float(kmeans.inertia_),
                "silhouette_score": None,
                "davies_bouldin_index": None,
                "calinski_harabasz_index": None,
            }
            notes.append(
                "Skipped silhouette/Davies-Bouldin/Calinski-Harabasz scores: require "
                f"2 <= n_clusters <= n_samples - 1, but got n_clusters_actual={n_clusters_actual} "
                f"with n_samples={n_samples}"
            )

        if n_clusters_actual != n_clusters_requested:
            notes.append(
                f"Requested n_clusters={n_clusters_requested} but k-means produced only "
                f"{n_clusters_actual} distinct cluster(s) (likely empty-cluster collapse)"
            )

        return {
            "metrics": metrics,
            "notes": notes,
            "n_clusters_actual": n_clusters_actual,
            "metrics_valid": can_score,
        }

    def build_assignment_rows(self, table: pa.Table, labels: np.ndarray, clustered_at: datetime) -> list[dict]:
        """Zip each surviving row's traceability columns with its cluster label.

        Args:
            table: The table `labels` was produced from (row order must match).
            labels: One cluster id per row of `table`.
            clustered_at: Wall-clock time this clustering run completed.

        Returns:
            One output dict per row, matching `CLUSTER_PARQUET_SCHEMA`'s columns.
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
        n_clusters_requested: int,
        metrics_result: dict,
        labels: np.ndarray,
        clustered_at: datetime,
        k_selection_mode: str,
        k_min: int | None,
        k_max: int | None,
        k_search_results: list[dict] | None,
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
            embedding_backend: Backend recorded on the clustered rows (or
                `None` if there were none, which can't happen given
                `n_samples > 0` is guaranteed before this is called).
            embedding_model: Model recorded on the clustered rows.
            n_clusters_requested: The `k` k-means was fit with (fixed or
                auto-selected).
            metrics_result: The dict returned by `compute_metrics`.
            labels: `kmeans.labels_`, used to compute per-cluster sizes.
            clustered_at: Wall-clock time this clustering run completed.
            k_selection_mode: `"fixed"` or `"auto"`.
            k_min: Search floor used, or `None` in fixed mode.
            k_max: Search ceiling used, or `None` in fixed mode.
            k_search_results: Per-`k` sweep results, or `None` in fixed mode.

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
            "n_clusters_requested": n_clusters_requested,
            "n_clusters_actual": metrics_result["n_clusters_actual"],
            "kmeans_params": {
                "random_state": self.random_state,
                "n_init": self.n_init,
                "max_iter": self.max_iter,
            },
            "metrics": metrics_result["metrics"],
            "metrics_valid": metrics_result["metrics_valid"],
            "notes": metrics_result["notes"],
            "k_selection": {
                "mode": k_selection_mode,
                "k_min": k_min,
                "k_max": k_max,
                "k_search_results": k_search_results,
            },
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
                ``OUTPUT_DIR / f"{slug}_trials_clusters.parquet"``.
            metrics_output_path: Where to write metrics. If omitted, defaults
                to ``OUTPUT_DIR / f"{slug}_trials_cluster_metrics.json"``.

        Returns:
            A tuple `(assignments_output_path, metrics_output_path)`.

        Raises:
            FileNotFoundError: If the resolved input Parquet does not exist.
            ValueError: If there are no successfully-embedded rows to
                cluster, if embedding dimensionality is inconsistent, if a
                fixed `n_clusters` is invalid for the available sample count,
                or if the auto-selection search range `[k_min, k_max]` is
                invalid/empty for the available sample count.
        """
        slug = EligibilityCriteriaFetcher.condition_to_slug(condition)
        default_input_path, default_assignments_path, default_metrics_path = clustering_io.resolve_default_paths(
            condition, OUTPUT_DIR, "_trials_clusters.parquet", "_trials_cluster_metrics.json"
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

        if self.n_clusters is not None:
            k_selection_mode = "fixed"
            k_min = k_max = k_search_results = None
            self.validate_k(self.n_clusters, n_samples)
            kmeans = self.fit_kmeans(embedding_matrix, self.n_clusters)
            n_clusters_requested = self.n_clusters
        else:
            k_selection_mode = "auto"
            k_min = self.k_min
            k_max = self.k_max if self.k_max is not None else self.resolve_k_max(n_samples)
            self.validate_k_range(n_samples, k_min, k_max)
            n_clusters_requested, kmeans, k_search_results = self.select_optimal_k(embedding_matrix, k_min, k_max)

        labels = kmeans.labels_
        metrics_result = self.compute_metrics(embedding_matrix, labels, kmeans, n_clusters_requested)

        clustered_at = datetime.now(timezone.utc)
        rows = self.build_assignment_rows(filtered, labels, clustered_at)
        clustering_io.write_parquet(rows, CLUSTER_PARQUET_SCHEMA, assignments_output_path)

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
            n_clusters_requested=n_clusters_requested,
            metrics_result=metrics_result,
            labels=labels,
            clustered_at=clustered_at,
            k_selection_mode=k_selection_mode,
            k_min=k_min,
            k_max=k_max,
            k_search_results=k_search_results,
        )
        clustering_io.write_json(metrics_dict, metrics_output_path)

        if k_selection_mode == "auto":
            selection_summary = f"auto-selected k={n_clusters_requested} (searched {k_min}..{k_max})"
        else:
            selection_summary = f"fixed k={n_clusters_requested}"

        logger.info(
            "run: clustered %d row(s) into %d cluster(s), %s, for %r -> %s, %s "
            "(embedding_dim=%d, inertia=%.4f, silhouette=%s, excluded=%d failed-embedding row(s))",
            n_samples,
            metrics_result["n_clusters_actual"],
            selection_summary,
            condition,
            assignments_output_path,
            metrics_output_path,
            embedding_dim,
            metrics_result["metrics"]["inertia"],
            metrics_result["metrics"]["silhouette_score"],
            n_excluded,
        )
        return assignments_output_path, metrics_output_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for this module.

    Returns:
        An ``argparse.ArgumentParser`` accepting a positional ``condition``
        argument plus k-means and path-override options.
    """
    parser = argparse.ArgumentParser(
        description="Cluster a condition's criterion embeddings via k-means."
    )
    parser.add_argument(
        "condition",
        help='Condition to cluster, e.g. "breast cancer" (same string passed to criteria_embedding.py)',
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help=(
            "Fix the number of k-means clusters and skip automatic selection "
            "(default: auto-select via a silhouette-score sweep over --k-min..--k-max)"
        ),
    )
    parser.add_argument(
        "--k-min",
        type=int,
        default=CriteriaClusterer.DEFAULT_K_MIN,
        help="Minimum k to search when auto-selecting (ignored if --n-clusters is set; default: 2)",
    )
    parser.add_argument(
        "--k-max",
        type=int,
        default=None,
        help=(
            "Maximum k to search when auto-selecting (ignored if --n-clusters is set); "
            "defaults to min(100, floor(sqrt(n_samples / 2)), n_samples - 1)"
        ),
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=CriteriaClusterer.DEFAULT_RANDOM_STATE,
        help="Random seed for k-means initialization, for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--n-init",
        type=int,
        default=CriteriaClusterer.DEFAULT_N_INIT,
        help="Number of k-means initializations to run (default: 10)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=CriteriaClusterer.DEFAULT_MAX_ITER,
        help="Maximum k-means iterations per initialization (default: 300)",
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

    clusterer = CriteriaClusterer(
        n_clusters=args.n_clusters,
        k_min=args.k_min,
        k_max=args.k_max,
        random_state=args.random_state,
        n_init=args.n_init,
        max_iter=args.max_iter,
    )
    clusterer.run(
        args.condition,
        input_path=args.input_path,
        assignments_output_path=args.assignments_output_path,
        metrics_output_path=args.metrics_output_path,
    )


if __name__ == "__main__":
    main()
