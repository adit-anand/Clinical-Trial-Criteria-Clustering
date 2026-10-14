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

    data/clustering/<condition>_trials_clusters.parquet
        One row per successfully-embedded criterion, carrying its assigned
        ``cluster_id`` plus enough source columns (``nct_id``,
        ``criterion_text``, etc.) to join back to the embeddings file.

    data/clustering/<condition>_trials_cluster_metrics.json
        Clustering performance metrics (inertia, silhouette score,
        Davies-Bouldin index, Calinski-Harabasz index) plus run provenance
        (sample/cluster counts, k-means parameters), so clustering quality
        can be reviewed without re-running the clustering step.

Re-running this step on the same condition overwrites both existing output
files wholesale — there is no append/skip-if-exists/merge behavior.
"""

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = REPO_ROOT / "data" / "embeddings"
OUTPUT_DIR = REPO_ROOT / "data" / "clustering"

# criteria_retrieval.py lives in a sibling directory with no shared package
# namespace (neither src/clustering/ nor src/preprocessing/ has an
# __init__.py, consistent with both modules being invoked as standalone
# scripts). Add the src/ root to sys.path explicitly so this still works
# when the script is run directly rather than via the installed package.
sys.path.insert(0, str(REPO_ROOT / "src"))

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


def _relative_to_repo_root(path: Path) -> str:
    """Render `path` as a POSIX-style string relative to `REPO_ROOT` if possible.

    Falls back to the absolute path string when `path` doesn't live under
    `REPO_ROOT` (e.g. an explicit `--input-path` override pointing elsewhere),
    so metrics JSON generation never fails on path formatting alone.
    """
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


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

    def read_embeddings_parquet(self, input_path: Path) -> pa.Table:
        """Read the embeddings Parquet file produced by `criteria_embedding.py`.

        Args:
            input_path: Path to a Parquet file written by
                `CriteriaEmbedder.write_embeddings_parquet`.

        Returns:
            The full Parquet table, including rows with failed embeddings.

        Raises:
            FileNotFoundError: If `input_path` does not exist.
        """
        if not input_path.exists():
            raise FileNotFoundError(f"Input embeddings Parquet not found: {input_path}")
        return pq.read_table(input_path)

    def filter_successful_rows(self, table: pa.Table) -> pa.Table:
        """Filter a table down to rows where `embedding_success` is True.

        Args:
            table: A table as returned by `read_embeddings_parquet`.

        Returns:
            The subset of `table` with `embedding_success == True`.

        Raises:
            ValueError: If the result has zero rows, covering both an empty
                input table and an input where every row failed embedding —
                there is nothing to cluster either way.
        """
        filtered = table.filter(pc.field("embedding_success"))
        if filtered.num_rows == 0:
            raise ValueError(
                f"No successfully embedded rows found in input "
                f"(0 of {table.num_rows} row(s) had embedding_success=True); "
                "refusing to cluster an empty dataset"
            )
        return filtered

    def validate_embedding_dimensions(self, table: pa.Table) -> int:
        """Confirm every row's `embedding` vector has the same length.

        Args:
            table: A table of successfully-embedded rows (e.g. the output of
                `filter_successful_rows`).

        Returns:
            The common embedding dimensionality.

        Raises:
            ValueError: If more than one distinct embedding length is found,
                naming each distinct dimension and an example `nct_id` for
                it, so a mixed-backend/mixed-model input fails with an
                actionable message rather than crashing inside numpy/sklearn.
        """
        nct_ids = table.column("nct_id").to_pylist()
        embeddings = table.column("embedding").to_pylist()

        dim_examples: dict[int, str] = {}
        for nct_id, embedding in zip(nct_ids, embeddings):
            dim_examples.setdefault(len(embedding), nct_id)

        if len(dim_examples) > 1:
            details = ", ".join(
                f"dim={dim} (e.g. nct_id={example!r})" for dim, example in sorted(dim_examples.items())
            )
            raise ValueError(
                f"Embeddings have inconsistent dimensionality, cannot cluster together: {details}"
            )
        return next(iter(dim_examples))

    def extract_embedding_matrix(self, table: pa.Table) -> np.ndarray:
        """Stack a table's `embedding` column into a 2D float32 matrix.

        Args:
            table: A table of successfully-embedded rows with uniform
                embedding dimensionality.

        Returns:
            An array of shape `(n_samples, embedding_dim)`.

        Raises:
            ValueError: Propagated from `validate_embedding_dimensions` if
                dimensions are not uniform.
        """
        self.validate_embedding_dimensions(table)
        return np.array(table.column("embedding").to_pylist(), dtype=np.float32)

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
        nct_ids = table.column("nct_id").to_pylist()
        dates = table.column("study_first_posted_date").to_pylist()
        texts = table.column("criterion_text").to_pylist()
        types = table.column("criterion_type").to_pylist()
        backends = table.column("embedding_backend").to_pylist()
        models = table.column("embedding_model").to_pylist()

        rows = []
        for i in range(table.num_rows):
            rows.append(
                {
                    "nct_id": nct_ids[i],
                    "study_first_posted_date": dates[i],
                    "criterion_text": texts[i],
                    "criterion_type": types[i],
                    "cluster_id": int(labels[i]),
                    "embedding_backend": backends[i],
                    "embedding_model": models[i],
                    "clustered_at": clustered_at,
                }
            )
        return rows

    def write_assignments_parquet(self, rows: list[dict], output_path: Path) -> None:
        """Write cluster assignment rows to a Parquet file, overwriting any existing file.

        Args:
            rows: Output row dicts as produced by `build_assignment_rows`.
            output_path: Destination Parquet file path. Parent directories
                are created if missing.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        columns = {name: [row[name] for row in rows] for name in CLUSTER_PARQUET_SCHEMA.names}
        table = pa.Table.from_pydict(columns, schema=CLUSTER_PARQUET_SCHEMA)
        pq.write_table(table, output_path)

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
        cluster_sizes: dict[str, int] = {}
        for label in labels:
            key = str(int(label))
            cluster_sizes[key] = cluster_sizes.get(key, 0) + 1

        return {
            "condition": condition,
            "condition_slug": slug,
            "input_path": _relative_to_repo_root(input_path),
            "assignments_output_path": _relative_to_repo_root(assignments_output_path),
            "n_samples": n_samples,
            "n_samples_excluded_failed_embeddings": n_samples_excluded,
            "embedding_dim": embedding_dim,
            "embedding_backend": embedding_backend,
            "embedding_model": embedding_model,
            "n_clusters_requested": n_clusters_requested,
            "n_clusters_actual": metrics_result["n_clusters_actual"],
            "cluster_sizes": dict(sorted(cluster_sizes.items(), key=lambda kv: int(kv[0]))),
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
            "clustered_at": clustered_at.isoformat(),
        }

    def write_metrics_json(self, metrics_dict: dict, output_path: Path) -> None:
        """Write a metrics dict to a JSON file, overwriting any existing file.

        Args:
            metrics_dict: As produced by `build_metrics_dict`.
            output_path: Destination JSON file path. Parent directories are
                created if missing.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, indent=2)

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
                ``INPUT_DIR / f"{slug}_trials_embeddings.parquet"``.
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
        if input_path is None:
            input_path = INPUT_DIR / f"{slug}_trials_embeddings.parquet"
        if assignments_output_path is None:
            assignments_output_path = OUTPUT_DIR / f"{slug}_trials_clusters.parquet"
        if metrics_output_path is None:
            metrics_output_path = OUTPUT_DIR / f"{slug}_trials_cluster_metrics.json"

        table = self.read_embeddings_parquet(input_path)
        total_rows = table.num_rows
        filtered = self.filter_successful_rows(table)
        n_samples = filtered.num_rows
        n_excluded = total_rows - n_samples

        embedding_dim = self.validate_embedding_dimensions(filtered)
        embedding_matrix = self.extract_embedding_matrix(filtered)

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
        self.write_assignments_parquet(rows, assignments_output_path)

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
        self.write_metrics_json(metrics_dict, metrics_output_path)

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
