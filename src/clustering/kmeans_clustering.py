"""Cluster eligibility criteria embeddings via k-means, sweeping hyperparameters.

Reads an embeddings Parquet file produced by
`src/embeddings/criteria_embedding.py`, drops rows whose embedding failed
(``embedding_success`` is False — there is no vector to cluster for those),
and fits `sklearn.cluster.KMeans` once per combination in the Cartesian
product of the supplied `--k-min`/`--k-max`/`--random-state` value lists
(`--n-init`/`--max-iter` are held fixed across the whole sweep, not swept).
Passing a single value for each (the default) is simply a sweep of size one
— every invocation goes through the same combination-evaluation and
winner-selection path, whether there's one candidate or fifty.

For each `(k_min, k_max, random_state)` combination, `k` is itself
auto-selected by sweeping `[k_min, k_max]` with that `random_state` and
keeping the `k` with the highest silhouette score (exactly as this module's
"auto" mode always has); the single overall winner across every combination
is then the one whose chosen `k` has the highest silhouette score of all.
Passing ``--n-clusters`` instead fixes `k` and skips all `k`-selection
entirely — only `--random-state` is swept in that case, over the fixed `k`.
Either way, the run writes two output artifacts, at the paths given via
`--assignments-output-path`/`--metrics-output-path`:

    Cluster assignments: one row per successfully-embedded criterion,
        carrying its assigned ``cluster_id`` plus enough source columns
        (``nct_id``, ``criterion_text``, etc.) to join back to the
        embeddings file.

    Cluster metrics: clustering performance metrics (inertia, silhouette
        score, Davies-Bouldin index, Calinski-Harabasz index) plus run
        provenance (sample/cluster counts, k-means parameters, the winning
        combination, and the full sweep audit trail), so clustering quality
        can be reviewed without re-running the clustering step.

A combination whose `[k_min, k_max]` range is invalid for the dataset (e.g.
`k_max < k_min`) is skipped with a logged reason rather than aborting the
rest of the sweep. If no combination produces a usable silhouette score —
including when only the single default combination was tried — nothing is
written. The total number of combinations is capped at
`MAX_SWEEP_COMBINATIONS`; exceeding it raises an error before any fitting is
attempted.

Re-running this step against the same output paths overwrites both existing
output files wholesale — there is no append/skip-if-exists/merge behavior.

Read/validate/write logic shared with the other clustering pipelines (see
`pacmap_hdbscan_clustering.py`/`umap_hdbscan_clustering.py`) lives in
`clustering_io.py`; this module holds only k-means-specific logic (fitting,
`k` selection, k-means's own metrics guard, and the hyperparameter sweep
itself — k-means is the only remaining clustering step still using a
CLI-flag-driven grid sweep rather than an Optuna search).
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
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

# Allow running this file directly (e.g. `uv run src/clustering/kmeans_clustering.py`)
# as well as via `-m` (e.g. `uv run python -m src.clustering.kmeans_clustering`).
# Direct execution leaves __package__ unset and never puts the repo root on
# sys.path, so the src.* imports below would otherwise fail with
# "ModuleNotFoundError: No module named 'src'"; -m already sets both up
# correctly, so this is a no-op in that case.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.paths import label_from_slug, slug_from_filename
from src.clustering import clustering_io

logger = logging.getLogger(__name__)

# The total number of combinations (the product of the swept lists'
# lengths) is capped here; exceeding it raises an error before any fitting
# is attempted. In fixed-`k` mode (no k_min/k_max sweep), this caps the
# `--random-state` list alone.
MAX_SWEEP_COMBINATIONS = 100

KMEANS_CLUSTER_PARQUET_SCHEMA = pa.schema(
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


class KMeansClusterer:
    """Fits and scores a single k-means hyperparameter combination.

    Holds only the k-means configuration that's fixed across a sweep
    (`random_state`/`n_init`/`max_iter`); `k`/`k_min`/`k_max` are passed
    explicitly to each method per combination, not stored on the instance —
    `run_sweep` (module-level, below) constructs one instance per
    `random_state` it tries.

    Typical usage:
        clusterer = KMeansClusterer()
        best_k, model, sweep_results = clusterer.select_optimal_k(embedding_matrix, 2, 50)
    """

    DEFAULT_K_MIN = 2
    DEFAULT_RANDOM_STATE = 42
    DEFAULT_N_INIT = 10
    DEFAULT_MAX_ITER = 300
    DEFAULT_METRIC = "euclidean"

    # sklearn's KMeans has no pluggable distance metric at all — Lloyd's
    # algorithm's centroid-averaging step is only mathematically well-defined
    # under squared Euclidean distance, unlike HDBSCAN/PaCMAP's own `metric`
    # parameters. "cosine" is still offered because it has a well-known,
    # exact equivalence: L2-normalizing every embedding vector to unit length
    # before fitting ordinary Euclidean k-means is "spherical k-means" —
    # squared Euclidean distance between unit vectors is a strictly
    # monotonic function of their cosine distance, so the two give identical
    # cluster assignments. No other metric (e.g. manhattan) has an analogous
    # equivalence for a mean-based centroid update, so none is offered.
    VALID_METRICS = ("euclidean", "cosine")

    def __init__(
        self,
        random_state: int = DEFAULT_RANDOM_STATE,
        n_init: int = DEFAULT_N_INIT,
        max_iter: int = DEFAULT_MAX_ITER,
        metric: str = DEFAULT_METRIC,
    ):
        """Store k-means configuration for this clusterer.

        Args:
            random_state: Seed for k-means's centroid initialization, fixed
                per instance so repeated runs (and repeated sweeps) are
                reproducible.
            n_init: Number of k-means initializations to run per `k`,
                keeping the best result by inertia.
            max_iter: Maximum number of iterations per initialization.
            metric: `"euclidean"` (sklearn's own default) or `"cosine"`. Held
                only for recording in `build_metrics_dict`'s `kmeans_params`
                — the actual cosine behavior is implemented by normalizing
                the embedding matrix *before* it ever reaches this class
                (see `run_sweep`), not by anything in this class itself.
        """
        self.random_state = random_state
        self.n_init = n_init
        self.max_iter = max_iter
        self.metric = metric

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
            One output dict per row, matching `KMEANS_CLUSTER_PARQUET_SCHEMA`'s columns.
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
        label: str,
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
        winning_k_min: int | None,
        winning_k_max: int | None,
        winning_k_search_results: list[dict] | None,
        k_min_values: list[int] | None,
        k_max_values: list[int] | None,
        random_state_values: list[int],
        all_candidates: list["KMeansSweepCandidate"],
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
            n_clusters_requested: The winning combination's `k`.
            metrics_result: The winning combination's `compute_metrics` result.
            labels: The winning combination's `kmeans.labels_`.
            clustered_at: Wall-clock time this clustering run completed.
            k_selection_mode: `"fixed"` or `"auto"`.
            winning_k_min: The winning combination's search floor, or `None` in fixed mode.
            winning_k_max: The winning combination's search ceiling, or `None` in fixed mode.
            winning_k_search_results: The winning combination's own per-`k`
                sweep results, or `None` in fixed mode.
            k_min_values: The full swept `k_min` list, or `None` in fixed mode.
            k_max_values: The full swept `k_max` list (already resolved from
                any `None`/auto entries), or `None` in fixed mode.
            random_state_values: The full swept `random_state` list.
            all_candidates: Every `(k_min, k_max, random_state)` or
                `(random_state)` combination tried, for the audit trail.

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
            "n_clusters_requested": n_clusters_requested,
            "n_clusters_actual": metrics_result["n_clusters_actual"],
            "kmeans_params": {
                "random_state": self.random_state,
                "n_init": self.n_init,
                "max_iter": self.max_iter,
                "metric": self.metric,
            },
            "metrics": metrics_result["metrics"],
            "metrics_valid": metrics_result["metrics_valid"],
            "notes": metrics_result["notes"],
            "k_selection": {
                "mode": k_selection_mode,
                "k_min": winning_k_min,
                "k_max": winning_k_max,
                "k_search_results": winning_k_search_results,
            },
            "combination_search": {
                "k_min_values": k_min_values,
                "k_max_values": k_max_values,
                "random_state_values": random_state_values,
                "candidates": serialize_candidates(all_candidates),
            },
        }


@dataclass(frozen=True)
class KMeansSweepCandidate:
    """The outcome of evaluating one `(k_min, k_max, random_state)` or `(random_state)` combination.

    In fixed-`k` mode, `k_min`/`k_max` are `None` and `best_k` is always the
    fixed `k`; in auto mode, `best_k` is whichever `k` in `[k_min, k_max]`
    scored highest for that combination's `random_state`.

    Attributes:
        k_min: Search floor for this combination, or `None` in fixed mode.
        k_max: Search ceiling for this combination, or `None` in fixed mode.
        random_state: The `random_state` tried for this combination.
        status: `"invalid"` if the `[k_min, k_max]` range (auto mode) or
            fixed `n_clusters` was rejected for this dataset before any fit
            was attempted; `"ok"` otherwise.
        detail: Human-readable explanation, set for `"invalid"`.
        best_k: The `k` used for this combination's fit, set iff `"ok"`.
        kmeans: The fitted `KMeans` estimator for `best_k`, set iff `"ok"`
            (kept so the eventual winner never needs to be refit).
        metrics_result: `KMeansClusterer.compute_metrics(...)`'s return
            value for `best_k`, set iff `"ok"`.
        k_search_results: Per-`k` sweep results within this combination's
            range (auto mode only), else `None`.
    """

    k_min: int | None
    k_max: int | None
    random_state: int
    status: Literal["ok", "invalid"]
    detail: str | None = None
    best_k: int | None = None
    kmeans: KMeans | None = None
    metrics_result: dict | None = None
    k_search_results: list[dict] | None = None


def generate_k_range_combinations(
    k_min_values: list[int], k_max_values: list[int], random_state_values: list[int]
) -> list[dict]:
    """Build the Cartesian product of the three swept auto-mode hyperparameters.

    Args:
        k_min_values: Candidate `k_min` values.
        k_max_values: Candidate `k_max` values (already resolved — no `None`
            entries; see `run_sweep`'s auto-mode branch).
        random_state_values: Candidate `random_state` values.

    Returns:
        One dict per combination, each with keys `k_min`, `k_max`, `random_state`.

    Raises:
        ValueError: If the product size exceeds `MAX_SWEEP_COMBINATIONS`,
            naming the actual count and the limit.
    """
    combos = [
        {"k_min": k_min, "k_max": k_max, "random_state": random_state}
        for k_min, k_max, random_state in itertools.product(k_min_values, k_max_values, random_state_values)
    ]
    if len(combos) > MAX_SWEEP_COMBINATIONS:
        raise ValueError(
            f"Requested sweep has {len(combos)} combination(s), exceeding the limit of "
            f"{MAX_SWEEP_COMBINATIONS}; narrow one or more of --k-min/--k-max/--random-state"
        )
    return combos


def evaluate_k_range_combination(
    combo: dict, n_init: int, max_iter: int, metric: str, embedding_matrix: np.ndarray, n_samples: int
) -> KMeansSweepCandidate:
    """Auto-select `k` within one `(k_min, k_max, random_state)` combination.

    Args:
        combo: One combination from `generate_k_range_combinations`.
        n_init: Held fixed across the sweep.
        max_iter: Held fixed across the sweep.
        metric: Held fixed across the sweep; recorded only (see
            `KMeansClusterer.__init__`) — `embedding_matrix` is already
            normalized by `run_sweep` beforehand if `metric == "cosine"`.
        embedding_matrix: Array of shape `(n_samples, embedding_dim)`,
            already normalized if `metric == "cosine"`.
        n_samples: `embedding_matrix.shape[0]`.

    Returns:
        A `KMeansSweepCandidate` reflecting whether the range was invalid or
        produced a usable result.
    """
    clusterer = KMeansClusterer(random_state=combo["random_state"], n_init=n_init, max_iter=max_iter, metric=metric)
    try:
        clusterer.validate_k_range(n_samples, combo["k_min"], combo["k_max"])
    except ValueError as exc:
        return KMeansSweepCandidate(
            k_min=combo["k_min"], k_max=combo["k_max"], random_state=combo["random_state"],
            status="invalid", detail=str(exc),
        )

    best_k, kmeans, k_search_results = clusterer.select_optimal_k(embedding_matrix, combo["k_min"], combo["k_max"])
    labels = kmeans.labels_
    metrics_result = clusterer.compute_metrics(embedding_matrix, labels, kmeans, best_k)

    return KMeansSweepCandidate(
        k_min=combo["k_min"], k_max=combo["k_max"], random_state=combo["random_state"],
        status="ok", best_k=best_k, kmeans=kmeans, metrics_result=metrics_result, k_search_results=k_search_results,
    )


def evaluate_fixed_k_combination(
    n_clusters: int, random_state: int, n_init: int, max_iter: int, metric: str, embedding_matrix: np.ndarray, n_samples: int
) -> KMeansSweepCandidate:
    """Fit a single fixed `k` for one `random_state` combination.

    Args:
        n_clusters: The fixed `k`, held the same across every combination.
        random_state: This combination's `random_state`.
        n_init: Held fixed across the sweep.
        max_iter: Held fixed across the sweep.
        metric: Held fixed across the sweep; recorded only — `embedding_matrix`
            is already normalized by `run_sweep` beforehand if `metric == "cosine"`.
        embedding_matrix: Array of shape `(n_samples, embedding_dim)`,
            already normalized if `metric == "cosine"`.
        n_samples: `embedding_matrix.shape[0]`.

    Returns:
        A `KMeansSweepCandidate` reflecting whether `n_clusters` was invalid
        or produced a usable result.
    """
    clusterer = KMeansClusterer(random_state=random_state, n_init=n_init, max_iter=max_iter, metric=metric)
    try:
        clusterer.validate_k(n_clusters, n_samples)
    except ValueError as exc:
        return KMeansSweepCandidate(k_min=None, k_max=None, random_state=random_state, status="invalid", detail=str(exc))

    kmeans = clusterer.fit_kmeans(embedding_matrix, n_clusters)
    labels = kmeans.labels_
    metrics_result = clusterer.compute_metrics(embedding_matrix, labels, kmeans, n_clusters)

    return KMeansSweepCandidate(
        k_min=None, k_max=None, random_state=random_state,
        status="ok", best_k=n_clusters, kmeans=kmeans, metrics_result=metrics_result,
    )


def select_best(candidates: list[KMeansSweepCandidate]) -> KMeansSweepCandidate | None:
    """Pick the best candidate by highest silhouette score.

    Args:
        candidates: As produced by `evaluate_k_range_combination`/
            `evaluate_fixed_k_combination`, one per combination.

    Returns:
        The `"ok"` candidate with the highest silhouette score, or `None` if
        no candidate has a computable one (e.g. every combination was
        invalid, or every fit's metrics were degenerate).
    """
    eligible = [
        c for c in candidates
        if c.status == "ok" and c.metrics_result is not None and c.metrics_result["metrics"]["silhouette_score"] is not None
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda c: c.metrics_result["metrics"]["silhouette_score"])


def log_candidate(index: int, total: int, candidate: KMeansSweepCandidate) -> None:
    """Log one INFO line summarizing a single evaluated combination."""
    if candidate.status == "invalid":
        logger.info(
            "evaluate: [%d/%d] k_min=%s k_max=%s random_state=%d -> invalid (%s)",
            index, total, candidate.k_min, candidate.k_max, candidate.random_state, candidate.detail,
        )
        return
    logger.info(
        "evaluate: [%d/%d] k_min=%s k_max=%s random_state=%d -> ok, k=%d, inertia=%.4f, silhouette=%s",
        index, total, candidate.k_min, candidate.k_max, candidate.random_state, candidate.best_k,
        candidate.metrics_result["metrics"]["inertia"], candidate.metrics_result["metrics"]["silhouette_score"],
    )


def serialize_candidates(candidates: list[KMeansSweepCandidate]) -> list[dict]:
    """Render the sweep's audit trail as JSON-serializable dicts.

    Each candidate's record is kept compact (status, chosen `k`, silhouette
    score only — not the full per-`k` inner sweep, which is kept in full
    only for the overall winner under `k_selection.k_search_results`).

    Args:
        candidates: As produced by `run_sweep`'s sweep loop.

    Returns:
        One dict per candidate, suitable for embedding in a metrics JSON's
        `combination_search` field.
    """
    serialized = []
    for candidate in candidates:
        serialized.append(
            {
                "k_min": candidate.k_min,
                "k_max": candidate.k_max,
                "random_state": candidate.random_state,
                "status": candidate.status,
                "detail": candidate.detail,
                "best_k": candidate.best_k,
                "silhouette_score": (
                    candidate.metrics_result["metrics"]["silhouette_score"] if candidate.metrics_result else None
                ),
            }
        )
    return serialized


def run_sweep(
    input_path: Path,
    assignments_output_path: Path,
    metrics_output_path: Path,
    n_clusters: int | None = None,
    k_min_values: list[int] | None = None,
    k_max_values: list[int | None] | None = None,
    random_state_values: list[int] | None = None,
    n_init: int = KMeansClusterer.DEFAULT_N_INIT,
    max_iter: int = KMeansClusterer.DEFAULT_MAX_ITER,
    metric: str = KMeansClusterer.DEFAULT_METRIC,
) -> tuple[Path, Path] | None:
    """Sweep k-means hyperparameters and write the best result.

    This is the module's only orchestration entry point — a single value per
    list (the CLI defaults) is simply a sweep of size one; the winner-
    selection and output-writing logic is identical regardless of how many
    combinations were tried.

    Args:
        input_path: Embeddings Parquet to read.
        assignments_output_path: Where to write the winning combination's
            cluster assignments.
        metrics_output_path: Where to write the winning combination's metrics.
        n_clusters: If given, fixes `k` and sweeps only `random_state_values`
            over that fixed `k` — `k_min_values`/`k_max_values` are ignored.
            If `None` (the default), `k` is auto-selected per combination by
            sweeping `[k_min, k_max]` with that combination's `random_state`.
        k_min_values: Candidate `k_min` values to sweep (auto mode only).
            Defaults to `[KMeansClusterer.DEFAULT_K_MIN]` if `None`.
        k_max_values: Candidate `k_max` values to sweep (auto mode only); a
            `None` entry resolves per-dataset via `resolve_k_max`. Defaults
            to `[None]` if `None`.
        random_state_values: Candidate `random_state` values to sweep (both
            modes). Defaults to `[KMeansClusterer.DEFAULT_RANDOM_STATE]` if `None`.
        n_init: Held fixed across the sweep.
        max_iter: Held fixed across the sweep.
        metric: `"euclidean"` (sklearn's own default) or `"cosine"`, held
            fixed across the sweep. sklearn's `KMeans` has no pluggable
            metric at all — Lloyd's algorithm's centroid-averaging step is
            only mathematically defined under squared Euclidean distance.
            `"cosine"` is implemented by L2-normalizing every embedding
            vector to unit length *before* any fitting ("spherical
            k-means") — squared Euclidean distance between unit vectors is
            a strictly monotonic function of their cosine distance, so this
            gives identical cluster assignments to a true cosine-distance
            k-means, not an approximation.

    Returns:
        `(assignments_output_path, metrics_output_path)` for the winning
        combination, or `None` if no combination produced a usable result
        (in which case nothing is written).

    Raises:
        FileNotFoundError: If the input Parquet does not exist.
        ValueError: If there are no successfully-embedded rows to cluster,
            if embedding dimensionality is inconsistent, if `metric` isn't
            one of `KMeansClusterer.VALID_METRICS`, or if the requested
            sweep has more than `MAX_SWEEP_COMBINATIONS` combinations.
    """
    if k_min_values is None:
        k_min_values = [KMeansClusterer.DEFAULT_K_MIN]
    if k_max_values is None:
        k_max_values = [None]
    if random_state_values is None:
        random_state_values = [KMeansClusterer.DEFAULT_RANDOM_STATE]
    if metric not in KMeansClusterer.VALID_METRICS:
        raise ValueError(f"Unknown metric {metric!r}; expected one of {KMeansClusterer.VALID_METRICS}")

    label = label_from_slug(slug_from_filename(input_path.name, "_trials_embeddings.parquet"))

    table = clustering_io.read_embeddings_parquet(input_path)
    total_rows = table.num_rows
    filtered = clustering_io.filter_successful_rows(table)
    n_samples = filtered.num_rows
    n_excluded = total_rows - n_samples

    embedding_dim = clustering_io.validate_embedding_dimensions(filtered)
    embedding_matrix = clustering_io.extract_embedding_matrix(filtered)

    if metric == "cosine":
        norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError(
                "Cannot normalize for metric='cosine': at least one embedding vector has zero norm"
            )
        embedding_matrix = embedding_matrix / norms
        logger.info("run_sweep: normalized embedding vectors to unit length for metric='cosine'")

    resolved_k_min_values: list[int] | None = None
    resolved_k_max_values: list[int] | None = None

    if n_clusters is not None:
        k_selection_mode = "fixed"
        if len(random_state_values) > MAX_SWEEP_COMBINATIONS:
            raise ValueError(
                f"Requested sweep has {len(random_state_values)} random_state value(s), exceeding "
                f"the limit of {MAX_SWEEP_COMBINATIONS}; narrow --random-state"
            )
        logger.info(
            "run_sweep: evaluating %d random_state value(s) for fixed n_clusters=%d, %r (n_samples=%d)",
            len(random_state_values), n_clusters, label, n_samples,
        )
        candidates = [
            evaluate_fixed_k_combination(n_clusters, random_state, n_init, max_iter, metric, embedding_matrix, n_samples)
            for random_state in random_state_values
        ]
    else:
        k_selection_mode = "auto"
        resolved_k_min_values = sorted(set(k_min_values))
        default_clusterer = KMeansClusterer()
        resolved_k_max_values = sorted(
            {k_max if k_max is not None else default_clusterer.resolve_k_max(n_samples) for k_max in k_max_values}
        )
        combos = generate_k_range_combinations(resolved_k_min_values, resolved_k_max_values, random_state_values)
        logger.info(
            "run_sweep: evaluating %d (k_min, k_max, random_state) combination(s) for %r (n_samples=%d)",
            len(combos), label, n_samples,
        )
        candidates = [
            evaluate_k_range_combination(combo, n_init, max_iter, metric, embedding_matrix, n_samples) for combo in combos
        ]

    for index, candidate in enumerate(candidates, start=1):
        log_candidate(index, len(candidates), candidate)

    best = select_best(candidates)
    if best is None:
        logger.warning(
            "run_sweep: no combination produced a usable silhouette score for %r; writing nothing",
            label,
        )
        return None

    clustered_at = datetime.now(timezone.utc)
    winning_clusterer = KMeansClusterer(random_state=best.random_state, n_init=n_init, max_iter=max_iter, metric=metric)
    labels = best.kmeans.labels_

    rows = winning_clusterer.build_assignment_rows(filtered, labels, clustered_at)
    clustering_io.write_parquet(rows, KMEANS_CLUSTER_PARQUET_SCHEMA, assignments_output_path)

    embedding_backend = filtered.column("embedding_backend")[0].as_py()
    embedding_model = filtered.column("embedding_model")[0].as_py()

    metrics_dict = winning_clusterer.build_metrics_dict(
        label=label,
        input_path=input_path,
        assignments_output_path=assignments_output_path,
        n_samples=n_samples,
        n_samples_excluded=n_excluded,
        embedding_dim=embedding_dim,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
        n_clusters_requested=best.best_k,
        metrics_result=best.metrics_result,
        labels=labels,
        clustered_at=clustered_at,
        k_selection_mode=k_selection_mode,
        winning_k_min=best.k_min,
        winning_k_max=best.k_max,
        winning_k_search_results=best.k_search_results,
        k_min_values=resolved_k_min_values,
        k_max_values=resolved_k_max_values,
        random_state_values=random_state_values,
        all_candidates=candidates,
    )
    clustering_io.write_json(metrics_dict, metrics_output_path)

    if k_selection_mode == "auto":
        selection_summary = f"auto-selected k={best.best_k} (k_min={best.k_min}, k_max={best.k_max})"
    else:
        selection_summary = f"fixed k={best.best_k}"

    logger.info(
        "run_sweep: clustered %d row(s) into %d cluster(s), %s, random_state=%d, for %r -> %s, %s "
        "(embedding_dim=%d, inertia=%.4f, silhouette=%s, excluded=%d failed-embedding row(s))",
        n_samples,
        best.metrics_result["n_clusters_actual"],
        selection_summary,
        best.random_state,
        label,
        assignments_output_path,
        metrics_output_path,
        embedding_dim,
        best.metrics_result["metrics"]["inertia"],
        best.metrics_result["metrics"]["silhouette_score"],
        n_excluded,
    )
    return assignments_output_path, metrics_output_path


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


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for this module.

    Returns:
        An ``argparse.ArgumentParser`` accepting comma-separated value lists
        for the swept hyperparameters, fixed-value flags for the un-swept
        ones, and required input/output path flags.
    """
    parser = argparse.ArgumentParser(
        description="Cluster criterion embeddings via k-means, sweeping hyperparameters."
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help=(
            "Fix the number of k-means clusters and skip automatic selection (--k-min/--k-max are "
            "ignored if set; default: auto-select via a silhouette-score sweep over --k-min..--k-max)"
        ),
    )
    parser.add_argument(
        "--k-min",
        type=parse_int_list,
        default=[KMeansClusterer.DEFAULT_K_MIN],
        help="Comma-separated list of k_min values to sweep when auto-selecting (default: 2)",
    )
    parser.add_argument(
        "--k-max",
        type=parse_int_list,
        default=None,
        help=(
            "Comma-separated list of k_max values to sweep when auto-selecting; "
            "defaults to a single auto-computed value, min(100, floor(sqrt(n_samples / 2)), n_samples - 1)"
        ),
    )
    parser.add_argument(
        "--random-state",
        type=parse_int_list,
        default=[KMeansClusterer.DEFAULT_RANDOM_STATE],
        help="Comma-separated list of random seeds to sweep for k-means initialization (default: 42)",
    )
    parser.add_argument(
        "--n-init",
        type=int,
        default=KMeansClusterer.DEFAULT_N_INIT,
        help="Number of k-means initializations to run, held fixed across the sweep (default: 10)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=KMeansClusterer.DEFAULT_MAX_ITER,
        help="Maximum k-means iterations per initialization, held fixed across the sweep (default: 300)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=KMeansClusterer.VALID_METRICS,
        default=KMeansClusterer.DEFAULT_METRIC,
        help=(
            "Distance metric, held fixed across the sweep (default: euclidean). 'cosine' "
            "L2-normalizes embedding vectors to unit length before fitting ordinary k-means "
            "('spherical k-means') -- sklearn's KMeans has no pluggable metric of its own, "
            "so no other metric is offered"
        ),
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Input embeddings Parquet path (as written by criteria_embedding.py)",
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
    """CLI entry point: sweep, cluster, and write cluster assignments/metrics."""
    logging.basicConfig(level=logging.INFO)

    args = build_arg_parser().parse_args()

    run_sweep(
        args.input_path,
        args.assignments_output_path,
        args.metrics_output_path,
        n_clusters=args.n_clusters,
        k_min_values=args.k_min,
        k_max_values=args.k_max,
        random_state_values=args.random_state,
        n_init=args.n_init,
        max_iter=args.max_iter,
        metric=args.metric,
    )


if __name__ == "__main__":
    main()
