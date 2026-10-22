"""Cluster eligibility criteria embeddings via HDBSCAN, tuned by an Optuna search.

Reads the same embeddings Parquet file format as `kmeans_clustering.py`,
drops rows whose embedding failed, and runs an Optuna study that fits
`sklearn.cluster.HDBSCAN` once per trial, suggesting `min_cluster_size`/
`min_samples`/`cluster_selection_epsilon`/`cluster_selection_method`/`alpha`
from the search space defined in a `--config-path` YAML file (`metric` is
held fixed, not swept — DBCV scores computed under different distance
metrics aren't comparable within one study). Unlike k-means, there is no `k`
to choose for any individual fit — HDBSCAN discovers clusters of varying
density directly from the data, and explicitly labels points that don't
belong to any cluster as noise (``cluster_id == -1``).

Each trial's composite objective score (`optuna_search.score_candidate`:
DBCV, folded with noise-fraction/dominant-cluster/cluster-count/micro-
cluster penalties) is the single ranking criterion — it both drives
Optuna's own trial selection and determines which trial's result gets
written. The study is resumable (SQLite-backed, via `optuna_search.build_study`):
re-running the same command against the same input continues the same
study rather than starting over, and the configured `n_trials` is treated
as a *total* budget across resumes. A full per-trial history CSV is written
under `data/clustering/reports/` on every run.

If every trial that fit is degenerate (none satisfy `HDBSCANClusterer.
compute_metrics`'s eligibility constraints — cluster count outside
`MIN_N_CLUSTERS`/`MAX_N_CLUSTERS`, a dominant cluster exceeding
`MAX_CLUSTER_SHARE`, or noise exceeding `MAX_NOISE_FRACTION`), the
least-bad trial by the same composite score is still written, with
`metrics_valid=False` recorded. Nothing is written only if every trial in
the study was `"invalid"` (its hyperparameters were rejected before a fit
was ever attempted).

Re-running this step against the same output paths overwrites both existing
output files (and the trial-history CSV) wholesale — only the Optuna
study's own SQLite storage accumulates across runs, a deliberate exception
to that convention (see `optuna_search.py`).

Read/validate/write logic shared with the k-means clustering pipeline (see
`kmeans_clustering.py`) lives in `clustering_io.py`; this module holds only
HDBSCAN-specific logic (fitting, noise-aware metrics, probability output).
`hdbscan_sweep.py` is a thin, backward-compatible alias for this module's
CLI.
"""

import argparse
import logging
import math
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import optuna
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

from src.paths import REPO_ROOT, label_from_slug, slug_from_filename
from src.clustering import clustering_io
from src.clustering import optuna_search

logger = logging.getLogger(__name__)

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

    Unlike `KMeansClusterer`, there is no "auto vs. fixed k" branching —
    HDBSCAN always discovers its own cluster count from the data; only its
    density hyperparameters are configurable. This class fits and scores one
    combination; `run_optuna_search` (module-level, below) constructs one
    instance per Optuna trial it tries.

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
        the same way the k-means pipeline guards its own metrics (requires
        2 <= n_clusters_actual <= n_non_noise - 1 among non-noise points);
        when that guard fails (e.g. an all-noise or single-cluster result),
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


def run_optuna_search(
    input_path: Path,
    assignments_output_path: Path,
    metrics_output_path: Path,
    config_path: Path,
) -> tuple[Path, Path] | None:
    """Run an Optuna-driven HDBSCAN hyperparameter search and write the winning trial's result.

    Args:
        input_path: Embeddings Parquet to read.
        assignments_output_path: Where to write the winning trial's cluster assignments.
        metrics_output_path: Where to write the winning trial's metrics.
        config_path: YAML config (search space, objective weights, trial
            budget, study/report directories) — see `optuna_search.load_optuna_config`.

    Returns:
        `(assignments_output_path, metrics_output_path)` for the winning
        trial, or `None` if every trial in the study was invalid (nothing
        ever fit — nothing is written in that case).

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

    config = optuna_search.load_optuna_config(config_path, require_reproducibility=False)
    bounds = optuna_search.EligibilityBounds(
        min_n_clusters=HDBSCANClusterer.MIN_N_CLUSTERS,
        max_n_clusters=HDBSCANClusterer.MAX_N_CLUSTERS,
        max_cluster_share=HDBSCANClusterer.MAX_CLUSTER_SHARE,
    )

    def fit_trial(hyperparams: dict) -> dict:
        """Fit one HDBSCAN hyperparameter combination into a trial-cache-shaped record.

        Pulled out of the objective closure so the post-`study.optimize` winner
        resolution can call it too: a resumed study's `study.best_trial` may be
        from a *previous* process invocation, never present in this process's
        own `trial_cache` (Optuna's storage persists only hyperparameters/score,
        not our fit artifacts) — refitting is the only way to recover its
        labels/probabilities, and is safe since HDBSCAN is fully deterministic
        given the same hyperparameters and data.
        """
        clusterer = HDBSCANClusterer(
            min_cluster_size=hyperparams["min_cluster_size"],
            min_samples=hyperparams["min_samples"],
            metric=config.fixed["metric"],
            cluster_selection_epsilon=hyperparams["cluster_selection_epsilon"],
            cluster_selection_method=hyperparams["cluster_selection_method"],
            alpha=hyperparams["alpha"],
        )
        try:
            clusterer.validate_hyperparameters(n_samples)
            model = clusterer.fit_hdbscan(embedding_matrix)
        except Exception as exc:  # noqa: BLE001 - an invalid/failed trial shouldn't abort the study
            return {"status": "invalid", "detail": str(exc), "hyperparams": hyperparams}

        labels = model.labels_
        probabilities = model.probabilities_
        metrics_result = clusterer.compute_metrics(embedding_matrix, labels)
        relative_validity = compute_relative_validity_index(embedding_matrix, labels, config.fixed["metric"])
        return {
            "status": "ok" if metrics_result["metrics_valid"] else "degenerate",
            "metrics_result": metrics_result,
            "relative_validity": relative_validity,
            "labels": labels,
            "probabilities": probabilities,
            "clusterer": clusterer,
            "hyperparams": hyperparams,
        }

    study_name = f"{slug}_hdbscan"
    storage_path = config.study_dir / f"{study_name}_study.db"
    study = optuna_search.build_study(storage_path, study_name, config.sampler_seed)

    trial_cache: dict[int, dict] = {}

    def objective(trial: optuna.Trial) -> float:
        hyperparams = optuna_search.suggest_hyperparams(trial, config.search_space)
        record = fit_trial(hyperparams)
        trial_cache[trial.number] = record
        # Persisted to the study's storage (unlike trial_cache, which is this
        # process's own memory) so count_trial_statuses reflects every trial
        # across every resumed invocation, not just this one.
        trial.set_user_attr("status", record["status"])
        if record["status"] == "invalid":
            return optuna_search.INVALID_TRIAL_SCORE
        return optuna_search.score_candidate(
            record["metrics_result"], record["relative_validity"], record["labels"],
            hyperparams["min_cluster_size"], config.objective_weights, bounds,
        )

    n_remaining = optuna_search.remaining_trials(study, config.n_trials)
    if n_remaining > 0:
        logger.info(
            "run_optuna_search: running %d new trial(s) for %r (total budget=%d)",
            n_remaining, label, config.n_trials,
        )
        study.optimize(objective, n_trials=n_remaining)
    else:
        logger.info(
            "run_optuna_search: study %r already has >= %d trial(s); skipping new trials",
            study_name, config.n_trials,
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trial_history_path = config.reports_dir / f"{study_name}_optuna_trials_{timestamp}.csv"
    optuna_search.export_trial_history(study, trial_history_path)

    best_trial = optuna_search.best_trial_or_none(study)
    if best_trial is None:
        logger.warning("run_optuna_search: study has no trials for %r; writing nothing", label)
        return None
    winner = trial_cache.get(best_trial.number)
    if winner is None:
        logger.info(
            "run_optuna_search: best trial #%d for %r predates this process; refitting its hyperparameters",
            best_trial.number, label,
        )
        winner = fit_trial(best_trial.params)
    if winner["status"] == "invalid":
        logger.warning("run_optuna_search: no trial ever fit for %r; writing nothing", label)
        return None

    clustered_at = datetime.now(timezone.utc)
    clusterer: HDBSCANClusterer = winner["clusterer"]
    rows = clusterer.build_assignment_rows(filtered, winner["labels"], winner["probabilities"], clustered_at)
    clustering_io.write_parquet(rows, HDBSCAN_CLUSTER_PARQUET_SCHEMA, assignments_output_path)

    embedding_backend = filtered.column("embedding_backend")[0].as_py()
    embedding_model = filtered.column("embedding_model")[0].as_py()
    status_counts = optuna_search.count_trial_statuses(study)

    metrics_dict = clusterer.build_metrics_dict(
        label=label,
        input_path=input_path,
        assignments_output_path=assignments_output_path,
        n_samples=n_samples,
        n_samples_excluded=n_excluded,
        embedding_dim=embedding_dim,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
        metrics_result=winner["metrics_result"],
        labels=winner["labels"],
        clustered_at=clustered_at,
    )
    metrics_dict["relative_validity_index"] = winner["relative_validity"]
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
        "run_optuna_search: best trial #%d for %r (score=%.4f, status=%s) -> %s, %s "
        "(trials: %d ok, %d degenerate, %d invalid)",
        best_trial.number, label, best_trial.value, winner["status"],
        assignments_output_path, metrics_output_path,
        status_counts["ok"], status_counts["degenerate"], status_counts["invalid"],
    )
    return assignments_output_path, metrics_output_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for this module.

    Returns:
        An ``argparse.ArgumentParser`` accepting required input/output path
        flags plus a `--config-path` override for the Optuna search config.
    """
    parser = argparse.ArgumentParser(
        description="Cluster criterion embeddings via HDBSCAN, tuned by an Optuna search."
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
    parser.add_argument(
        "--config-path",
        type=Path,
        default=REPO_ROOT / "config" / "hdbscan_optuna.yaml",
        help="Optuna search config YAML (search space, objective weights, trial budget); "
        "default: config/hdbscan_optuna.yaml",
    )
    return parser


def main() -> None:
    """CLI entry point: run the Optuna search and write cluster assignments/metrics."""
    logging.basicConfig(level=logging.INFO)
    args = build_arg_parser().parse_args()
    run_optuna_search(
        args.input_path,
        args.assignments_output_path,
        args.metrics_output_path,
        args.config_path,
    )


if __name__ == "__main__":
    main()
