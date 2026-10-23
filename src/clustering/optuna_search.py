"""Shared Optuna-driven hyperparameter search engine for HDBSCAN-based clustering steps.

Used by `hdbscan_clustering.py` (plain HDBSCAN) and `pacmap_hdbscan_clustering.py`/
`umap_hdbscan_clustering.py` (reducer + HDBSCAN) in place of the grid search
those modules used to run. Each module loads its own YAML config via
`load_optuna_config`, builds/resumes a study via `build_study`, and runs an
Optuna objective closure (defined in that module) that scores each trial via
`score_candidate` (plain HDBSCAN) or `score_reduction_candidate` (reducer
pipelines). Every trial — whether it fit cleanly, fit but was degenerate, or
never fit at all — returns a value on the same comparable scale, so
`study.best_trial` is itself the winner; `resolve_winner` just distinguishes
"the best trial never fit" (nothing to write) from every other case.

Deliberate exception to this repo's overwrite-on-rerun convention: the
Optuna study's file-based storage (`build_study`) accumulates across runs by
design — re-running the same command resumes the same study rather than
starting over, joining the BERT feature's embedding cache as the second such
exception in this repo. Every other artifact (assignments Parquet, metrics
JSON, trial-history CSV) is still wholesale-overwritten on every run.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import optuna
import yaml

from src.clustering import clustering_io
from src.paths import REPO_ROOT

logger = logging.getLogger(__name__)

DEFAULT_N_TRIALS = 50
MAX_N_TRIALS = 100

# Caps on the reducer pipelines' `reproducibility` block. A trial now means
# one hyperparameter combination; this bounds how much work one trial (and
# the study as a whole) can demand.
MAX_SEEDS = 10
MAX_REPEATS_PER_SEED = 10
MAX_TOTAL_FITS_PER_STUDY = 25000

# A trial that never fits at all (hyperparameters rejected pre-fit, or the
# fit itself raised) has no metrics to score; this sentinel guarantees it
# can never outrank a trial that actually produced a clustering (realistic
# scores are roughly in [-5, 1]).
INVALID_TRIAL_SCORE = -1e6

# Neighborhood size used for the dimensionality-reduction fidelity metrics
# (subfeature 6) — mirrors zadu's own ZADU.DEFAULT_K.
_DR_METRIC_K = 20
_DR_NORMALIZE_EPSILON = 1e-12


@dataclass(frozen=True)
class HyperparamSpec:
    """One hyperparameter's Optuna search-space definition, as parsed from YAML."""

    kind: Literal["int", "float", "categorical"]
    low: float | int | None = None
    high: float | int | None = None
    choices: list | None = None
    log: bool = False


@dataclass(frozen=True)
class ObjectiveWeights:
    """Weights for every term folded into a trial's composite objective score.

    `trustworthiness_weight`/`continuity_weight`/`steadiness_weight`/
    `cohesiveness_weight`/`missing_dr_metrics_fallback_score` are only read
    by `score_reduction_candidate` — `score_candidate` (plain HDBSCAN) never
    reads them.
    """

    noise_penalty: float = 1.0
    dominant_cluster_penalty: float = 1.0
    cluster_count_violation_penalty: float = 2.0
    micro_cluster_penalty: float = 1.0
    micro_cluster_size_multiplier: float = 1.5
    missing_validity_fallback_score: float = 0.0
    trustworthiness_weight: float = 1.0
    continuity_weight: float = 1.0
    steadiness_weight: float = 1.0
    cohesiveness_weight: float = 1.0
    missing_dr_metrics_fallback_score: float = 0.0


@dataclass(frozen=True)
class ReproducibilityConfig:
    """How many seeds/repeats-per-seed each reducer-pipeline trial runs internally."""

    seeds: list[int]
    repeats_per_seed: int


@dataclass(frozen=True)
class OptunaSearchConfig:
    """A fully-parsed, validated `--config-path` YAML file."""

    search_space: dict[str, HyperparamSpec]
    fixed: dict[str, Any]
    objective_weights: ObjectiveWeights
    n_trials: int
    sampler_seed: int
    study_dir: Path
    results_dir: Path
    reproducibility: ReproducibilityConfig | None = None


@dataclass(frozen=True)
class EligibilityBounds:
    """The degenerate-result eligibility bounds a composite score penalizes against.

    Callers pass `HDBSCANClusterer.MIN_N_CLUSTERS`/`MAX_N_CLUSTERS`/
    `MAX_CLUSTER_SHARE` (from `hdbscan_model.py`) here rather than this
    module importing `HDBSCANClusterer` itself, keeping this module generic
    and reusable by any future clustering algorithm's objective, not just
    HDBSCAN-based ones.
    """

    min_n_clusters: int
    max_n_clusters: int
    max_cluster_share: float


@dataclass(frozen=True)
class DRMetrics:
    """Dimensionality-reduction fidelity scores for one reducer-pipeline trial.

    Deliberately excludes `stress` — this feature folds trustworthiness,
    continuity, steadiness, and cohesiveness into the objective, but not
    stress, per explicit instruction.
    """

    trustworthiness: float
    continuity: float
    steadiness: float
    cohesiveness: float


def _parse_hyperparam_spec(name: str, raw: dict) -> HyperparamSpec:
    """Parse one `search_space` entry from YAML into a `HyperparamSpec`."""
    try:
        return HyperparamSpec(
            kind=raw["kind"],
            low=raw.get("low"),
            high=raw.get("high"),
            choices=raw.get("choices"),
            log=raw.get("log", False),
        )
    except KeyError as exc:
        raise ValueError(f"search_space entry {name!r} is missing required key {exc}") from exc


def load_optuna_config(path: Path, require_reproducibility: bool) -> OptunaSearchConfig:
    """Load and validate a `--config-path` YAML file.

    Args:
        path: Path to the YAML config file.
        require_reproducibility: `True` for the two reducer pipelines (which
            must specify a `reproducibility` block); `False` for plain
            HDBSCAN (which must not).

    Returns:
        A fully-parsed, validated `OptunaSearchConfig`.

    Raises:
        FileNotFoundError: If `path` does not exist.
        ValueError: If `n_trials` is outside `[1, MAX_N_TRIALS]`; if
            `require_reproducibility` is `True` and the config has no
            `reproducibility` block (or vice versa); if `reproducibility.seeds`
            exceeds `MAX_SEEDS`, `repeats_per_seed` exceeds
            `MAX_REPEATS_PER_SEED`, or their product with `n_trials` exceeds
            `MAX_TOTAL_FITS_PER_STUDY`.
    """
    if not path.exists():
        raise FileNotFoundError(f"Optuna config not found: {path}")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    search_space = {
        name: _parse_hyperparam_spec(name, spec) for name, spec in raw["search_space"].items()
    }
    fixed = raw.get("fixed", {})
    objective_weights = ObjectiveWeights(**raw.get("objective_weights", {}))
    n_trials = raw.get("n_trials", DEFAULT_N_TRIALS)
    sampler_seed = raw.get("sampler_seed", 42)
    study_dir = REPO_ROOT / raw["study_dir"]
    results_dir = REPO_ROOT / raw["results_dir"]

    if not (1 <= n_trials <= MAX_N_TRIALS):
        raise ValueError(f"n_trials={n_trials} must be between 1 and {MAX_N_TRIALS}")

    reproducibility = None
    has_reproducibility_block = "reproducibility" in raw
    if require_reproducibility and not has_reproducibility_block:
        raise ValueError(f"Config at {path} is missing a required 'reproducibility' block")
    if not require_reproducibility and has_reproducibility_block:
        raise ValueError(f"Config at {path} has a 'reproducibility' block, but this module doesn't use one")

    if has_reproducibility_block:
        repro_raw = raw["reproducibility"]
        seeds = repro_raw["seeds"]
        repeats_per_seed = repro_raw["repeats_per_seed"]
        if len(seeds) > MAX_SEEDS:
            raise ValueError(f"Requested {len(seeds)} seed(s), exceeding the limit of {MAX_SEEDS}")
        if repeats_per_seed > MAX_REPEATS_PER_SEED:
            raise ValueError(
                f"Requested repeats_per_seed={repeats_per_seed}, exceeding the limit of {MAX_REPEATS_PER_SEED}"
            )
        total_fits = n_trials * len(seeds) * repeats_per_seed
        if total_fits > MAX_TOTAL_FITS_PER_STUDY:
            raise ValueError(
                f"Planned study requires {total_fits} total fit(s) ({n_trials} trial(s) x {len(seeds)} "
                f"seed(s) x {repeats_per_seed} repeat(s) per seed), exceeding the limit of "
                f"{MAX_TOTAL_FITS_PER_STUDY}; narrow n_trials, reproducibility.seeds, or "
                "reproducibility.repeats_per_seed"
            )
        reproducibility = ReproducibilityConfig(seeds=seeds, repeats_per_seed=repeats_per_seed)

    return OptunaSearchConfig(
        search_space=search_space,
        fixed=fixed,
        objective_weights=objective_weights,
        n_trials=n_trials,
        sampler_seed=sampler_seed,
        study_dir=study_dir,
        results_dir=results_dir,
        reproducibility=reproducibility,
    )


def suggest_hyperparams(trial: "optuna.Trial", search_space: dict[str, HyperparamSpec]) -> dict:
    """Suggest a value for every named hyperparameter in `search_space`.

    Args:
        trial: The current Optuna trial.
        search_space: Maps each hyperparameter name to its `HyperparamSpec`.

    Returns:
        One suggested value per key in `search_space`.

    Raises:
        ValueError: If a `HyperparamSpec.kind` isn't `"int"`/`"float"`/`"categorical"`.
    """
    hyperparams = {}
    for name, spec in search_space.items():
        if spec.kind == "int":
            hyperparams[name] = trial.suggest_int(name, int(spec.low), int(spec.high), log=spec.log)
        elif spec.kind == "float":
            hyperparams[name] = trial.suggest_float(name, float(spec.low), float(spec.high), log=spec.log)
        elif spec.kind == "categorical":
            hyperparams[name] = trial.suggest_categorical(name, spec.choices)
        else:
            raise ValueError(f"Unknown hyperparameter kind {spec.kind!r} for {name!r}")
    return hyperparams


def compute_micro_cluster_fraction(labels: np.ndarray, min_cluster_size: int, multiplier: float) -> float:
    """Fraction of non-noise points sitting in "micro" clusters.

    A cluster is "micro" if its size is below `multiplier * min_cluster_size`
    — barely above the bare minimum HDBSCAN was configured to accept, likely
    a fragile/borderline grouping rather than robust structure.

    Args:
        labels: Cluster labels (`-1` for noise).
        min_cluster_size: This trial's `min_cluster_size` hyperparameter.
        multiplier: `ObjectiveWeights.micro_cluster_size_multiplier`.

    Returns:
        The fraction of non-noise points in micro clusters, or `0.0` if
        there are no non-noise points.
    """
    cluster_sizes = clustering_io.compute_cluster_sizes(labels)
    non_noise_sizes = [count for key, count in cluster_sizes.items() if key != "-1"]
    total_non_noise = sum(non_noise_sizes)
    if total_non_noise == 0:
        return 0.0
    threshold = multiplier * min_cluster_size
    micro_count = sum(count for count in non_noise_sizes if count < threshold)
    return micro_count / total_non_noise


def _max_cluster_share(labels: np.ndarray) -> float:
    """Largest non-noise cluster's share of all points (`0.0` if none)."""
    cluster_sizes = clustering_io.compute_cluster_sizes(labels)
    non_noise_sizes = [count for key, count in cluster_sizes.items() if key != "-1"]
    if not non_noise_sizes:
        return 0.0
    return max(non_noise_sizes) / len(labels)


def _eligibility_penalties(
    metrics_result: dict,
    labels: np.ndarray,
    min_cluster_size: int,
    weights: ObjectiveWeights,
    bounds: EligibilityBounds,
) -> float:
    """Shared continuous penalty terms behind both `score_candidate` and `score_reduction_candidate`.

    Mirrors `HDBSCANClusterer.compute_metrics`'s binary eligibility gate
    (cluster count, dominant-cluster share, noise fraction) as continuous
    penalties, plus the micro-cluster fragmentation penalty. This naturally
    ranks degenerate trials below eligible ones without a separate fixed
    penalty — no changes are needed to `compute_metrics`'s own gate, which
    keeps deciding `metrics_valid` exactly as before.
    """
    n_clusters_actual = metrics_result["n_clusters_actual"]
    noise_fraction = metrics_result["noise_fraction"]
    cluster_count_violation = (
        max(0, bounds.min_n_clusters - n_clusters_actual, n_clusters_actual - bounds.max_n_clusters)
        / bounds.max_n_clusters
    )
    return (
        weights.noise_penalty * noise_fraction
        + weights.dominant_cluster_penalty * max(0.0, _max_cluster_share(labels) - bounds.max_cluster_share)
        + weights.cluster_count_violation_penalty * cluster_count_violation
        + weights.micro_cluster_penalty
        * compute_micro_cluster_fraction(labels, min_cluster_size, weights.micro_cluster_size_multiplier)
    )


def score_candidate(
    metrics_result: dict,
    relative_validity: float | None,
    labels: np.ndarray,
    min_cluster_size: int,
    weights: ObjectiveWeights,
    bounds: EligibilityBounds,
) -> float:
    """Composite objective score for a plain-HDBSCAN trial (maximize).

    Args:
        metrics_result: `HDBSCANClusterer.compute_metrics`'s return value.
        relative_validity: The DBCV score, or `None` if not computable.
        labels: This trial's fitted cluster labels.
        min_cluster_size: This trial's `min_cluster_size` hyperparameter.
        weights: The configured objective weights.
        bounds: The eligibility bounds to penalize against.

    Returns:
        A single float; higher is better.
    """
    base = relative_validity if relative_validity is not None else weights.missing_validity_fallback_score
    return base - _eligibility_penalties(metrics_result, labels, min_cluster_size, weights, bounds)


def score_reduction_candidate(
    metrics_result: dict,
    relative_validity: float | None,
    labels: np.ndarray,
    min_cluster_size: int,
    weights: ObjectiveWeights,
    dr_metrics: DRMetrics | None,
    bounds: EligibilityBounds,
) -> float:
    """Composite objective score for a reducer-pipeline trial (maximize).

    Extends `score_candidate` with the dimensionality-reduction fidelity
    terms: trustworthiness/continuity/steadiness/cohesiveness are each
    conventionally normalized to `[0, 1]` with higher being better, so each
    is added as a reward.

    Args:
        metrics_result: `HDBSCANClusterer.compute_metrics`'s return value for
            this trial's representative repeat.
        relative_validity: The DBCV score for the representative repeat, or
            `None` if not computable.
        labels: The representative repeat's fitted cluster labels.
        min_cluster_size: This trial's `min_cluster_size` hyperparameter.
        weights: The configured objective weights.
        dr_metrics: This trial's `DRMetrics`, or `None` if they couldn't be computed.
        bounds: The eligibility bounds to penalize against.

    Returns:
        A single float; higher is better.
    """
    base = relative_validity if relative_validity is not None else weights.missing_validity_fallback_score
    if dr_metrics is not None:
        base += (
            weights.trustworthiness_weight * dr_metrics.trustworthiness
            + weights.continuity_weight * dr_metrics.continuity
            + weights.steadiness_weight * dr_metrics.steadiness
            + weights.cohesiveness_weight * dr_metrics.cohesiveness
        )
    else:
        base += weights.missing_dr_metrics_fallback_score
    return base - _eligibility_penalties(metrics_result, labels, min_cluster_size, weights, bounds)


def l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize every row of `matrix`, guarding a near-zero-norm row.

    Shared by `compute_dr_metrics` (below) and
    `pacmap_hdbscan_clustering.make_fit_pacmap_reduced`, which normalizes
    its input and fits PaCMAP with `distance="euclidean"` instead of
    PaCMAP's own `"angular"` cosine-equivalent — see that module for why.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, _DR_NORMALIZE_EPSILON, None)


def compute_dr_metrics(
    original_matrix: np.ndarray, reduced_matrix: np.ndarray, labels: np.ndarray, metric: str
) -> DRMetrics | None:
    """Compute trustworthiness/continuity/steadiness/cohesiveness via `zadu`.

    `zadu`'s metrics have no distance-metric parameter of their own — every
    metric's internal pairwise-distance computation is hardcoded to
    Euclidean (confirmed against the installed `zadu==0.5.4`). Cosine
    distance is achieved instead by L2-normalizing every row of both inputs
    before handing them to `zadu` when `metric == "cosine"`: Euclidean
    distance between unit vectors is a monotonic function of cosine
    distance (`||a-b||^2 = 2*(1 - cos_sim(a,b))`), so every rank/neighbor/
    distance comparison these metrics make becomes equivalent to a
    cosine-distance computation. Any other `metric` value is passed through
    unnormalized (zadu's native Euclidean behavior).

    `zadu`'s steadiness/cohesiveness (`"snc"`) always reclusters internally
    via its own `clustering_strategy` (`"dbscan"` or `"kmeans"` only — there
    is no way to supply this trial's own fitted HDBSCAN `labels` to it), so
    `"dbscan"` is used as the closer conceptual match. `labels` is accepted
    here for interface symmetry but is not passed into `zadu`.

    Args:
        original_matrix: The full, non-reduced embedding matrix.
        reduced_matrix: This trial's representative repeat's reducer output.
        labels: This trial's representative repeat's fitted cluster labels
            (unused today — see above).
        metric: `"cosine"` to L2-normalize before measuring; anything else
            is passed through as-is.

    Returns:
        A `DRMetrics`, or `None` if the underlying computation fails for any
        reason (mirroring `compute_relative_validity_index`'s fail-soft contract).
    """
    from zadu import ZADU, make_spec

    try:
        original = original_matrix.astype(np.float64)
        reduced = reduced_matrix.astype(np.float64)
        if metric == "cosine":
            original = l2_normalize_rows(original)
            reduced = l2_normalize_rows(reduced)

        k = min(_DR_METRIC_K, original.shape[0] - 1)
        specs = [make_spec("tnc", k=k), make_spec("snc", k=k, clustering_strategy="dbscan")]
        zadu_obj = ZADU(specs, original)
        tnc_result, snc_result = zadu_obj.measure(reduced)
        return DRMetrics(
            trustworthiness=float(tnc_result["trustworthiness"]),
            continuity=float(tnc_result["continuity"]),
            steadiness=float(snc_result["steadiness"]),
            cohesiveness=float(snc_result["cohesiveness"]),
        )
    except Exception:
        logger.exception("compute_dr_metrics: failed to compute dimensionality-reduction fidelity metrics")
        return None


def build_study(storage_path: Path, study_name: str, sampler_seed: int) -> "optuna.Study":
    """Create or resume a file-based Optuna study.

    Uses `optuna.storages.JournalStorage` over a `JournalFileBackend`
    (a plain append-only file) rather than the SQLite/SQLAlchemy storage
    backend, so resuming a study carries no SQLite dependency.

    Args:
        storage_path: Path to the study's journal file. Parent directories
            are created if missing.
        study_name: The study's name (used as the resume key).
        sampler_seed: Seed for `optuna.samplers.TPESampler`, for reproducible sampling.

    Returns:
        The created or resumed `optuna.Study`, with `direction="maximize"`.
    """
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(str(storage_path))
    )
    return optuna.create_study(
        storage=storage,
        study_name=study_name,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=sampler_seed),
    )


def remaining_trials(study: "optuna.Study", n_trials_budget: int) -> int:
    """How many more trials to run so `n_trials_budget` is a *total* across resumes.

    Args:
        study: The (possibly already-populated, resumed) study.
        n_trials_budget: The configured `n_trials`.

    Returns:
        `max(0, n_trials_budget - <trials already in the study>)`.
    """
    return max(0, n_trials_budget - len(study.get_trials(deepcopy=False)))


def export_trial_history(study: "optuna.Study", output_path: Path) -> None:
    """Write the study's full per-trial history to CSV, overwriting any existing file.

    Args:
        study: The study to export.
        output_path: Destination CSV path. Parent directories are created if missing.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(output_path, index=False)


def count_trial_statuses(study: "optuna.Study") -> dict[str, int]:
    """Count every trial in the study by its persisted `status` user-attribute.

    Reads `status` from each trial's `user_attrs` (the caller's objective
    closure must call `trial.set_user_attr("status", ...)` before
    returning), so this reflects every trial across every resumed process
    invocation — unlike an in-memory `trial_cache`, which only ever holds
    the trials the *current* process itself just ran.

    Args:
        study: The study to summarize.

    Returns:
        A dict with keys `"ok"`/`"degenerate"`/`"invalid"`, counting every
        trial whose `status` user-attribute matches one of those values.
    """
    counts = {"ok": 0, "degenerate": 0, "invalid": 0}
    for trial in study.get_trials(deepcopy=False):
        status = trial.user_attrs.get("status")
        if status in counts:
            counts[status] += 1
    return counts


def best_trial_or_none(study: "optuna.Study") -> "optuna.trial.FrozenTrial | None":
    """Return the study's best trial, or `None` if the study has no trials at all.

    Because every trial's returned score is already the single ranking
    criterion, `study.best_trial` already *is* the winner — callers never
    need to re-rank anything.

    Note: `study.best_trial` may be a trial from a *previous* process
    invocation (the study resumed from its file-based storage). Optuna's storage
    only persists a trial's hyperparameters and score, never the caller's
    own in-memory fit artifacts (fitted labels/probabilities/reduced
    matrices) — a resumed best trial is therefore never present in the
    current process's own trial-result cache. Callers must handle a cache
    miss by refitting `study.best_trial.params` directly (HDBSCAN and the
    reducers are both fully deterministic given the same hyperparameters/
    seed, so refitting reproduces the identical result) rather than assuming
    their cache always has an entry for `study.best_trial.number`.

    Args:
        study: The study whose trials were just run (or already existed, resumed).

    Returns:
        `study.best_trial`, or `None` if the study has zero trials.
    """
    if not study.get_trials(deepcopy=False):
        return None
    return study.best_trial
