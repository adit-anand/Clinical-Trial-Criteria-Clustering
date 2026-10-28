"""Shared Optuna-driven hyperparameter search engine for HDBSCAN-based clustering steps.

Used by `reduction_hdbscan_pipeline.run` (shared by `pacmap_hdbscan_clustering.py`
and `umap_hdbscan_clustering.py`) in place of the grid search those modules
used to run. `reduction_hdbscan_pipeline.run` loads its config via
`load_optuna_config`, builds/resumes a study via `build_study`, and runs an
Optuna objective closure that scores each trial via `score_reduction_candidate`.
Every trial — whether it fit cleanly, fit but was degenerate, or never fit at
all — returns a value on the same comparable scale, but `study.best_trial`
is *not* necessarily a trial callers should write to disk: `score_reduction_candidate`
only discourages degenerate trials via `DEGENERATE_TRIAL_PENALTY`, it doesn't
exclude them, so the single best-scoring trial overall can still be
degenerate. `best_ok_trial_or_none` is what callers should use to pick the
trial they actually write out — it only considers trials whose own `status`
user-attribute is `"ok"`. `best_trial_or_none` just distinguishes "the study
has no trials at all" from every other case, regardless of status.

Deliberate exception to this repo's overwrite-on-rerun convention: the
Optuna study's file-based storage (`build_study`) accumulates across runs by
design — re-running the same command resumes the same study rather than
starting over, joining the BERT feature's embedding cache as the second such
exception in this repo. Every other artifact (assignments Parquet, metrics
JSON, trial-history CSV) is still wholesale-overwritten on every run.
"""

import logging
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import optuna
import yaml

from src.clustering import clustering_io
from src.paths import REPO_ROOT

if TYPE_CHECKING:
    from scipy.sparse import csc_matrix

    from src.clustering.bertopic_model import BERTopicFitResult, TopicRepresentation

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

# A trial that fit but whose representative repeat failed
# `HDBSCANClusterer.compute_metrics`'s own eligibility gate (dominant
# cluster, excess noise, or cluster-count out of range) is "degenerate", not
# "invalid" -- it still fit and still has real metrics, so it isn't scored
# with INVALID_TRIAL_SCORE. But `_eligibility_penalties`'s continuous terms
# alone are small relative to the DBCV + dimensionality-reduction-fidelity
# reward terms (realistic rewards are roughly in [-1, 5]), so without a
# decisive penalty a degenerate trial can legitimately outscore an eligible
# one -- which both lets a degenerate trial become the study's reported
# winner and biases the TPE sampler toward resampling the degenerate region.
# Subtracting this constant whenever `metrics_valid` is False guarantees
# `INVALID_TRIAL_SCORE < any degenerate score < any ok score`.
DEGENERATE_TRIAL_PENALTY = 1000.0

# Neighborhood size used for the dimensionality-reduction fidelity metrics
# (subfeature 6) — mirrors zadu's own ZADU.DEFAULT_K.
_DR_METRIC_K = 20
_DR_NORMALIZE_EPSILON = 1e-12

# Standard floor for NPMI (normalized pointwise mutual information) when a
# keyword pair never co-occurs in any document (Lau et al. 2014; also
# gensim's own NPMI coherence uses this same convention) — NPMI's own range
# is otherwise [-1, 1], so -1.0 ("maximally anti-correlated") is already the
# mathematically correct limit as co-occurrence probability -> 0, not an
# arbitrary clamp.
_NPMI_FLOOR = -1.0


@dataclass(frozen=True)
class HyperparamSpec:
    """One hyperparameter's Optuna search-space definition, as parsed from YAML.

    `kind: "int_ratio"` is a dependent hyperparameter: instead of sampling
    `low`/`high` directly, it samples a float ratio in that range and
    resolves to `max(1, round(ratio * <relative_to's already-suggested
    value>))` — see `suggest_hyperparams`. Used to keep e.g. `min_samples`
    bounded relative to `min_cluster_size` rather than swept independently of
    it (an independent sweep lets `min_samples` land far above
    `min_cluster_size`, which HDBSCAN tends to resolve into a degenerate
    dominant-cluster-or-all-noise result). `relative_to` must name another
    key in the same `search_space` block that's suggested earlier (YAML/dict
    key order; `suggest_hyperparams` raises if it isn't).
    """

    kind: Literal["int", "float", "categorical", "int_ratio"]
    low: float | int | None = None
    high: float | int | None = None
    choices: list | None = None
    log: bool = False
    relative_to: str | None = None


@dataclass(frozen=True)
class ObjectiveWeights:
    """Weights for every term folded into a trial's composite objective score.

    `trustworthiness_weight`/`continuity_weight`/`steadiness_weight`/
    `cohesiveness_weight`/`missing_dr_metrics_fallback_score` and
    `topic_diversity_weight`/`inter_topic_redundancy_penalty`/
    `topic_coherence_npmi_weight`/`zeroshot_label_confidence_weight`/
    `topic_size_entropy_weight`/`missing_bertopic_quality_fallback_score`
    are only read by `score_reduction_candidate`.
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
    # BERTopic topic-representation quality terms (see `BERTopicQualityMetrics`).
    topic_diversity_weight: float = 1.0
    inter_topic_redundancy_penalty: float = 1.0
    topic_coherence_npmi_weight: float = 1.0
    zeroshot_label_confidence_weight: float = 1.0
    topic_size_entropy_weight: float = 1.0
    missing_bertopic_quality_fallback_score: float = 0.0


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


@dataclass(frozen=True)
class BERTopicQualityMetrics:
    """BERTopic topic-representation quality scores for one reducer-pipeline trial.

    All five metrics exclude the noise/outlier topic (`topic_id == -1`) —
    noise is already penalized elsewhere via `noise_fraction`/`noise_penalty`.
    See `compute_bertopic_quality_metrics` for exact definitions/edge cases.
    """

    topic_diversity: float
    inter_topic_redundancy: float
    topic_coherence_npmi: float
    zeroshot_label_confidence_rate: float
    topic_size_entropy: float


@dataclass(frozen=True)
class DocumentTermIndex:
    """A binary (0/1) document-occurrence index over a fixed corpus.

    Built once per `run()` call via `build_document_term_index` and reused
    across every trial in a study — `criterion_texts` never changes between
    trials, so rebuilding this per trial (a naive per-call
    `CountVectorizer().fit_transform(criterion_texts)`) would rebuild the
    identical vocabulary/matrix up to `n_trials` times for no benefit.
    """

    vocabulary: dict[str, int]
    doc_term_matrix: "csc_matrix"
    n_docs: int


def _parse_hyperparam_spec(name: str, raw: dict) -> HyperparamSpec:
    """Parse one `search_space` entry from YAML into a `HyperparamSpec`."""
    try:
        return HyperparamSpec(
            kind=raw["kind"],
            low=raw.get("low"),
            high=raw.get("high"),
            choices=raw.get("choices"),
            log=raw.get("log", False),
            relative_to=raw.get("relative_to"),
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
            For an `"int_ratio"` entry, its `relative_to` key must already be
            present in `search_space` *and* ordered before it (dict/YAML key
            order) — this function suggests values in iteration order and
            resolves each `"int_ratio"` against whatever has already been
            suggested.

    Returns:
        One suggested value per key in `search_space`.

    Raises:
        ValueError: If a `HyperparamSpec.kind` isn't
            `"int"`/`"float"`/`"categorical"`/`"int_ratio"`, or if an
            `"int_ratio"` entry's `relative_to` doesn't name an
            already-suggested hyperparameter.
    """
    hyperparams = {}
    for name, spec in search_space.items():
        if spec.kind == "int":
            hyperparams[name] = trial.suggest_int(name, int(spec.low), int(spec.high), log=spec.log)
        elif spec.kind == "float":
            hyperparams[name] = trial.suggest_float(name, float(spec.low), float(spec.high), log=spec.log)
        elif spec.kind == "categorical":
            hyperparams[name] = trial.suggest_categorical(name, spec.choices)
        elif spec.kind == "int_ratio":
            if spec.relative_to not in hyperparams:
                raise ValueError(
                    f"search_space entry {name!r} has kind='int_ratio' with relative_to={spec.relative_to!r}, "
                    "which must name another search_space key already suggested earlier (dict/YAML key order)"
                )
            ratio = trial.suggest_float(name, float(spec.low), float(spec.high), log=spec.log)
            hyperparams[name] = max(1, round(ratio * hyperparams[spec.relative_to]))
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
    """Shared continuous penalty terms behind `score_reduction_candidate`.

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


def _bertopic_quality_reward(bertopic_quality: BERTopicQualityMetrics, weights: ObjectiveWeights) -> float:
    """The 4 BERTopic reward terms (diversity/coherence/zero-shot-confidence/size-entropy), summed.

    `inter_topic_redundancy` is deliberately excluded here — it's a
    penalty, subtracted in `score_reduction_candidate` alongside
    `DEGENERATE_TRIAL_PENALTY` rather than folded into this reward sum.
    """
    return (
        weights.topic_diversity_weight * bertopic_quality.topic_diversity
        + weights.topic_coherence_npmi_weight * bertopic_quality.topic_coherence_npmi
        + weights.zeroshot_label_confidence_weight * bertopic_quality.zeroshot_label_confidence_rate
        + weights.topic_size_entropy_weight * bertopic_quality.topic_size_entropy
    )


def score_reduction_candidate(
    metrics_result: dict,
    relative_validity: float | None,
    labels: np.ndarray,
    min_cluster_size: int,
    weights: ObjectiveWeights,
    dr_metrics: DRMetrics | None,
    bertopic_quality: BERTopicQualityMetrics | None,
    bounds: EligibilityBounds,
) -> float:
    """Composite objective score for a reducer-pipeline trial (maximize).

    Folds relative validity (DBCV) and the shared eligibility penalties
    (`_eligibility_penalties`) with the dimensionality-reduction fidelity
    terms: trustworthiness/continuity/steadiness/cohesiveness are each
    conventionally normalized to `[0, 1]` with higher being better, so each
    is added as a reward. Also folds in the 5 BERTopic topic-quality terms
    (`_bertopic_quality_reward`'s 4 rewards, plus `inter_topic_redundancy`
    subtracted as its own penalty) — these look at the *semantic content*
    BERTopic produced for this trial, which nothing else in this score does.
    Whenever `metrics_result["metrics_valid"]` is `False` (this trial's
    representative repeat is degenerate), an additional
    `DEGENERATE_TRIAL_PENALTY` is subtracted on top of
    `_eligibility_penalties`'s continuous terms, so a degenerate trial can
    never outscore an eligible one regardless of how favorable its DBCV/DR/
    BERTopic reward terms happen to be.

    Args:
        metrics_result: `HDBSCANClusterer.compute_metrics`'s return value for
            this trial's representative repeat.
        relative_validity: The DBCV score for the representative repeat, or
            `None` if not computable.
        labels: The representative repeat's fitted cluster labels.
        min_cluster_size: This trial's `min_cluster_size` hyperparameter.
        weights: The configured objective weights.
        dr_metrics: This trial's `DRMetrics`, or `None` if they couldn't be computed.
        bertopic_quality: This trial's `BERTopicQualityMetrics`, or `None` if
            they couldn't be computed.
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
    if bertopic_quality is not None:
        base += _bertopic_quality_reward(bertopic_quality, weights)
    else:
        base += weights.missing_bertopic_quality_fallback_score
    score = base - _eligibility_penalties(metrics_result, labels, min_cluster_size, weights, bounds)
    if bertopic_quality is not None:
        score -= weights.inter_topic_redundancy_penalty * bertopic_quality.inter_topic_redundancy
    if not metrics_result["metrics_valid"]:
        score -= DEGENERATE_TRIAL_PENALTY
    return score


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


def build_document_term_index(criterion_texts: list[str]) -> DocumentTermIndex | None:
    """Build a binary document-occurrence index over `criterion_texts`, once per `run()` call.

    Uses `CountVectorizer`'s own default tokenizer/preprocessing (lowercasing
    plus its default token pattern) — there's no existing tokenization
    utility in this repo to reuse, and matching BERTopic's own c-TF-IDF
    tokenization exactly is not attempted (see `_topic_keyword_npmi`'s
    vocabulary-mismatch handling for the consequence of that).

    Args:
        criterion_texts: One document string per surviving row (loop-
            invariant across an entire Optuna study — callers should build
            this once in `run()`, not once per trial).

    Returns:
        A `DocumentTermIndex`, or `None` if `CountVectorizer` raises (e.g.
        every document is empty after tokenization, which raises
        `ValueError: empty vocabulary`) — fail-soft, mirroring
        `compute_dr_metrics`'s contract.
    """
    from sklearn.feature_extraction.text import CountVectorizer

    try:
        vectorizer = CountVectorizer(binary=True)
        doc_term_matrix = vectorizer.fit_transform(criterion_texts)
        return DocumentTermIndex(
            vocabulary=vectorizer.vocabulary_,
            doc_term_matrix=doc_term_matrix.tocsc(),
            n_docs=doc_term_matrix.shape[0],
        )
    except Exception:
        logger.exception("build_document_term_index: failed to build the document-term index")
        return None


def _compute_topic_diversity(non_noise: list["TopicRepresentation"]) -> float:
    """Fraction of unique keywords among all non-noise topics' keyword lists.

    Returns `0.0` if there are no non-noise topics or no keywords at all —
    a neutral, non-rewarding value rather than an error.
    """
    all_keywords = [keyword for topic in non_noise for keyword in topic.keywords]
    if not all_keywords:
        return 0.0
    return len(set(all_keywords)) / len(all_keywords)


def _compute_inter_topic_redundancy(
    non_noise: list["TopicRepresentation"], topic_embeddings: dict[int, np.ndarray]
) -> float:
    """Max pairwise cosine similarity between any two non-noise topics' centroid embeddings.

    Returns `0.0` if fewer than 2 non-noise topics (no pair exists to be
    redundant) or if fewer than 2 of them have an entry in `topic_embeddings`
    (defensive; `fit_bertopic_topics`'s own shape-mismatch guard should make
    a missing id unreachable in practice).
    """
    vectors = [topic_embeddings[topic.topic_id] for topic in non_noise if topic.topic_id in topic_embeddings]
    if len(vectors) < 2:
        return 0.0
    matrix = l2_normalize_rows(np.asarray(vectors, dtype=np.float64))
    similarities = matrix @ matrix.T
    np.fill_diagonal(similarities, -np.inf)  # exclude self-similarity (always 1.0)
    return float(np.max(similarities))


def _compute_zeroshot_label_confidence_rate(non_noise: list["TopicRepresentation"]) -> float:
    """Fraction of non-noise topics with exactly one keyword (a confident zero-shot match).

    See `TopicRepresentation.keywords`'s own docstring for why `len(keywords)
    == 1` is the established signal for this, with no extra BERTopic
    plumbing needed. Returns `0.0` if there are no non-noise topics.
    """
    if not non_noise:
        return 0.0
    confident = sum(1 for topic in non_noise if len(topic.keywords) == 1)
    return confident / len(non_noise)


def _compute_topic_size_entropy(non_noise: list["TopicRepresentation"]) -> float:
    """Normalized Shannon entropy of non-noise topics' member counts.

    Returns `1.0` when there are 0 or 1 non-noise topics (trivially
    "nothing to unbalance") or when their total count is 0 (defensive;
    BERTopic shouldn't emit a zero-count topic row, but this still guards
    the division).
    """
    if len(non_noise) <= 1:
        return 1.0
    total = sum(topic.count for topic in non_noise)
    if total == 0:
        return 1.0
    shares = np.array([topic.count / total for topic in non_noise], dtype=np.float64)
    shares = shares[shares > 0]  # 0 * log(0) convention -> drop; contributes 0 to entropy either way
    entropy = -float(np.sum(shares * np.log(shares)))
    return entropy / math.log(len(non_noise))


def _topic_keyword_npmi(word_a: str, word_b: str, document_term_index: DocumentTermIndex) -> float | None:
    """Pairwise NPMI between two keywords, from `document_term_index`'s binary document co-occurrence.

    Returns `None` (not a score) if either word is missing from the
    vectorizer's vocabulary — BERTopic's own c-TF-IDF tokenization and
    `CountVectorizer`'s default tokenizer can disagree on a word's exact
    surface form (e.g. punctuation-adjacent tokens or single characters
    below `CountVectorizer`'s default minimum token length). Callers must
    skip (not zero-fill) any such pair, since `0.0` would misleadingly mean
    "co-occurs no more than chance" rather than "unmeasurable".
    """
    vocabulary = document_term_index.vocabulary
    column_a = vocabulary.get(word_a.lower())
    column_b = vocabulary.get(word_b.lower())
    if column_a is None or column_b is None:
        return None

    matrix = document_term_index.doc_term_matrix
    n_docs = document_term_index.n_docs
    doc_count_a = int(matrix[:, column_a].sum())
    doc_count_b = int(matrix[:, column_b].sum())
    if doc_count_a == 0 or doc_count_b == 0:
        return None  # a vocabulary word with zero document frequency shouldn't occur, but guard anyway

    co_doc_count = int(matrix[:, column_a].multiply(matrix[:, column_b]).sum())
    if co_doc_count == 0:
        return _NPMI_FLOOR

    p_a = doc_count_a / n_docs
    p_b = doc_count_b / n_docs
    p_ab = co_doc_count / n_docs
    return math.log(p_ab / (p_a * p_b)) / -math.log(p_ab)


def _compute_topic_coherence_npmi(
    non_noise: list["TopicRepresentation"], document_term_index: DocumentTermIndex
) -> float:
    """Mean pairwise-keyword NPMI per topic, averaged across non-noise topics.

    A topic with fewer than 2 keywords (including the `len == 1` confident
    zero-shot-match case) contributes no pairs and is skipped entirely, not
    scored as `0.0` — it has no keyword pair to measure coherence between,
    and forcing it to `0.0` would wrongly fight against
    `zeroshot_label_confidence_rate` rewarding exactly that topic. A pair
    where either keyword is out of `document_term_index`'s vocabulary is
    likewise skipped (see `_topic_keyword_npmi`).

    Returns `0.0` if no topic contributes any scorable pair at all (a
    neutral default, not an error).
    """
    per_topic_scores = []
    for topic in non_noise:
        if len(topic.keywords) < 2:
            continue
        pair_scores = [
            score
            for word_a, word_b in combinations(topic.keywords, 2)
            if (score := _topic_keyword_npmi(word_a, word_b, document_term_index)) is not None
        ]
        if pair_scores:
            per_topic_scores.append(sum(pair_scores) / len(pair_scores))

    if not per_topic_scores:
        return 0.0
    return sum(per_topic_scores) / len(per_topic_scores)


def compute_bertopic_quality_metrics(
    bertopic_result: "BERTopicFitResult", document_term_index: DocumentTermIndex | None
) -> BERTopicQualityMetrics | None:
    """Compute the 5 non-noise BERTopic topic-quality metrics for one trial.

    All 5 metrics exclude `topic_id == -1` (noise is penalized elsewhere via
    `noise_fraction`/`noise_penalty`). Mirrors `compute_dr_metrics`'s
    fail-soft `try/except -> None` contract: any unexpected failure anywhere
    in this computation degrades to `None` (-> `missing_bertopic_quality_fallback_score`
    in `score_reduction_candidate`) rather than failing the whole trial.

    Args:
        bertopic_result: This trial's `BERTopicFitResult` (already fit on
            the representative repeat).
        document_term_index: The corpus-wide binary document-occurrence
            index built once per `run()` call via `build_document_term_index`,
            or `None` if that one-time build itself failed — in which case
            `topic_coherence_npmi` alone is set to `0.0` (a neutral score)
            while the other 4 metrics (which don't depend on it) are still
            computed normally.

    Returns:
        A `BERTopicQualityMetrics`, or `None` if computation fails
        unexpectedly.
    """
    try:
        non_noise = [topic for topic in bertopic_result.topics if topic.topic_id != -1]

        topic_coherence_npmi = (
            _compute_topic_coherence_npmi(non_noise, document_term_index)
            if document_term_index is not None
            else 0.0
        )
        return BERTopicQualityMetrics(
            topic_diversity=_compute_topic_diversity(non_noise),
            inter_topic_redundancy=_compute_inter_topic_redundancy(non_noise, bertopic_result.topic_embeddings),
            topic_coherence_npmi=topic_coherence_npmi,
            zeroshot_label_confidence_rate=_compute_zeroshot_label_confidence_rate(non_noise),
            topic_size_entropy=_compute_topic_size_entropy(non_noise),
        )
    except Exception:
        logger.exception("compute_bertopic_quality_metrics: failed to compute BERTopic topic-quality metrics")
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
            The sampler is `multivariate=True`, so it models correlated
            hyperparameters (e.g. `min_cluster_size`/`min_samples`) jointly
            rather than as independent per-parameter marginals.

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
        sampler=optuna.samplers.TPESampler(seed=sampler_seed, multivariate=True),
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


def best_ok_trial_or_none(study: "optuna.Study") -> "optuna.trial.FrozenTrial | None":
    """Return the highest-scoring trial whose own `status` user-attribute is `"ok"`.

    Unlike `best_trial_or_none` (which returns `study.best_trial` regardless
    of status), a `"degenerate"` trial is never eligible here — `score_reduction_candidate`'s
    `DEGENERATE_TRIAL_PENALTY` makes a degenerate trial unlikely to outscore
    an eligible one, but doesn't make it impossible (e.g. if every trial in a
    study happened to be degenerate), and callers that write a trial's result
    to disk (`reduction_hdbscan_pipeline.run`) must never do so for a
    degenerate clustering just because it was the best-scoring trial
    available.

    Args:
        study: The study whose trials were just run (or already existed, resumed).

    Returns:
        The `"ok"` trial with the highest `.value`, or `None` if the study
        has no trial with `status == "ok"` (including if it has no trials at
        all).
    """
    ok_trials = [
        trial
        for trial in study.get_trials(deepcopy=False)
        if trial.user_attrs.get("status") == "ok" and trial.value is not None
    ]
    if not ok_trials:
        return None
    return max(ok_trials, key=lambda trial: trial.value)
