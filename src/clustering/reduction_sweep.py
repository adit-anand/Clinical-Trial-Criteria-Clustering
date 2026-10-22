"""Generic per-trial seed/repeat evaluation engine shared by dimensionality-reduction + HDBSCAN steps.

`umap_hdbscan_clustering.py` and `pacmap_hdbscan_clustering.py` each reduce a
condition's embeddings with their own reducer (UMAP, PaCMAP) before
clustering the reduced vectors with HDBSCAN. Both share an identical
reproducibility-evaluation shape that has nothing to do with which reducer is
used, so it lives here once. For one hyperparameter combination (`evaluate_config`,
called once per Optuna trial by each of those modules — see their own
Optuna objective closures):

    for each user-supplied seed:
        fit the reducer + HDBSCAN `repeats_per_seed` times (default 10)
        pick the seed's most *reproducible* repeat set: the seed with the
        lowest standard deviation of noise_fraction across its repeats
        wins, not the seed with the best score — a trial's winning seed
        is the one whose repeated runs most agree with each other.
    the overall winning seed's representative repeat (`representative_repeat`)
    is what the caller scores via `optuna_search.score_reduction_candidate`
    and, if it wins the Optuna study, what gets written to disk.

Reproducibility is measured as the standard deviation of `noise_fraction`
(population stdev, via `statistics.pstdev`, since a seed's repeats are
exhaustively run rather than sampled) across a seed's repeats, deliberately
not as a coordinate-level distance between repeats' reduced embeddings:
UMAP/PaCMAP's reduced coordinate systems are not rotation/reflection/scale
invariant across runs, so two equally reproducible *clusterings* could have
globally different coordinate layouts; penalizing that would conflate
reduction-output instability with an irrelevant axis flip.

A reducer-internal failure for one (seed, repeat) unit of work is caught and
recorded as `"errored"` rather than aborting the trial; if every repeat for a
seed errors, that seed is dropped from seed selection; if every seed errors
(or the trial's hyperparameters fail their own precondition check before any
fit is attempted), the whole trial is `"invalid"`. A trial whose winning
seed's representative repeat has degenerate clustering metrics (mirroring
`HDBSCANClusterer.compute_metrics`'s own guard) is marked `"degenerate"` and
is scored accordingly by `optuna_search.score_reduction_candidate`, but is
still recorded.

This module holds no caps of its own (how many seeds/repeats are allowed, or
how many total fits a study may demand) — those live in `optuna_search.py`'s
`load_optuna_config`, which validates a trial's `seeds`/`repeats_per_seed`
before any study runs.
"""

import logging
import statistics
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np

from src.clustering.hdbscan_clustering import HDBSCAN_PARAM_NAMES, HDBSCANClusterer, compute_relative_validity_index

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RepeatResult:
    """The outcome of one reducer fit + one HDBSCAN fit for a fixed (config, seed) pair.

    Attributes:
        repeat_index: 0-based index within this seed's `repeats_per_seed` runs.
        status: `"ok"` if both fits succeeded and `HDBSCANClusterer.compute_metrics`
            reported `metrics_valid=True`; `"degenerate"` if both fits
            succeeded but it didn't (see that method's eligibility
            constraints — cluster count, dominant-cluster share, noise
            fraction); `"errored"` if the reducer or HDBSCAN itself raised.
        detail: Human-readable explanation, set for `"degenerate"`/`"errored"`.
        metrics_result: `HDBSCANClusterer.compute_metrics(...)`'s return
            value, set for `"ok"`/`"degenerate"` (not `"errored"`).
        relative_validity: The DBCV relative validity index, set only for
            `"ok"` repeats where it was itself successfully computed.
        reduced_matrix: This repeat's `(n_samples, n_components)` reducer
            output, kept so the eventual representative repeat never needs
            to be refit.
        labels: This repeat's fitted HDBSCAN labels.
        probabilities: This repeat's fitted HDBSCAN cluster-membership probabilities.
    """

    repeat_index: int
    status: Literal["ok", "degenerate", "errored"]
    detail: str | None = None
    metrics_result: dict | None = None
    relative_validity: float | None = None
    reduced_matrix: np.ndarray | None = None
    labels: np.ndarray | None = None
    probabilities: np.ndarray | None = None


@dataclass(frozen=True)
class SeedResult:
    """The aggregated outcome of `repeats_per_seed` runs for one seed, for one trial.

    Attributes:
        seed: The reducer's `random_state` used for every repeat in this group.
        repeats: Every `RepeatResult` for this seed (length `repeats_per_seed`).
        status: `"ok"` if at least one repeat is `"ok"`/`"degenerate"` (i.e.
            fitting itself succeeded at least once); `"errored"` if every
            repeat errored.
        noise_fraction_std: Population standard deviation of `noise_fraction`
            across this seed's `"ok"`/`"degenerate"` repeats — the primary
            reproducibility signal. `None` if fewer than 2 such repeats exist.
        relative_validity_std: Population standard deviation of
            `relative_validity` across repeats where it was computed, used
            only as a tie-break. `None` if fewer than 2 such values exist.
        representative_repeat_index: Index into `repeats` of the repeat whose
            `noise_fraction` is closest to this seed's own mean — the repeat
            used as "the" result if this seed wins. `None` if no
            `"ok"`/`"degenerate"` repeat exists.
    """

    seed: int
    repeats: list[RepeatResult]
    status: Literal["ok", "errored"]
    noise_fraction_std: float | None = None
    relative_validity_std: float | None = None
    representative_repeat_index: int | None = None


@dataclass(frozen=True)
class ReducerConfigCandidate:
    """The outcome of evaluating one hyperparameter combination across its full seed sweep.

    Attributes:
        hyperparams: This trial's reducer + HDBSCAN hyperparameter values
            (e.g. `{"n_components": 2, "n_neighbors": 15, "min_dist": 0.1,
            "min_cluster_size": 200, ...}`).
        status: `"ok"` if a winning seed was found with valid clustering
            metrics; `"invalid"` if this trial's own precondition check
            failed before any fit was attempted, or if every seed errored;
            `"degenerate"` if a winning seed was found but its representative
            repeat's clustering metrics are invalid.
        detail: Human-readable explanation, set for `"invalid"`/`"degenerate"`.
        seed_results: Every `SeedResult` tried for this trial (full audit trail).
        winning_seed: The most reproducible `SeedResult`, set whenever status
            is `"ok"` or `"degenerate"`.
        notes: Non-fatal, informational caveats about this trial (e.g.
            `n_components >= embedding_dim`).
    """

    hyperparams: dict
    status: Literal["ok", "invalid", "degenerate"]
    detail: str | None = None
    seed_results: list[SeedResult] = field(default_factory=list)
    winning_seed: SeedResult | None = None
    notes: list[str] = field(default_factory=list)


def representative_repeat(candidate: ReducerConfigCandidate) -> RepeatResult:
    """Return the `RepeatResult` a trial's winning seed designated as representative.

    Args:
        candidate: A candidate with `status` in `{"ok", "degenerate"}` (i.e.
            `winning_seed` and `winning_seed.representative_repeat_index`
            are both set).

    Returns:
        The representative `RepeatResult`.
    """
    assert candidate.winning_seed is not None and candidate.winning_seed.representative_repeat_index is not None
    return candidate.winning_seed.repeats[candidate.winning_seed.representative_repeat_index]


def evaluate_repeat(
    fit_reduced: Callable[..., np.ndarray],
    hyperparams: dict,
    seed: int,
    repeat_index: int,
    hdbscan_clusterer: HDBSCANClusterer,
    embedding_matrix: np.ndarray,
) -> RepeatResult:
    """Fit the reducer once, then HDBSCAN once, for a single (config, seed, repeat) unit of work.

    Args:
        fit_reduced: A reducer-specific callable
            `(embedding_matrix, seed, **reducer_hyperparams) -> reduced_matrix`.
            Called with every key of `hyperparams` *except* the HDBSCAN
            hyperparameter names (those belong to `hdbscan_clusterer`, not
            the reducer — see each module's own `build_hdbscan_clusterer`).
        hyperparams: This trial's hyperparameter values — the reducer's own
            plus HDBSCAN's.
        seed: The reducer's `random_state` for this repeat.
        repeat_index: 0-based index within this seed's repeats.
        hdbscan_clusterer: A single `HDBSCANClusterer` instance already
            built for this trial (see `evaluate_config`); reused across
            every repeat since it holds no per-fit state.
        embedding_matrix: The original, non-reduced embedding matrix.

    Returns:
        A `RepeatResult` reflecting whether fitting errored, produced a
        degenerate clustering, or a usable result.
    """
    reducer_hyperparams = {key: value for key, value in hyperparams.items() if key not in HDBSCAN_PARAM_NAMES}
    try:
        reduced_matrix = fit_reduced(embedding_matrix, seed=seed, **reducer_hyperparams)
    except Exception as exc:  # noqa: BLE001 - a reducer-internal failure shouldn't abort the whole trial
        return RepeatResult(repeat_index=repeat_index, status="errored", detail=f"Reduction failed: {exc}")

    try:
        model = hdbscan_clusterer.fit_hdbscan(reduced_matrix)
    except Exception as exc:  # noqa: BLE001 - an HDBSCAN-internal failure shouldn't abort the whole trial
        return RepeatResult(repeat_index=repeat_index, status="errored", detail=f"HDBSCAN fit failed: {exc}")

    labels = model.labels_
    probabilities = model.probabilities_
    metrics_result = hdbscan_clusterer.compute_metrics(reduced_matrix, labels)

    if not metrics_result["metrics_valid"]:
        return RepeatResult(
            repeat_index=repeat_index,
            status="degenerate",
            detail="; ".join(metrics_result["notes"]) or "No usable quality metrics for this repeat",
            metrics_result=metrics_result,
            reduced_matrix=reduced_matrix,
            labels=labels,
            probabilities=probabilities,
        )

    relative_validity = compute_relative_validity_index(reduced_matrix, labels, hdbscan_clusterer.metric)

    return RepeatResult(
        repeat_index=repeat_index,
        status="ok",
        metrics_result=metrics_result,
        relative_validity=relative_validity,
        reduced_matrix=reduced_matrix,
        labels=labels,
        probabilities=probabilities,
    )


def evaluate_seed(
    fit_reduced: Callable[..., np.ndarray],
    hyperparams: dict,
    seed: int,
    repeats_per_seed: int,
    hdbscan_clusterer: HDBSCANClusterer,
    embedding_matrix: np.ndarray,
) -> SeedResult:
    """Run `repeats_per_seed` repeats for one (config, seed) pair and aggregate reproducibility.

    Args:
        fit_reduced: As passed to `evaluate_repeat`.
        hyperparams: This trial's hyperparameter values (reducer's own plus HDBSCAN's).
        seed: The reducer's `random_state` for every repeat in this group.
        repeats_per_seed: Number of repeats to run.
        hdbscan_clusterer: As passed to `evaluate_repeat` — already built for
            this trial by `evaluate_config`.
        embedding_matrix: The original, non-reduced embedding matrix.

    Returns:
        A `SeedResult` with every repeat's outcome plus the aggregated
        reproducibility statistics and representative-repeat choice.
    """
    repeats = [
        evaluate_repeat(fit_reduced, hyperparams, seed, repeat_index, hdbscan_clusterer, embedding_matrix)
        for repeat_index in range(repeats_per_seed)
    ]
    usable = [r for r in repeats if r.status in ("ok", "degenerate")]
    if not usable:
        return SeedResult(seed=seed, repeats=repeats, status="errored")

    noise_fractions = [r.metrics_result["noise_fraction"] for r in usable]
    noise_fraction_std = statistics.pstdev(noise_fractions) if len(noise_fractions) >= 2 else None

    validities = [r.relative_validity for r in usable if r.relative_validity is not None]
    relative_validity_std = statistics.pstdev(validities) if len(validities) >= 2 else None

    mean_noise_fraction = statistics.fmean(noise_fractions)
    representative = min(usable, key=lambda r: abs(r.metrics_result["noise_fraction"] - mean_noise_fraction))

    return SeedResult(
        seed=seed,
        repeats=repeats,
        status="ok",
        noise_fraction_std=noise_fraction_std,
        relative_validity_std=relative_validity_std,
        representative_repeat_index=representative.repeat_index,
    )


def select_best_seed(seed_results: list[SeedResult]) -> SeedResult | None:
    """Pick the most reproducible seed: lowest noise_fraction_std, tie-broken by relative_validity_std then seed.

    Args:
        seed_results: As produced by `evaluate_seed`, one per seed tried.

    Returns:
        The `"ok"` `SeedResult` with the lowest `noise_fraction_std` (`None`
        treated as worst), or `None` if every seed errored.
    """
    eligible = [s for s in seed_results if s.status == "ok"]
    if not eligible:
        return None

    def sort_key(seed_result: SeedResult) -> tuple[float, float, int]:
        noise_std = seed_result.noise_fraction_std if seed_result.noise_fraction_std is not None else float("inf")
        validity_std = (
            seed_result.relative_validity_std if seed_result.relative_validity_std is not None else float("inf")
        )
        return (noise_std, validity_std, seed_result.seed)

    return min(eligible, key=sort_key)


def evaluate_config(
    fit_reduced: Callable[..., np.ndarray],
    hyperparams: dict,
    seeds: list[int],
    repeats_per_seed: int,
    validate_fn: Callable[[dict, int], None],
    build_hdbscan_clusterer: Callable[[dict], HDBSCANClusterer],
    embedding_matrix: np.ndarray,
    n_samples: int,
) -> ReducerConfigCandidate:
    """Validate, then sweep every seed for one hyperparameter combination (one Optuna trial).

    Args:
        fit_reduced: As passed to `evaluate_repeat`.
        hyperparams: This trial's hyperparameter values — the reducer's own
            plus HDBSCAN's (see `build_hdbscan_clusterer`).
        seeds: The full configured seed list.
        repeats_per_seed: Number of repeats to run per seed.
        validate_fn: A reducer-specific `(hyperparams, n_samples) -> None`
            precondition check; raises `ValueError` if this trial is
            unusable for the dataset (e.g. `n_neighbors >= n_samples`).
        build_hdbscan_clusterer: `(hyperparams) -> HDBSCANClusterer` —
            constructs this trial's `HDBSCANClusterer` from `hyperparams`.
            Built once per trial, then reused across every seed/repeat for
            that trial, since the clusterer holds no per-fit state.
        embedding_matrix: The original, non-reduced embedding matrix.
        n_samples: `embedding_matrix.shape[0]`.

    Returns:
        A `ReducerConfigCandidate` reflecting whether this trial was
        invalid, degenerate, or produced a usable result.
    """
    try:
        validate_fn(hyperparams, n_samples)
    except ValueError as exc:
        return ReducerConfigCandidate(hyperparams=hyperparams, status="invalid", detail=str(exc))

    hdbscan_clusterer = build_hdbscan_clusterer(hyperparams)
    try:
        hdbscan_clusterer.validate_hyperparameters(n_samples)
    except ValueError as exc:
        return ReducerConfigCandidate(hyperparams=hyperparams, status="invalid", detail=str(exc))

    seed_results = [
        evaluate_seed(fit_reduced, hyperparams, seed, repeats_per_seed, hdbscan_clusterer, embedding_matrix)
        for seed in seeds
    ]
    winning_seed = select_best_seed(seed_results)
    if winning_seed is None:
        return ReducerConfigCandidate(
            hyperparams=hyperparams,
            status="invalid",
            detail="Every seed errored for this trial (reduction or HDBSCAN fitting failed for all repeats)",
            seed_results=seed_results,
        )

    candidate = ReducerConfigCandidate(
        hyperparams=hyperparams,
        status="ok",
        seed_results=seed_results,
        winning_seed=winning_seed,
    )
    representative = representative_repeat(candidate)
    if representative.status == "degenerate":
        return ReducerConfigCandidate(
            hyperparams=hyperparams,
            status="degenerate",
            detail=representative.detail,
            seed_results=seed_results,
            winning_seed=winning_seed,
        )
    return candidate


def log_config_candidate(trial_number: int, candidate: ReducerConfigCandidate) -> None:
    """Log one INFO line summarizing a single evaluated trial's hyperparameter combination."""
    if candidate.status == "invalid":
        logger.info("evaluate_config: trial #%d %s -> invalid (%s)", trial_number, candidate.hyperparams, candidate.detail)
        return
    representative = representative_repeat(candidate)
    if candidate.status == "degenerate":
        logger.info(
            "evaluate_config: trial #%d %s -> degenerate, winning_seed=%d, noise=%.1f%% (%s)",
            trial_number, candidate.hyperparams, candidate.winning_seed.seed,
            representative.metrics_result["noise_fraction"] * 100, candidate.detail,
        )
        return
    logger.info(
        "evaluate_config: trial #%d %s -> ok, winning_seed=%d (noise_fraction_std=%s), clusters=%d, "
        "noise=%.1f%%, silhouette=%s, relative_validity=%s",
        trial_number, candidate.hyperparams, candidate.winning_seed.seed, candidate.winning_seed.noise_fraction_std,
        representative.metrics_result["n_clusters_actual"], representative.metrics_result["noise_fraction"] * 100,
        representative.metrics_result["metrics"]["silhouette_score"], representative.relative_validity,
    )


def serialize_candidates(candidates: list[ReducerConfigCandidate]) -> list[dict]:
    """Render the full config/seed/repeat audit trail as JSON-serializable dicts.

    Each repeat's record is kept deliberately minimal (status, noise
    fraction, relative validity, cluster count only — not the full
    `compute_metrics` output).

    Args:
        candidates: One or more `ReducerConfigCandidate`s to serialize (e.g.
            every trial cached across an Optuna study, for ad-hoc inspection
            — the Optuna trial-history CSV is the primary audit trail now).

    Returns:
        One dict per candidate, suitable for ad-hoc JSON embedding.
    """
    serialized = []
    for candidate in candidates:
        serialized.append(
            {
                "hyperparams": candidate.hyperparams,
                "status": candidate.status,
                "detail": candidate.detail,
                "notes": candidate.notes,
                "winning_seed": candidate.winning_seed.seed if candidate.winning_seed else None,
                "seed_results": [
                    {
                        "seed": seed_result.seed,
                        "status": seed_result.status,
                        "noise_fraction_std": seed_result.noise_fraction_std,
                        "relative_validity_std": seed_result.relative_validity_std,
                        "representative_repeat_index": seed_result.representative_repeat_index,
                        "repeats": [
                            {
                                "repeat_index": repeat.repeat_index,
                                "status": repeat.status,
                                "noise_fraction": (
                                    repeat.metrics_result["noise_fraction"] if repeat.metrics_result else None
                                ),
                                "relative_validity": repeat.relative_validity,
                                "n_clusters_actual": (
                                    repeat.metrics_result["n_clusters_actual"] if repeat.metrics_result else None
                                ),
                            }
                            for repeat in seed_result.repeats
                        ],
                    }
                    for seed_result in candidate.seed_results
                ],
            }
        )
    return serialized
