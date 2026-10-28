"""Reduce criterion embeddings with PaCMAP, then cluster via HDBSCAN, tuned by an Optuna search.

This module supplies PaCMAP's reducer-specific pieces (its own
`n_components`/`n_neighbors`/`mn_ratio`/`fp_ratio` hyperparameter validation
and fit function, plus its own output schema/CLI defaults) to the generic
reduce-then-HDBSCAN orchestration in `reduction_hdbscan_pipeline.py`, which
`umap_hdbscan_clustering.py` shares via the same seam, swapping in its own
UMAP-specific pieces instead. See that module's docstring for the full
read-filter-search-write pipeline description (Optuna study setup/resumption,
reproducibility seed/repeat handling via `reduction_sweep.py`, the composite
objective via `optuna_search.score_reduction_candidate`, and the three output
artifacts written at the paths given via
`--reduced-embeddings-output-path`/`--assignments-output-path`/
`--metrics-output-path`).

PaCMAP has no `min_dist` equivalent to UMAP's — its neighborhood/distance
behavior is instead shaped by `mn_ratio`/`fp_ratio`, which weight mid-near vs.
further point pairs during optimization.

Re-running this step against the same output paths overwrites all three
existing output files (and the trial-history CSV) wholesale; only the
Optuna study's own file-based storage accumulates across runs.
"""

import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pacmap
import pyarrow as pa

# Allow running this file directly (e.g. `uv run src/clustering/pacmap_hdbscan_clustering.py`)
# as well as via `-m` (e.g. `uv run python -m src.clustering.pacmap_hdbscan_clustering`).
# Direct execution leaves __package__ unset and never puts the repo root on
# sys.path, so the src.* imports below would otherwise fail with
# "ModuleNotFoundError: No module named 'src'"; -m already sets both up
# correctly, so this is a no-op in that case.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.paths import REPO_ROOT
from src.clustering.optimization import optuna_search
from src.clustering.reduction_hdbscan_pipeline import ReducerPlugin
from src.clustering import reduction_hdbscan_pipeline

PACMAP_REDUCED_EMBEDDING_PARQUET_SCHEMA = pa.schema(
    [
        ("nct_id", pa.string()),
        ("study_first_posted_date", pa.string()),
        ("criterion_text", pa.string()),
        ("criterion_type", pa.string()),
        ("reduced_embedding", pa.list_(pa.float32())),
        ("embedding_backend", pa.string()),
        ("embedding_model", pa.string()),
        ("reduced_at", pa.timestamp("us", tz="UTC")),
    ]
)


# PaCMAP's own `distance` parameter only accepts {"euclidean", "manhattan",
# "angular", "dot", "hamming"} — it has no "cosine" spelling of its own,
# calling the same cosine-like metric "angular" instead. That "angular" path
# (faiss.IndexHNSWFlat with METRIC_INNER_PRODUCT, inside pacmap's own
# compute_nearest_neighbors) has a confirmed numerical bug in the installed
# pacmap version: at n_neighbors >= 30 it reliably raises "RuntimeWarning:
# overflow encountered in multiply" in its `sqrt(2 * (1 - D))` cosine-
# distance conversion, and can crash a run outright depending on the data —
# reproduced directly against this pipeline's own embeddings. Since the
# embedding backends this pipeline reads from (`bert`/`huggingface`) already
# emit L2-normalized (unit-norm) vectors, Euclidean distance between them is
# a monotonic function of cosine distance (`||a-b||^2 = 2*(1-cos_sim(a,b))`)
# — mathematically equivalent for ranking/reduction purposes, not an
# approximation. So instead of translating "cosine" to pacmap's "angular",
# this module normalizes the input itself and fits with "euclidean",
# sidestepping the buggy "angular" code path entirely. `metric`/`distance`
# stays "cosine" everywhere else in this module (HDBSCAN's own `metric`,
# DBCV, and `optuna_search.compute_dr_metrics`'s `metric == "cosine"` check
# all accept that spelling natively, and have no equivalent bug) — only the
# call into `pacmap.PaCMAP` itself is affected.
_PACMAP_NATIVE_DISTANCES = {"euclidean", "manhattan", "angular", "dot", "hamming"}


def make_fit_pacmap_reduced(distance: str) -> Callable[..., np.ndarray]:
    """Bind PaCMAP's fixed (non-swept) distance metric into a `reduction_sweep`-compatible callable.

    Args:
        distance: The distance metric held fixed across the whole search.
            `"cosine"` normalizes the input and fits with `"euclidean"`
            instead of PaCMAP's own buggy `"angular"` (see
            `_PACMAP_NATIVE_DISTANCES` above); any of PaCMAP's own native
            distance names is passed through unchanged.

    Returns:
        A callable `(embedding_matrix, seed, n_components, n_neighbors,
        mn_ratio, fp_ratio) -> reduced_matrix`, matching the shape
        `reduction_sweep.evaluate_repeat` expects for `fit_reduced`.

    Raises:
        ValueError: If `distance` is neither `"cosine"` nor one of PaCMAP's
            own native distance names.
    """
    if distance == "cosine":
        normalize_before_fit = True
        pacmap_distance = "euclidean"
    elif distance in _PACMAP_NATIVE_DISTANCES:
        normalize_before_fit = False
        pacmap_distance = distance
    else:
        raise ValueError(
            f"Unknown PaCMAP distance {distance!r}; expected 'cosine' or one of {sorted(_PACMAP_NATIVE_DISTANCES)}"
        )

    def fit_reduced(
        embedding_matrix: np.ndarray, seed: int, n_components: int, n_neighbors: int, mn_ratio: float, fp_ratio: float
    ) -> np.ndarray:
        matrix = optuna_search.l2_normalize_rows(embedding_matrix) if normalize_before_fit else embedding_matrix
        reducer = pacmap.PaCMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            MN_ratio=mn_ratio,
            FP_ratio=fp_ratio,
            distance=pacmap_distance,
            random_state=seed,
        )
        return reducer.fit_transform(matrix)

    return fit_reduced


def validate_pacmap_hyperparameters(hyperparams: dict, n_samples: int) -> None:
    """Confirm one PaCMAP hyperparameter combination is usable for this dataset.

    Args:
        hyperparams: This trial's `n_components`/`n_neighbors`/`mn_ratio`/`fp_ratio`.
        n_samples: Number of embeddings available to reduce.

    Raises:
        ValueError: If `n_components` is less than 1, if `mn_ratio` or
            `fp_ratio` is not greater than 0, or if `n_neighbors` is not less
            than `n_samples` (PaCMAP's own neighborhood graph is degenerate
            otherwise) — naming both numbers, pre-empting pacmap's own
            less-contextualized failure for the same condition.
    """
    n_components = hyperparams["n_components"]
    n_neighbors = hyperparams["n_neighbors"]
    mn_ratio = hyperparams["mn_ratio"]
    fp_ratio = hyperparams["fp_ratio"]
    if n_components < 1:
        raise ValueError(f"n_components must be at least 1, got {n_components}")
    if mn_ratio <= 0:
        raise ValueError(f"mn_ratio must be greater than 0, got {mn_ratio}")
    if fp_ratio <= 0:
        raise ValueError(f"fp_ratio must be greater than 0, got {fp_ratio}")
    if n_neighbors >= n_samples:
        raise ValueError(
            f"n_neighbors={n_neighbors} must be less than the number of available embeddings "
            f"({n_samples}); reduce --n-neighbors or provide more data"
        )


PLUGIN = ReducerPlugin(
    name="pacmap",
    display_name="PaCMAP",
    reduced_embedding_schema=PACMAP_REDUCED_EMBEDDING_PARQUET_SCHEMA,
    fixed_metric_config_key="distance",
    winning_config_field="winning_pacmap_config",
    fixed_metric_field="pacmap_fixed_distance",
    make_fit_reduced=make_fit_pacmap_reduced,
    validate_hyperparameters=validate_pacmap_hyperparameters,
)

DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "pacmap_hdbscan_optuna.yaml"


def main() -> None:
    """CLI entry point: run the Optuna search and write reduced embeddings/cluster assignments/metrics."""
    reduction_hdbscan_pipeline.main(
        PLUGIN,
        "Reduce criterion embeddings via PaCMAP, then cluster via HDBSCAN, tuned by an Optuna search.",
        DEFAULT_CONFIG_PATH,
    )


if __name__ == "__main__":
    main()
