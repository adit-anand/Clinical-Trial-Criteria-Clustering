"""Reduce criterion embeddings with UMAP, then cluster via HDBSCAN, tuned by an Optuna search.

This module supplies UMAP's reducer-specific pieces (its own
`n_components`/`n_neighbors`/`min_dist` hyperparameter validation and fit
function, plus its own output schema/CLI defaults) to the generic
reduce-then-HDBSCAN orchestration in `reduction_hdbscan_pipeline.py`, which
`pacmap_hdbscan_clustering.py` shares via the same seam, swapping in its own
PaCMAP-specific pieces instead. See that module's docstring for the full
read-filter-search-write pipeline description (Optuna study setup/resumption,
reproducibility seed/repeat handling via `reduction_sweep.py`, the composite
objective via `optuna_search.score_reduction_candidate`, and the three output
artifacts written at the paths given via
`--reduced-embeddings-output-path`/`--assignments-output-path`/
`--metrics-output-path`).

Re-running this step against the same output paths overwrites all three
existing output files (and the trial-history CSV) wholesale; only the
Optuna study's own file-based storage accumulates across runs.
"""

import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pyarrow as pa
import umap

# Allow running this file directly (e.g. `uv run src/clustering/umap_hdbscan_clustering.py`)
# as well as via `-m` (e.g. `uv run python -m src.clustering.umap_hdbscan_clustering`).
# Direct execution leaves __package__ unset and never puts the repo root on
# sys.path, so the src.* imports below would otherwise fail with
# "ModuleNotFoundError: No module named 'src'"; -m already sets both up
# correctly, so this is a no-op in that case.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.paths import REPO_ROOT
from src.clustering.reduction_hdbscan_pipeline import ReducerPlugin
from src.clustering import reduction_hdbscan_pipeline

UMAP_REDUCED_EMBEDDING_PARQUET_SCHEMA = pa.schema(
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


def make_fit_umap_reduced(metric: str) -> Callable[..., np.ndarray]:
    """Bind UMAP's fixed (non-swept) distance metric into a `reduction_sweep`-compatible callable.

    Args:
        metric: UMAP's own distance metric, held fixed across the whole search.

    Returns:
        A callable `(embedding_matrix, seed, n_components, n_neighbors,
        min_dist) -> reduced_matrix`, matching the shape
        `reduction_sweep.evaluate_repeat` expects for `fit_reduced`.
    """

    def fit_reduced(
        embedding_matrix: np.ndarray, seed: int, n_components: int, n_neighbors: int, min_dist: float
    ) -> np.ndarray:
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=seed,
        )
        return reducer.fit_transform(embedding_matrix)

    return fit_reduced


def validate_umap_hyperparameters(hyperparams: dict, n_samples: int) -> None:
    """Confirm one UMAP hyperparameter combination is usable for this dataset.

    Args:
        hyperparams: This trial's `n_components`/`n_neighbors`/`min_dist`.
        n_samples: Number of embeddings available to reduce.

    Raises:
        ValueError: If `n_components` is less than 1, if `min_dist` is not in
            `[0.0, 1.0]`, or if `n_neighbors` is not less than `n_samples`
            (UMAP's own neighborhood graph is degenerate otherwise) — naming
            both numbers, pre-empting umap-learn's own less-contextualized
            failure for the same condition.
    """
    n_components = hyperparams["n_components"]
    n_neighbors = hyperparams["n_neighbors"]
    min_dist = hyperparams["min_dist"]
    if n_components < 1:
        raise ValueError(f"n_components must be at least 1, got {n_components}")
    if not (0.0 <= min_dist <= 1.0):
        raise ValueError(f"min_dist must be between 0.0 and 1.0, got {min_dist}")
    if n_neighbors >= n_samples:
        raise ValueError(
            f"n_neighbors={n_neighbors} must be less than the number of available embeddings "
            f"({n_samples}); reduce --n-neighbors or provide more data"
        )


PLUGIN = ReducerPlugin(
    name="umap",
    display_name="UMAP",
    reduced_embedding_schema=UMAP_REDUCED_EMBEDDING_PARQUET_SCHEMA,
    fixed_metric_config_key="metric",
    winning_config_field="winning_umap_config",
    fixed_metric_field="umap_fixed_metric",
    make_fit_reduced=make_fit_umap_reduced,
    validate_hyperparameters=validate_umap_hyperparameters,
)

DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "umap_hdbscan_optuna.yaml"


def main() -> None:
    """CLI entry point: run the Optuna search and write reduced embeddings/cluster assignments/metrics."""
    reduction_hdbscan_pipeline.main(
        PLUGIN,
        "Reduce criterion embeddings via UMAP, then cluster via HDBSCAN, tuned by an Optuna search.",
        DEFAULT_CONFIG_PATH,
    )


if __name__ == "__main__":
    main()
