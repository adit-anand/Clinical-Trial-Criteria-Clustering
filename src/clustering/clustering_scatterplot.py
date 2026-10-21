"""Render an interactive 2D UMAP scatterplot of an already-clustered method's output.

Every clustering step (`kmeans_clustering.py`, `hdbscan_clustering.py`,
`umap_hdbscan_clustering.py`, `pacmap_hdbscan_clustering.py`) writes cluster
assignments and metrics, but none of them render anything visual. This
module reads whichever one of those four methods' output the caller points
it at via explicit path flags (it never re-runs clustering) and writes an
interactive HTML scatterplot of those criteria in a 2-dimensional UMAP
space, colored by cluster assignment, with a legend and a hover tooltip
showing each point's criterion text.

The 2-dimensional projection always reuses that method's own "optimal run"
seed rather than introducing new randomness:

    kmeans          -> `kmeans_params.random_state` (always a concrete value)
    hdbscan         -> no seed concept in this codebase; falls back to
                       `DEFAULT_FALLBACK_SEED`, logged explicitly rather than
                       silently
    umap_hdbscan    -> `winning_seed`; if that run's winning configuration
                       was already 2-dimensional, its persisted reduced
                       embeddings are reused directly (via
                       `--reduced-embeddings-path`) instead of re-fitting UMAP
    pacmap_hdbscan  -> `winning_seed` (its own persisted projection is a
                       PaCMAP projection, never substituted for a UMAP one)

Every per-method difference above is captured once in `VISUALIZATION_SPECS`;
no function below branches on which method it's looking at.

Writes one HTML file, at the path given via `--output-path`. Re-running this
step against the same output path overwrites the existing file wholesale.
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pyarrow as pa
import pyarrow.parquet as pq
import umap

# Allow running this file directly (e.g. `uv run src/clustering/clustering_scatterplot.py`)
# as well as via `-m` (e.g. `uv run python -m src.clustering.clustering_scatterplot`).
# Direct execution leaves __package__ unset and never puts the repo root on
# sys.path, so the src.* imports below would otherwise fail with
# "ModuleNotFoundError: No module named 'src'"; -m already sets both up
# correctly, so this is a no-op in that case.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.paths import label_from_slug, slug_from_filename
from src.clustering import clustering_io

logger = logging.getLogger(__name__)

# Used only when a method has no recorded clustering-time seed at all
# (currently just plain `hdbscan_clustering.py`) — matches this project's
# existing seed-default convention (e.g. KMeansClusterer.DEFAULT_RANDOM_STATE).
DEFAULT_FALLBACK_SEED = 42


@dataclass(frozen=True)
class VisualizationSpec:
    """Everything this module needs to know about one clustering method.

    Attributes:
        key: Short identifier, also the CLI's `--method` value.
        assignments_suffix: Filename suffix for that method's cluster
            assignments Parquet — used only to recover a human-readable
            label from the `--assignments-path` the caller provides.
        extract_seed_fn: `(metrics_dict) -> seed | None`. `None` means this
            method has no recorded clustering-time seed at all.
        persisted_projection_fn: `(metrics_dict, reduced_embeddings_path) ->
            (path, n_components) | None`. Non-`None` only when this method
            already persisted a *UMAP* projection usable as-is (currently
            only `umap_hdbscan`, and only when its winning `n_components ==
            2` and the caller passed `--reduced-embeddings-path`).
    """

    key: str
    assignments_suffix: str
    extract_seed_fn: Callable[[dict], int | None]
    persisted_projection_fn: Callable[[dict, Path | None], tuple[Path, int] | None]


def _umap_hdbscan_persisted_projection(metrics: dict, reduced_embeddings_path: Path | None) -> tuple[Path, int] | None:
    """Return `umap_hdbscan_clustering.py`'s persisted reduced embeddings, only if already 2D and given."""
    if reduced_embeddings_path is None:
        return None
    winning_config = metrics.get("winning_umap_config") or {}
    n_components = winning_config.get("n_components")
    if n_components != 2:
        return None
    return reduced_embeddings_path, n_components


def _no_persisted_projection(metrics: dict, reduced_embeddings_path: Path | None) -> tuple[Path, int] | None:
    """Shared no-op for methods that never persist a reusable UMAP projection."""
    return None


VISUALIZATION_SPECS: dict[str, VisualizationSpec] = {
    "kmeans": VisualizationSpec(
        key="kmeans",
        assignments_suffix="_trials_clusters.parquet",
        extract_seed_fn=lambda m: (m.get("kmeans_params") or {}).get("random_state"),
        persisted_projection_fn=_no_persisted_projection,
    ),
    "hdbscan": VisualizationSpec(
        key="hdbscan",
        assignments_suffix="_trials_hdbscan_clusters.parquet",
        extract_seed_fn=lambda m: None,
        persisted_projection_fn=_no_persisted_projection,
    ),
    "umap_hdbscan": VisualizationSpec(
        key="umap_hdbscan",
        assignments_suffix="_trials_umap_hdbscan_clusters.parquet",
        extract_seed_fn=lambda m: m.get("winning_seed"),
        persisted_projection_fn=_umap_hdbscan_persisted_projection,
    ),
    "pacmap_hdbscan": VisualizationSpec(
        key="pacmap_hdbscan",
        assignments_suffix="_trials_pacmap_hdbscan_clusters.parquet",
        extract_seed_fn=lambda m: m.get("winning_seed"),
        persisted_projection_fn=_no_persisted_projection,
    ),
}


def load_metrics(spec: VisualizationSpec, metrics_path: Path) -> dict:
    """Read a method's metrics JSON.

    Args:
        spec: The method's `VisualizationSpec`.
        metrics_path: Path to that method's metrics JSON.

    Returns:
        The parsed metrics dict.

    Raises:
        FileNotFoundError: If `metrics_path` doesn't exist, naming which
            clustering module to run first.
    """
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"No {spec.key} metrics found at {metrics_path}; run the {spec.key} clustering step "
            "first and pass its metrics path via --metrics-path"
        )
    with open(metrics_path, encoding="utf-8") as f:
        return json.load(f)


def load_assignments(spec: VisualizationSpec, assignments_path: Path) -> pa.Table:
    """Read a method's cluster assignments Parquet.

    Args:
        spec: The method's `VisualizationSpec`.
        assignments_path: Path to the Parquet file whose rows (`nct_id`,
            `cluster_id`, `criterion_text`) become the plotted points.

    Returns:
        The cluster assignments table.

    Raises:
        FileNotFoundError: If `assignments_path` doesn't exist, naming which
            clustering module to run first.
    """
    if not assignments_path.exists():
        raise FileNotFoundError(
            f"No {spec.key} cluster assignments found at {assignments_path}; run the {spec.key} "
            "clustering step first and pass its assignments path via --assignments-path"
        )
    return pq.read_table(assignments_path)


def resolve_seed(spec: VisualizationSpec, metrics: dict) -> tuple[int, bool]:
    """Resolve the seed to use for this method's 2D UMAP projection.

    Args:
        spec: The method's `VisualizationSpec`.
        metrics: That method's parsed metrics dict for this condition.

    Returns:
        `(seed, explicit)` — `explicit` is `False` only when the method has
        no recorded clustering-time seed at all and `DEFAULT_FALLBACK_SEED`
        was used instead (logged as a warning, never silently).
    """
    seed = spec.extract_seed_fn(metrics)
    if seed is not None:
        return seed, True
    logger.warning(
        "resolve_seed: %s has no recorded clustering-time seed for this condition; falling back "
        "to DEFAULT_FALLBACK_SEED=%d (this is not the actual seed used during clustering, since "
        "none was recorded)",
        spec.key, DEFAULT_FALLBACK_SEED,
    )
    return DEFAULT_FALLBACK_SEED, False


def resolve_points(
    spec: VisualizationSpec,
    metrics: dict,
    n_rows: int,
    seed: int,
    embeddings_path: Path | None,
    reduced_embeddings_path: Path | None,
) -> np.ndarray:
    """Resolve the 2D points to plot, reusing a persisted UMAP projection when one is given.

    Args:
        spec: The method's `VisualizationSpec`.
        metrics: That method's parsed metrics dict.
        n_rows: Expected row count (the cluster assignments table's row
            count), used to sanity-check row alignment against a freshly
            re-read embeddings file.
        seed: The seed resolved by `resolve_seed` (or a CLI override).
        embeddings_path: Embeddings Parquet path to read when a fresh
            projection is needed (every method except `umap_hdbscan` reusing
            an already-2D persisted projection). Required in that case;
            ignored when a persisted projection is reused instead.
        reduced_embeddings_path: `umap_hdbscan`'s persisted reduced-embeddings
            Parquet path, reused as-is if its winning config was already 2D.
            Ignored for every other method.

    Returns:
        An `(n_rows, 2)` array of 2D coordinates, row-aligned with the
        cluster assignments table.

    Raises:
        ValueError: If a fresh projection is needed and `embeddings_path` was
            not given, or if the re-read embeddings file's
            successfully-embedded row count doesn't match `n_rows` (e.g. the
            embeddings file changed since clustering ran).
    """
    persisted = spec.persisted_projection_fn(metrics, reduced_embeddings_path)
    if persisted is not None:
        path, n_components = persisted
        logger.info(
            "resolve_points: reusing %s's persisted UMAP projection at %s (n_components=%d)",
            spec.key, path, n_components,
        )
        table = pq.read_table(path)
        return np.array(table.column("reduced_embedding").to_pylist(), dtype=np.float64)

    logger.info("resolve_points: computing a fresh 2D UMAP projection for %s (seed=%d)", spec.key, seed)
    if embeddings_path is None:
        raise ValueError(
            f"A fresh 2D UMAP projection is needed for {spec.key}, but no --embeddings-path was given"
        )
    table = clustering_io.read_embeddings_parquet(embeddings_path)
    filtered = clustering_io.filter_successful_rows(table)
    if filtered.num_rows != n_rows:
        raise ValueError(
            f"Embeddings file at {embeddings_path} has {filtered.num_rows} successfully-embedded "
            f"row(s), but {spec.key}'s cluster assignments have {n_rows}; the embeddings file may "
            "have changed since clustering was run, so row alignment can't be guaranteed"
        )
    embedding_matrix = clustering_io.extract_embedding_matrix(filtered)
    reducer = umap.UMAP(n_components=2, random_state=seed)
    return reducer.fit_transform(embedding_matrix)


def build_plot_frame(assignments_table: pa.Table, points_2d: np.ndarray) -> dict[str, list]:
    """Zip cluster assignments with their 2D coordinates into a plotting-ready frame.

    Args:
        assignments_table: As returned by `load_assignments`.
        points_2d: As returned by `resolve_points`, row-aligned with `assignments_table`.

    Returns:
        A dict with keys `x`, `y`, `cluster_label` (`"noise"` for `cluster_id
        == -1`; k-means never produces this), `criterion_text`, `nct_id`.
    """
    cluster_ids = assignments_table.column("cluster_id").to_pylist()
    cluster_labels = ["noise" if cluster_id == -1 else str(cluster_id) for cluster_id in cluster_ids]
    return {
        "x": points_2d[:, 0].tolist(),
        "y": points_2d[:, 1].tolist(),
        "cluster_label": cluster_labels,
        "criterion_text": assignments_table.column("criterion_text").to_pylist(),
        "nct_id": assignments_table.column("nct_id").to_pylist(),
    }


def render_scatterplot(frame: dict[str, list], label: str, method: str, seed: int, seed_explicit: bool) -> go.Figure:
    """Build the interactive Plotly scatterplot.

    Args:
        frame: As returned by `build_plot_frame`.
        label: Human-readable label for the plot title, derived from the
            cluster-assignments filename.
        method: The clustering method key being visualized.
        seed: The resolved seed.
        seed_explicit: Whether `seed` was actually recorded by clustering
            (`True`) or a fallback default (`False`) — reflected in the title.

    Returns:
        A `plotly.graph_objects.Figure` with cluster-colored points, a
        legend, and a criterion-text hover tooltip.
    """
    seed_note = f"seed={seed}" if seed_explicit else f"seed={seed} (fallback default; not recorded by clustering)"
    title = f'"{label}" — {method} clusters (2D UMAP, {seed_note})'
    fig = px.scatter(
        frame,
        x="x",
        y="y",
        color="cluster_label",
        hover_data={"criterion_text": True, "nct_id": True, "x": False, "y": False},
        title=title,
        labels={"cluster_label": "Cluster"},
    )
    return fig


def run(
    method: str,
    metrics_path: Path,
    assignments_path: Path,
    output_path: Path,
    seed_override: int | None = None,
    embeddings_path: Path | None = None,
    reduced_embeddings_path: Path | None = None,
) -> Path:
    """Render and write a 2D UMAP scatterplot for one method's clustering output.

    Args:
        method: Which clustering method's output to visualize — one of
            `VISUALIZATION_SPECS`'s keys.
        metrics_path: That method's metrics JSON path.
        assignments_path: That method's cluster-assignments Parquet path —
            the file whose rows become the plotted points.
        output_path: Where to write the HTML.
        seed_override: If given, used instead of that method's own
            recorded/fallback seed.
        embeddings_path: Embeddings Parquet path to read when a fresh 2D
            projection is needed (see `resolve_points`) — required unless
            `reduced_embeddings_path` lets a persisted 2D projection be
            reused instead.
        reduced_embeddings_path: `umap_hdbscan`'s persisted
            reduced-embeddings Parquet path, reused as-is if its winning
            config was already 2D. Ignored for every other method.

    Returns:
        The path the scatterplot was written to.

    Raises:
        ValueError: If `method` isn't a recognized key in
            `VISUALIZATION_SPECS`, or if a fresh projection is needed and
            `embeddings_path` was not given.
        FileNotFoundError: If `metrics_path` or `assignments_path` doesn't exist.
    """
    if method not in VISUALIZATION_SPECS:
        raise ValueError(f"Unknown method {method!r}; expected one of {sorted(VISUALIZATION_SPECS)}")
    spec = VISUALIZATION_SPECS[method]
    label = label_from_slug(slug_from_filename(assignments_path.name, spec.assignments_suffix))

    metrics = load_metrics(spec, metrics_path)
    assignments_table = load_assignments(spec, assignments_path)

    if seed_override is not None:
        seed, seed_explicit = seed_override, True
        logger.info("run: using CLI-provided --seed=%d override for %s", seed_override, method)
    else:
        seed, seed_explicit = resolve_seed(spec, metrics)

    points_2d = resolve_points(
        spec, metrics, assignments_table.num_rows, seed,
        embeddings_path=embeddings_path, reduced_embeddings_path=reduced_embeddings_path,
    )
    frame = build_plot_frame(assignments_table, points_2d)
    fig = render_scatterplot(frame, label, method, seed, seed_explicit)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)

    logger.info(
        "run: wrote %s scatterplot for %r (n_points=%d, seed=%d, seed_explicit=%s) -> %s",
        method, label, assignments_table.num_rows, seed, seed_explicit, output_path,
    )
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for this module.

    Returns:
        An ``argparse.ArgumentParser`` accepting a required ``--method``
        choice, required metrics/assignments/output path flags, and
        optional seed/embeddings-path overrides.
    """
    parser = argparse.ArgumentParser(
        description="Render an interactive 2D UMAP scatterplot of an already-clustered method's output."
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=sorted(VISUALIZATION_SPECS),
        help="Which clustering method's output to visualize",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        required=True,
        help="That method's metrics JSON path",
    )
    parser.add_argument(
        "--assignments-path",
        type=Path,
        required=True,
        help="That method's cluster-assignments Parquet path (the file whose rows become the plotted points)",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Output HTML path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the 2D UMAP projection's seed (default: reuse that method's own optimal-run seed)",
    )
    parser.add_argument(
        "--embeddings-path",
        type=Path,
        default=None,
        help="Embeddings Parquet path to read when a fresh 2D projection is needed (every method "
        "except umap_hdbscan reusing an already-2D persisted projection via --reduced-embeddings-path)",
    )
    parser.add_argument(
        "--reduced-embeddings-path",
        type=Path,
        default=None,
        help="For --method umap_hdbscan only: that run's persisted reduced-embeddings Parquet path, "
        "reused as-is instead of fitting a fresh projection if its winning config was already 2D",
    )
    return parser


def main() -> None:
    """CLI entry point: render and write a scatterplot for one method's clustering output."""
    logging.basicConfig(level=logging.INFO)
    args = build_arg_parser().parse_args()
    run(
        args.method,
        args.metrics_path,
        args.assignments_path,
        args.output_path,
        seed_override=args.seed,
        embeddings_path=args.embeddings_path,
        reduced_embeddings_path=args.reduced_embeddings_path,
    )


if __name__ == "__main__":
    main()
