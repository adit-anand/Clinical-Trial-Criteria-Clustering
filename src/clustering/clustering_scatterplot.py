"""Render an interactive 2D UMAP scatterplot of an already-clustered condition.

Every clustering step (`kmeans_clustering.py`, `hdbscan_clustering.py`,
`umap_hdbscan_clustering.py`, `pacmap_hdbscan_clustering.py`) writes cluster
assignments and metrics, but none of them render anything visual. This
module reads whichever one of those four methods' output already exists for
a condition (it never re-runs clustering) and writes an interactive HTML
scatterplot of that condition's criteria in a 2-dimensional UMAP space,
colored by cluster assignment, with a legend and a hover tooltip showing
each point's criterion text.

The 2-dimensional projection always reuses that method's own "optimal run"
seed rather than introducing new randomness:

    kmeans          -> `kmeans_params.random_state` (always a concrete value)
    hdbscan         -> no seed concept in this codebase; falls back to
                       `DEFAULT_FALLBACK_SEED`, logged explicitly rather than
                       silently
    umap_hdbscan    -> `winning_seed`; if that run's winning configuration
                       was already 2-dimensional, its persisted reduced
                       embeddings are reused directly instead of re-fitting
                       UMAP
    pacmap_hdbscan  -> `winning_seed` (its own persisted projection is a
                       PaCMAP projection, never substituted for a UMAP one)

Every per-method difference above is captured once in `VISUALIZATION_SPECS`;
no function below branches on which method it's looking at.

Writes one HTML file per condition/method pair, into that method's own
existing output directory, named consistently with its `_clusters.parquet`/
`_cluster_metrics.json` files (e.g.
``data/clustering/hdbscan/<condition>_trials_hdbscan_scatterplot.html``).
Re-running this step for the same condition/method overwrites the existing
file wholesale.
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

from src.preprocessing.criteria_retrieval import EligibilityCriteriaFetcher
from src.clustering import clustering_io
from src.clustering import hdbscan_clustering, kmeans_clustering, pacmap_hdbscan_clustering, umap_hdbscan_clustering

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
        output_dir: That method's own output directory, imported directly
            from its module (never duplicated).
        assignments_suffix: Filename suffix for that method's cluster
            assignments Parquet.
        metrics_suffix: Filename suffix for that method's metrics JSON.
        scatterplot_suffix: Filename suffix for this module's own HTML
            output, mirroring `assignments_suffix`'s naming convention.
        extract_seed_fn: `(metrics_dict) -> seed | None`. `None` means this
            method has no recorded clustering-time seed at all.
        persisted_projection_fn: `(metrics_dict, slug) -> (path, n_components)
            | None`. Non-`None` only when this method already persisted a
            *UMAP* projection usable as-is (currently only `umap_hdbscan`,
            and only when its winning `n_components == 2`).
    """

    key: str
    output_dir: Path
    assignments_suffix: str
    metrics_suffix: str
    scatterplot_suffix: str
    extract_seed_fn: Callable[[dict], int | None]
    persisted_projection_fn: Callable[[dict, str], tuple[Path, int] | None]


def _umap_hdbscan_persisted_projection(metrics: dict, slug: str) -> tuple[Path, int] | None:
    """Return `umap_hdbscan_clustering.py`'s persisted reduced embeddings, only if already 2D."""
    winning_config = metrics.get("winning_umap_config") or {}
    n_components = winning_config.get("n_components")
    if n_components != 2:
        return None
    path = umap_hdbscan_clustering.OUTPUT_DIR / f"{slug}_trials_umap_reduced_embeddings.parquet"
    return path, n_components


def _no_persisted_projection(metrics: dict, slug: str) -> tuple[Path, int] | None:
    """Shared no-op for methods that never persist a reusable UMAP projection."""
    return None


VISUALIZATION_SPECS: dict[str, VisualizationSpec] = {
    "kmeans": VisualizationSpec(
        key="kmeans",
        output_dir=kmeans_clustering.OUTPUT_DIR,
        assignments_suffix="_trials_clusters.parquet",
        metrics_suffix="_trials_cluster_metrics.json",
        scatterplot_suffix="_trials_cluster_scatterplot.html",
        extract_seed_fn=lambda m: (m.get("kmeans_params") or {}).get("random_state"),
        persisted_projection_fn=_no_persisted_projection,
    ),
    "hdbscan": VisualizationSpec(
        key="hdbscan",
        output_dir=hdbscan_clustering.OUTPUT_DIR,
        assignments_suffix="_trials_hdbscan_clusters.parquet",
        metrics_suffix="_trials_hdbscan_cluster_metrics.json",
        scatterplot_suffix="_trials_hdbscan_scatterplot.html",
        extract_seed_fn=lambda m: None,
        persisted_projection_fn=_no_persisted_projection,
    ),
    "umap_hdbscan": VisualizationSpec(
        key="umap_hdbscan",
        output_dir=umap_hdbscan_clustering.OUTPUT_DIR,
        assignments_suffix="_trials_umap_hdbscan_clusters.parquet",
        metrics_suffix="_trials_umap_hdbscan_cluster_metrics.json",
        scatterplot_suffix="_trials_umap_hdbscan_scatterplot.html",
        extract_seed_fn=lambda m: m.get("winning_seed"),
        persisted_projection_fn=_umap_hdbscan_persisted_projection,
    ),
    "pacmap_hdbscan": VisualizationSpec(
        key="pacmap_hdbscan",
        output_dir=pacmap_hdbscan_clustering.OUTPUT_DIR,
        assignments_suffix="_trials_pacmap_hdbscan_clusters.parquet",
        metrics_suffix="_trials_pacmap_hdbscan_cluster_metrics.json",
        scatterplot_suffix="_trials_pacmap_hdbscan_scatterplot.html",
        extract_seed_fn=lambda m: m.get("winning_seed"),
        persisted_projection_fn=_no_persisted_projection,
    ),
}


def load_metrics(spec: VisualizationSpec, slug: str) -> dict:
    """Read a method's metrics JSON for a condition.

    Args:
        spec: The method's `VisualizationSpec`.
        slug: The condition's filesystem-safe slug.

    Returns:
        The parsed metrics dict.

    Raises:
        FileNotFoundError: If the metrics file doesn't exist, naming the
            exact path and which clustering module to run first.
    """
    metrics_path = spec.output_dir / f"{slug}{spec.metrics_suffix}"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"No {spec.key} metrics found at {metrics_path}; run the {spec.key} clustering step "
            "for this condition before generating a scatterplot for it"
        )
    with open(metrics_path, encoding="utf-8") as f:
        return json.load(f)


def load_assignments(spec: VisualizationSpec, slug: str, assignments_path: Path | None = None) -> pa.Table:
    """Read a method's cluster assignments Parquet for a condition.

    Args:
        spec: The method's `VisualizationSpec`.
        slug: The condition's filesystem-safe slug.
        assignments_path: Explicit Parquet path to read instead of the
            default `spec.output_dir / f"{slug}{spec.assignments_suffix}"`
            location — the file whose rows (`nct_id`, `cluster_id`,
            `criterion_text`) become the plotted points. If omitted, the
            default per-method/-condition path is used, as before.

    Returns:
        The cluster assignments table.

    Raises:
        FileNotFoundError: If the resolved assignments file doesn't exist,
            naming the exact path and which clustering module to run first.
    """
    if assignments_path is None:
        assignments_path = spec.output_dir / f"{slug}{spec.assignments_suffix}"
    if not assignments_path.exists():
        raise FileNotFoundError(
            f"No {spec.key} cluster assignments found at {assignments_path}; run the {spec.key} "
            "clustering step for this condition before generating a scatterplot for it"
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


def resolve_points(spec: VisualizationSpec, slug: str, metrics: dict, n_rows: int, seed: int) -> np.ndarray:
    """Resolve the 2D points to plot, reusing a persisted UMAP projection when one exists.

    Args:
        spec: The method's `VisualizationSpec`.
        slug: The condition's filesystem-safe slug.
        metrics: That method's parsed metrics dict for this condition.
        n_rows: Expected row count (the cluster assignments table's row
            count), used to sanity-check row alignment against a freshly
            re-read embeddings file.
        seed: The seed resolved by `resolve_seed` (or a CLI override).

    Returns:
        An `(n_rows, 2)` array of 2D coordinates, row-aligned with the
        cluster assignments table.

    Raises:
        ValueError: If a fresh projection is needed and the re-read
            embeddings file's successfully-embedded row count doesn't match
            `n_rows` (e.g. the embeddings file changed since clustering ran).
    """
    persisted = spec.persisted_projection_fn(metrics, slug)
    if persisted is not None:
        path, n_components = persisted
        logger.info(
            "resolve_points: reusing %s's persisted UMAP projection at %s (n_components=%d)",
            spec.key, path, n_components,
        )
        table = pq.read_table(path)
        return np.array(table.column("reduced_embedding").to_pylist(), dtype=np.float64)

    logger.info("resolve_points: computing a fresh 2D UMAP projection for %s (seed=%d)", spec.key, seed)
    embeddings_path = clustering_io.EMBEDDINGS_INPUT_DIR / f"{slug}_trials_embeddings.parquet"
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


def render_scatterplot(frame: dict[str, list], condition: str, method: str, seed: int, seed_explicit: bool) -> go.Figure:
    """Build the interactive Plotly scatterplot.

    Args:
        frame: As returned by `build_plot_frame`.
        condition: Condition string the plot is for.
        method: The clustering method key being visualized.
        seed: The resolved seed.
        seed_explicit: Whether `seed` was actually recorded by clustering
            (`True`) or a fallback default (`False`) — reflected in the title.

    Returns:
        A `plotly.graph_objects.Figure` with cluster-colored points, a
        legend, and a criterion-text hover tooltip.
    """
    seed_note = f"seed={seed}" if seed_explicit else f"seed={seed} (fallback default; not recorded by clustering)"
    title = f'"{condition}" — {method} clusters (2D UMAP, {seed_note})'
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
    condition: str,
    method: str,
    seed_override: int | None = None,
    input_path: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    """Render and write a 2D UMAP scatterplot for one condition/method pair.

    Args:
        condition: Condition string, e.g. "breast cancer" — must match the
            condition already clustered by `method`, unless `input_path` is
            given (in which case `condition` is only used for the metrics
            lookup, the plot title, and the default output filename).
        method: Which clustering method's output to visualize — one of
            `VISUALIZATION_SPECS`'s keys.
        seed_override: If given, used instead of that method's own
            recorded/fallback seed.
        input_path: Explicit cluster-assignments Parquet path to visualize,
            overriding the default `spec.output_dir /
            f"{slug}{spec.assignments_suffix}"` lookup — the file whose rows
            become the plotted points. If omitted, the default
            per-method/-condition path is used, as before.
        output_path: Where to write the HTML. If omitted, defaults to
            ``spec.output_dir / f"{slug}{spec.scatterplot_suffix}"``.

    Returns:
        The path the scatterplot was written to.

    Raises:
        ValueError: If `method` isn't a recognized key in `VISUALIZATION_SPECS`.
        FileNotFoundError: If the condition hasn't been clustered by `method`
            yet, or if `input_path` is given but doesn't exist.
    """
    if method not in VISUALIZATION_SPECS:
        raise ValueError(f"Unknown method {method!r}; expected one of {sorted(VISUALIZATION_SPECS)}")
    spec = VISUALIZATION_SPECS[method]
    slug = EligibilityCriteriaFetcher.condition_to_slug(condition)

    metrics = load_metrics(spec, slug)
    assignments_table = load_assignments(spec, slug, assignments_path=input_path)

    if seed_override is not None:
        seed, seed_explicit = seed_override, True
        logger.info("run: using CLI-provided --seed=%d override for %s", seed_override, method)
    else:
        seed, seed_explicit = resolve_seed(spec, metrics)

    points_2d = resolve_points(spec, slug, metrics, assignments_table.num_rows, seed)
    frame = build_plot_frame(assignments_table, points_2d)
    fig = render_scatterplot(frame, condition, method, seed, seed_explicit)

    if output_path is None:
        output_path = spec.output_dir / f"{slug}{spec.scatterplot_suffix}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)

    logger.info(
        "run: wrote %s scatterplot for %r (n_points=%d, seed=%d, seed_explicit=%s) -> %s",
        method, condition, assignments_table.num_rows, seed, seed_explicit, output_path,
    )
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for this module.

    Returns:
        An ``argparse.ArgumentParser`` accepting a positional ``condition``
        argument, a required ``--method`` choice, and optional
        seed/input/output overrides.
    """
    parser = argparse.ArgumentParser(
        description="Render an interactive 2D UMAP scatterplot of an already-clustered condition."
    )
    parser.add_argument(
        "condition",
        help='Condition to visualize, e.g. "breast cancer" (must already be clustered by --method)',
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=sorted(VISUALIZATION_SPECS),
        help="Which clustering method's output to visualize",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the 2D UMAP projection's seed (default: reuse that method's own optimal-run seed)",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="Override the default cluster-assignments Parquet path to visualize (the file whose "
        "rows become the plotted points), instead of looking it up from --method/condition",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Override the default output HTML path",
    )
    return parser


def main() -> None:
    """CLI entry point: render and write a scatterplot for one condition/method pair."""
    logging.basicConfig(level=logging.INFO)
    args = build_arg_parser().parse_args()
    run(args.condition, args.method, seed_override=args.seed, input_path=args.input_path, output_path=args.output_path)


if __name__ == "__main__":
    main()
