"""Render an interactive 2D UMAP scatterplot of an already-clustered method's output.

Every clustering step (`umap_hdbscan_clustering.py`,
`pacmap_hdbscan_clustering.py`) writes cluster assignments and metrics, but
none of them render anything visual. This module reads whichever one of
those two methods' output the caller points it at via explicit path flags
(it never re-runs clustering) and writes an interactive HTML scatterplot of
those criteria in a 2-dimensional UMAP space, colored by cluster assignment,
with a legend naming each cluster by its own topic label (not a bare id)
and a hover tooltip showing each point's criterion text. Cluster colors are
drawn from `PASTEL_COLORBLIND_PALETTE` -- a pastel (lightened-toward-white)
variant of Plotly's own colorblind-safe `Safe` qualitative palette, so hue
separation (the actual colorblind-accessibility signal) is unchanged from
the original, more saturated palette.

The 2-dimensional projection always reuses that method's own "optimal run"
seed (`winning_seed`) rather than introducing new randomness, and is always
freshly fit from `--embeddings-path` -- neither method persists a reusable
projection of its own.

Every per-method difference above is captured once in `VISUALIZATION_SPECS`;
no function below branches on which method it's looking at.

Writes one HTML file, at the path given via `--output-path`. Re-running this
step against the same output path overwrites the existing file wholesale.
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pyarrow as pa
import pyarrow.parquet as pq
import umap

from amia_2026.logging_utils import configure_logging
from amia_2026.paths import label_from_slug, slug_from_filename
from amia_2026.clustering import clustering_io

logger = logging.getLogger(__name__)

# Used only when a method has no recorded clustering-time seed at all. No
# currently-registered method lacks one (every `VisualizationSpec` below
# resolves a concrete seed), but this stays as a safety net for a future
# method that might not.
DEFAULT_FALLBACK_SEED = 42


def _parse_rgb(rgb_str: str) -> tuple[int, int, int]:
    """Parse a plotly `"rgb(r, g, b)"` string into an `(r, g, b)` int tuple."""
    r, g, b = (int(part.strip()) for part in rgb_str.split("(")[1].split(")")[0].split(","))
    return r, g, b


def _pastelize(rgb_str: str, white_mix: float) -> str:
    """Lighten one `"rgb(r, g, b)"` color by blending it toward white, preserving its hue.

    Hue (not saturation/lightness) is what makes a categorical palette
    colorblind-distinguishable, so blending toward white keeps that
    distinguishability while producing a pastel look.

    Args:
        rgb_str: A plotly-style `"rgb(r, g, b)"` color string.
        white_mix: Fraction of white to blend in, `0.0` (unchanged) to `1.0` (pure white).

    Returns:
        A new `"rgb(r, g, b)"` string, each channel moved `white_mix` of the way to 255.
    """
    r, g, b = _parse_rgb(rgb_str)
    blended = tuple(round(c * (1 - white_mix) + 255 * white_mix) for c in (r, g, b))
    return f"rgb({blended[0]}, {blended[1]}, {blended[2]})"


# Plotly's own `Safe` qualitative palette (from CARTOColors) is explicitly
# designed to stay distinguishable under the common forms of color vision
# deficiency. Its own trailing entry is a neutral gray, intended for
# "not applicable" categories -- reserved here as `NOISE_COLOR` rather than
# pastelized, so noise points stay visually distinct from every pastelized
# real cluster. Every other entry is lightened toward white (see
# `_pastelize`) so the palette reads as pastel without losing the hue
# separation that makes it colorblind-safe in the first place.
_PASTEL_WHITE_MIX = 0.45
_SAFE_PALETTE = px.colors.qualitative.Safe
NOISE_COLOR = _SAFE_PALETTE[-1]
PASTEL_COLORBLIND_PALETTE = [_pastelize(color, _PASTEL_WHITE_MIX) for color in _SAFE_PALETTE[:-1]]


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
    """

    key: str
    assignments_suffix: str
    extract_seed_fn: Callable[[dict], int | None]


VISUALIZATION_SPECS: dict[str, VisualizationSpec] = {
    "umap_hdbscan": VisualizationSpec(
        key="umap_hdbscan",
        assignments_suffix="_trials_umap_hdbscan_clusters.parquet",
        extract_seed_fn=lambda m: m.get("winning_seed"),
    ),
    "pacmap_hdbscan": VisualizationSpec(
        key="pacmap_hdbscan",
        assignments_suffix="_trials_pacmap_hdbscan_clusters.parquet",
        extract_seed_fn=lambda m: m.get("winning_seed"),
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
    n_rows: int,
    seed: int,
    embeddings_path: Path | None,
) -> np.ndarray:
    """Resolve the 2D points to plot by fitting a fresh 2D UMAP projection.

    Args:
        spec: The method's `VisualizationSpec`.
        n_rows: Expected row count (the cluster assignments table's row
            count), used to sanity-check row alignment against the
            freshly-read embeddings file.
        seed: The seed resolved by `resolve_seed` (or a CLI override).
        embeddings_path: Embeddings Parquet path to fit the projection from. Required.

    Returns:
        An `(n_rows, 2)` array of 2D coordinates, row-aligned with the
        cluster assignments table.

    Raises:
        ValueError: If `embeddings_path` was not given, or if the re-read
            embeddings file's successfully-embedded row count doesn't match
            `n_rows` (e.g. the embeddings file changed since clustering ran).
    """
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
        A dict with keys `x`, `y`, `cluster_label` (`"Noise"` for
        `cluster_id == -1`; `"<cluster_id>: <topic_label>"` otherwise, so
        the legend names each cluster by its own topic label rather than a
        bare id), `criterion_text`, `nct_id`.
    """
    cluster_ids = assignments_table.column("cluster_id").to_pylist()
    topic_labels = assignments_table.column("topic_label").to_pylist()
    cluster_labels = [
        "Noise" if cluster_id == -1 else f"{cluster_id}: {topic_label}"
        for cluster_id, topic_label in zip(cluster_ids, topic_labels)
    ]
    return {
        "x": points_2d[:, 0].tolist(),
        "y": points_2d[:, 1].tolist(),
        "cluster_label": cluster_labels,
        "criterion_text": assignments_table.column("criterion_text").to_pylist(),
        "nct_id": assignments_table.column("nct_id").to_pylist(),
    }


def resolve_legend_order(cluster_labels: list[str]) -> list[str]:
    """Order `cluster_label` values for consistent legend display: ascending cluster id, `"Noise"` last.

    Args:
        cluster_labels: The `cluster_label` list from `build_plot_frame`.

    Returns:
        Every distinct value in `cluster_labels`, real clusters sorted
        ascending by their numeric id prefix, with `"Noise"` (if present)
        placed last.
    """
    distinct = set(cluster_labels)
    non_noise = sorted(
        (label for label in distinct if label != "Noise"), key=lambda label: int(label.split(":", 1)[0])
    )
    return non_noise + (["Noise"] if "Noise" in distinct else [])


def build_color_discrete_map(legend_order: list[str]) -> dict[str, str]:
    """Assign each legend entry a fixed color: pastel colorblind-safe hues for clusters, gray for noise.

    Args:
        legend_order: As returned by `resolve_legend_order`. Colors are
            assigned by position, cycling through `PASTEL_COLORBLIND_PALETTE`
            if there are more clusters than palette colors, so the same
            cluster id always gets the same color across repeated runs.

    Returns:
        A dict suitable for `plotly.express.scatter`'s `color_discrete_map`.
    """
    non_noise = [label for label in legend_order if label != "Noise"]
    color_map = {
        label: PASTEL_COLORBLIND_PALETTE[i % len(PASTEL_COLORBLIND_PALETTE)] for i, label in enumerate(non_noise)
    }
    if "Noise" in legend_order:
        color_map["Noise"] = NOISE_COLOR
    return color_map


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
        A `plotly.graph_objects.Figure` with cluster-colored points (pastel,
        colorblind-safe hues — see `PASTEL_COLORBLIND_PALETTE`), a legend
        naming each cluster by its own topic label, and a criterion-text
        hover tooltip.
    """
    seed_note = f"seed={seed}" if seed_explicit else f"seed={seed} (fallback default; not recorded by clustering)"
    title = f'"{label}" — {method} clusters (2D UMAP, {seed_note})'
    legend_order = resolve_legend_order(frame["cluster_label"])
    fig = px.scatter(
        frame,
        x="x",
        y="y",
        color="cluster_label",
        category_orders={"cluster_label": legend_order},
        color_discrete_map=build_color_discrete_map(legend_order),
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
        embeddings_path: Embeddings Parquet path to fit the fresh 2D
            projection from (see `resolve_points`). Required.

    Returns:
        The path the scatterplot was written to.

    Raises:
        ValueError: If `method` isn't a recognized key in
            `VISUALIZATION_SPECS`, or if `embeddings_path` was not given.
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

    points_2d = resolve_points(spec, assignments_table.num_rows, seed, embeddings_path=embeddings_path)
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
        choice, required metrics/assignments/output/embeddings path flags,
        and an optional seed override.
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
        required=True,
        help="Embeddings Parquet path to fit the fresh 2D projection from",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="If given, also append logs to this file (in addition to the console)",
    )
    return parser


def main() -> None:
    """CLI entry point: render and write a scatterplot for one method's clustering output."""
    args = build_arg_parser().parse_args()
    configure_logging(args.log_file)
    run(
        args.method,
        args.metrics_path,
        args.assignments_path,
        args.output_path,
        seed_override=args.seed,
        embeddings_path=args.embeddings_path,
    )


if __name__ == "__main__":
    main()
