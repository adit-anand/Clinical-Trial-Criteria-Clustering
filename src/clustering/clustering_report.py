"""Generate a consolidated clustering performance report across algorithms.

Reads the existing per-algorithm clustering metrics JSON artifacts already
written by `kmeans_clustering.py`, `umap_hdbscan_clustering.py`, and
`pacmap_hdbscan_clustering.py` (whichever of their metrics paths the caller
passes in) and renders one consolidated,
human-readable Markdown report comparing their clustering performance side
by side. This module never re-runs clustering — it only reads artifacts
that already exist on disk.

Each algorithm's presence in the report is driven entirely by one
`AlgorithmSpec` registry entry (see `ALGORITHM_SPECS` below); adding support
for a future additional clustering algorithm means adding one entry to that
list plus one small extractor function — no other function in this module
branches on which algorithm it's looking at.

An algorithm whose metrics path wasn't given, or whose metrics file is
missing/unparseable, never aborts the whole report: it's reflected as a
clearly labeled per-algorithm status (`"missing"` or `"malformed"`) so the
report still renders for whichever algorithm(s) are available.

The report is written to the path given via `--output-path`. Each run
overwrites that path wholesale.
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal

# Allow running this file directly (e.g. `uv run src/clustering/clustering_report.py`)
# as well as via `-m` (e.g. `uv run python -m src.clustering.clustering_report`).
# Direct execution leaves __package__ unset and never puts the repo root on
# sys.path, so the src.* imports below would otherwise fail with
# "ModuleNotFoundError: No module named 'src'"; -m already sets both up
# correctly, so this is a no-op in that case.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.paths import label_from_slug, slug_from_filename

logger = logging.getLogger(__name__)

# The three clustering-quality metrics every algorithm's own `build_metrics_dict`
# reports under the identical `metrics_valid`-guarded key names, letting the
# comparison table below iterate this list generically rather than branching
# per algorithm. `direction` records which way is "better", for display only.
SHARED_METRICS: list[tuple[str, str, Literal["higher", "lower"]]] = [
    ("silhouette_score", "Silhouette score", "higher"),
    ("davies_bouldin_index", "Davies-Bouldin index", "lower"),
    ("calinski_harabasz_index", "Calinski-Harabasz index", "higher"),
]

# Keys every algorithm's metrics JSON is expected to have, regardless of
# which algorithm produced it. Used only to detect a truncated/malformed/
# incompatible-schema file cheaply, not to validate nested structure.
REQUIRED_TOP_LEVEL_KEYS = ("n_samples", "cluster_sizes", "metrics", "metrics_valid", "notes", "clustered_at")


@dataclass(frozen=True)
class AlgorithmSpec:
    """Registry entry describing one clustering algorithm's report inputs.

    Attributes:
        key: Short identifier used to key per-algorithm results (e.g.
            "kmeans"); also used to derive this algorithm's CLI flag
            (`--{key.replace("_", "-")}-metrics-path`).
        display_name: Human-readable name shown in the report (e.g. "K-Means").
        metrics_suffix: This algorithm's metrics JSON filename suffix (e.g.
            "_trials_cluster_metrics.json") — used only to recover a
            human-readable label from whichever metrics path the caller gives.
        extra_fields_fn: Extracts this algorithm's non-shared metric fields
            from its parsed metrics dict as a list of (label, formatted value)
            pairs for display in that algorithm's own report subsection.
    """

    key: str
    display_name: str
    metrics_suffix: str
    extra_fields_fn: Callable[[dict], list[tuple[str, str]]]


@dataclass(frozen=True)
class LoadResult:
    """The outcome of trying to load one algorithm's metrics JSON for a condition.

    Attributes:
        status: `"ok"` if the file was found and parsed with all required
            top-level keys present; `"missing"` if no file exists at the
            resolved path; `"malformed"` if a file exists but couldn't be
            parsed as JSON or is missing required top-level keys (e.g.
            truncated, or written by an incompatible/older schema version).
        detail: Human-readable explanation, set for `"missing"`/`"malformed"`.
        data: The parsed metrics dict, set only when `status == "ok"`.
    """

    status: Literal["ok", "missing", "malformed"]
    detail: str | None = None
    data: dict | None = None


def _format_percent(count: int, total: int) -> str:
    """Format `count` out of `total` as "<count> / <total> (<pct>%)"."""
    pct = (count / total * 100) if total else 0.0
    return f"{count} / {total} ({pct:.1f}%)"


def kmeans_extra_fields(data: dict) -> list[tuple[str, str]]:
    """Extract k-means-only fields (no equivalent in other algorithms).

    Args:
        data: A parsed k-means metrics dict (as produced by
            `kmeans_clustering.KMeansClusterer.build_metrics_dict`).

    Returns:
        Label/formatted-value pairs for inertia, requested/actual cluster
        counts, and how `k` was selected. Uses `.get(...)` throughout so an
        unexpected/partial schema degrades to "N/A" fields rather than raising.
    """
    metrics = data.get("metrics") or {}
    inertia = metrics.get("inertia")
    k_selection = data.get("k_selection") or {}
    mode = k_selection.get("mode")
    if mode == "auto":
        k_range = f"auto (searched k={k_selection.get('k_min')}..{k_selection.get('k_max')})"
    elif mode == "fixed":
        k_range = "fixed"
    else:
        k_range = "N/A"

    return [
        ("Inertia", f"{inertia:.4f}" if inertia is not None else "N/A"),
        ("Clusters requested", str(data.get("n_clusters_requested", "N/A"))),
        ("Clusters found", str(data.get("n_clusters_actual", "N/A"))),
        ("k-selection mode", k_range),
    ]


def umap_hdbscan_extra_fields(data: dict) -> list[tuple[str, str]]:
    """Extract UMAP+HDBSCAN-only fields (no equivalent in other algorithms).

    Args:
        data: A parsed UMAP+HDBSCAN metrics dict (as produced by
            `umap_hdbscan_clustering.build_metrics_dict`).

    Returns:
        Label/formatted-value pairs for cluster/noise statistics plus the
        winning UMAP configuration, seed, and that seed's reproducibility
        (noise_fraction standard deviation across its repeats). Uses
        `.get(...)` throughout so an unexpected/partial schema degrades to
        "N/A" fields rather than raising.
    """
    noise_count = data.get("noise_count")
    n_samples = data.get("n_samples")
    noise_display = _format_percent(noise_count, n_samples) if noise_count is not None and n_samples is not None else "N/A"

    winning_config = data.get("winning_umap_config") or {}
    reproducibility_std = data.get("winning_seed_noise_fraction_std")

    return [
        ("Clusters found", str(data.get("n_clusters_actual", "N/A"))),
        ("Noise points", noise_display),
        ("Winning UMAP config", str(winning_config) if winning_config else "N/A"),
        ("Winning seed", str(data.get("winning_seed", "N/A"))),
        ("Reproducibility (noise fraction std)", f"{reproducibility_std:.4f}" if reproducibility_std is not None else "N/A"),
    ]


def pacmap_hdbscan_extra_fields(data: dict) -> list[tuple[str, str]]:
    """Extract PaCMAP+HDBSCAN-only fields (no equivalent in other algorithms).

    Args:
        data: A parsed PaCMAP+HDBSCAN metrics dict (as produced by
            `pacmap_hdbscan_clustering.build_metrics_dict`).

    Returns:
        Label/formatted-value pairs for cluster/noise statistics plus the
        winning PaCMAP configuration, seed, and that seed's reproducibility
        (noise_fraction standard deviation across its repeats). Uses
        `.get(...)` throughout so an unexpected/partial schema degrades to
        "N/A" fields rather than raising.
    """
    noise_count = data.get("noise_count")
    n_samples = data.get("n_samples")
    noise_display = _format_percent(noise_count, n_samples) if noise_count is not None and n_samples is not None else "N/A"

    winning_config = data.get("winning_pacmap_config") or {}
    reproducibility_std = data.get("winning_seed_noise_fraction_std")

    return [
        ("Clusters found", str(data.get("n_clusters_actual", "N/A"))),
        ("Noise points", noise_display),
        ("Winning PaCMAP config", str(winning_config) if winning_config else "N/A"),
        ("Winning seed", str(data.get("winning_seed", "N/A"))),
        ("Reproducibility (noise fraction std)", f"{reproducibility_std:.4f}" if reproducibility_std is not None else "N/A"),
    ]


ALGORITHM_SPECS: list[AlgorithmSpec] = [
    AlgorithmSpec(
        key="kmeans",
        display_name="K-Means",
        metrics_suffix="_trials_cluster_metrics.json",
        extra_fields_fn=kmeans_extra_fields,
    ),
    AlgorithmSpec(
        key="umap_hdbscan",
        display_name="UMAP + HDBSCAN",
        metrics_suffix="_trials_umap_hdbscan_cluster_metrics.json",
        extra_fields_fn=umap_hdbscan_extra_fields,
    ),
    AlgorithmSpec(
        key="pacmap_hdbscan",
        display_name="PaCMAP + HDBSCAN",
        metrics_suffix="_trials_pacmap_hdbscan_cluster_metrics.json",
        extra_fields_fn=pacmap_hdbscan_extra_fields,
    ),
]


def metrics_path_flag(spec: AlgorithmSpec) -> str:
    """Derive this algorithm's CLI flag name, e.g. "--umap-hdbscan-metrics-path"."""
    return f"--{spec.key.replace('_', '-')}-metrics-path"


def resolve_report_label(metrics_paths: dict[str, Path | None]) -> str:
    """Derive a human-readable label for the report title/header.

    Uses the first algorithm (in `ALGORITHM_SPECS` order) whose metrics path
    was given, stripping that algorithm's `metrics_suffix` from the
    filename to recover the original slug.

    Args:
        metrics_paths: Maps each `AlgorithmSpec.key` to the metrics path
            given for it, or `None` if omitted.

    Returns:
        The recovered label, or `"unknown"` if no metrics path was given at all.
    """
    for spec in ALGORITHM_SPECS:
        path = metrics_paths.get(spec.key)
        if path is not None:
            return label_from_slug(slug_from_filename(path.name, spec.metrics_suffix))
    return "unknown"


def load_metrics(metrics_path: Path) -> LoadResult:
    """Load and sanity-check one algorithm's metrics JSON.

    Args:
        metrics_path: Path to this algorithm's metrics JSON, as given by the caller.

    Returns:
        A `LoadResult` with `status="missing"` if `metrics_path` doesn't
        exist, `status="malformed"` if it exists but can't be parsed as JSON
        or is missing one of `REQUIRED_TOP_LEVEL_KEYS` (covering truncated
        files and incompatible/older schema versions), or `status="ok"` with
        the parsed dict otherwise.
    """
    if not metrics_path.exists():
        return LoadResult(status="missing", detail=f"No metrics file found at {metrics_path}")

    try:
        with open(metrics_path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        return LoadResult(status="malformed", detail=f"Could not parse {metrics_path} as JSON: {exc}")

    missing_keys = [key for key in REQUIRED_TOP_LEVEL_KEYS if key not in data]
    if missing_keys:
        return LoadResult(
            status="malformed",
            detail=(
                f"{metrics_path} is missing required key(s): {', '.join(missing_keys)} "
                "(file may be truncated or from an incompatible/older schema)"
            ),
        )

    return LoadResult(status="ok", data=data)


def load_all(metrics_paths: dict[str, Path | None]) -> dict[str, LoadResult]:
    """Load every registered algorithm's metrics.

    Args:
        metrics_paths: Maps each `AlgorithmSpec.key` to the metrics path
            given for it on the CLI, or `None` if that algorithm's flag was
            omitted (treated as `"missing"` without attempting a read).

    Returns:
        A dict mapping each `AlgorithmSpec.key` to its `LoadResult`.
    """
    results: dict[str, LoadResult] = {}
    for spec in ALGORITHM_SPECS:
        metrics_path = metrics_paths.get(spec.key)
        if metrics_path is None:
            result = LoadResult(status="missing", detail=f"No {metrics_path_flag(spec)} given")
        else:
            result = load_metrics(metrics_path)
        logger.info(
            "load_all: %s metrics -> status=%s (%s)",
            spec.display_name, result.status, metrics_path,
        )
        results[spec.key] = result
    return results


def compute_best_algorithm(loaded: dict[str, LoadResult]) -> tuple[str, float] | None:
    """Pick the algorithm with the highest valid silhouette score.

    Silhouette score is the only metric present under an identical key in
    every algorithm's schema, is bounded in [-1, 1], and is consistently
    "higher is better" — unlike Davies-Bouldin (lower is better) or
    algorithm-specific values (inertia, noise fraction) that aren't
    comparable across algorithms at all. This is the only cross-algorithm
    derived statistic this report computes.

    Args:
        loaded: Per-algorithm `LoadResult`s, as returned by `load_all`.

    Returns:
        `(display_name, silhouette_score)` for the highest-scoring algorithm
        among those with `status == "ok"` and `metrics_valid is True`, or
        `None` if no algorithm qualifies.
    """
    best: tuple[str, float] | None = None
    for spec in ALGORITHM_SPECS:
        result = loaded[spec.key]
        if result.status != "ok" or not result.data.get("metrics_valid"):
            continue
        score = (result.data.get("metrics") or {}).get("silhouette_score")
        if score is None:
            continue
        if best is None or score > best[1]:
            best = (spec.display_name, score)
    return best


def _status_label(result: LoadResult) -> str:
    """Human-readable one-word(ish) status label for the summary table."""
    if result.status == "missing":
        return "Not yet run"
    if result.status == "malformed":
        return "Metrics file unreadable"
    if result.data.get("metrics_valid"):
        return "Available"
    return "Available (metrics invalid)"


def render_summary_section(loaded: dict[str, LoadResult], best: tuple[str, float] | None) -> str:
    """Render the top-level "## Summary" section.

    Args:
        loaded: Per-algorithm `LoadResult`s, as returned by `load_all`.
        best: As returned by `compute_best_algorithm`.

    Returns:
        A Markdown fragment: an algorithm-status table followed by the
        best-by-silhouette line (or an explicit "none qualify" line).
    """
    lines = ["## Summary", "", "| Algorithm | Status |", "| --- | --- |"]
    for spec in ALGORITHM_SPECS:
        lines.append(f"| {spec.display_name} | {_status_label(loaded[spec.key])} |")
    lines.append("")
    if best is not None:
        name, score = best
        lines.append(f"**Best by silhouette score:** {name} ({score:.4f})")
    else:
        lines.append("**Best by silhouette score:** No algorithm produced a valid silhouette score for comparison.")
    return "\n".join(lines)


def render_shared_metrics_table(loaded: dict[str, LoadResult]) -> str:
    """Render the "## Shared Metrics Comparison" section.

    Only algorithms with `status == "ok"` get a column — an algorithm that
    never ran has nothing to compare and is omitted entirely from this table
    (its status is already stated in the Summary and its own subsection);
    an algorithm that ran but has `metrics_valid=False` still gets a column,
    showing `N/A` for each guarded metric, since it did produce a comparable
    (if unusable) result.

    Args:
        loaded: Per-algorithm `LoadResult`s, as returned by `load_all`.

    Returns:
        A Markdown fragment, or a one-line note if no algorithm is "ok".
    """
    ok_specs = [spec for spec in ALGORITHM_SPECS if loaded[spec.key].status == "ok"]
    lines = ["## Shared Metrics Comparison", ""]
    if not ok_specs:
        lines.append("No algorithm has produced metrics yet.")
        return "\n".join(lines)

    header = ["Metric", "Direction"] + [spec.display_name for spec in ok_specs]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for metric_key, label, direction in SHARED_METRICS:
        direction_label = "higher is better" if direction == "higher" else "lower is better"
        row = [label, direction_label]
        for spec in ok_specs:
            value = (loaded[spec.key].data.get("metrics") or {}).get(metric_key)
            row.append(f"{value:.4f}" if value is not None else "N/A")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_algorithm_section(spec: AlgorithmSpec, result: LoadResult) -> str:
    """Render one `### <Display Name>` subsection for a single algorithm.

    Args:
        spec: The algorithm's registry entry.
        result: Its `LoadResult`, as returned by `load_all`.

    Returns:
        A Markdown fragment. Always present regardless of `result.status`,
        so a reader can see exactly why an algorithm's results are
        unavailable rather than finding it silently omitted.
    """
    lines = [f"### {spec.display_name}", ""]
    if result.status == "missing":
        lines.append(f"Not yet run ({result.detail}).")
        return "\n".join(lines)
    if result.status == "malformed":
        lines.append(f"Metrics file could not be read: {result.detail}")
        return "\n".join(lines)

    data = result.data
    metrics_valid = bool(data.get("metrics_valid"))
    lines.append(f"- **Metrics valid:** {metrics_valid}")
    for label, value in spec.extra_fields_fn(data):
        lines.append(f"- **{label}:** {value}")

    notes = data.get("notes") or []
    if notes:
        lines.append("- **Notes:**")
        for note in notes:
            lines.append(f"  - {note}")
    return "\n".join(lines)


def render_markdown(label: str, loaded: dict[str, LoadResult], generated_at: datetime) -> str:
    """Assemble the full Markdown report.

    Args:
        label: Human-readable label for the report title, as returned by
            `resolve_report_label`.
        loaded: Per-algorithm `LoadResult`s, as returned by `load_all`.
        generated_at: Wall-clock time this report was generated.

    Returns:
        The complete report as a Markdown string.
    """
    best = compute_best_algorithm(loaded)
    algorithm_names = ", ".join(spec.display_name for spec in ALGORITHM_SPECS)

    sections = [
        f"# Clustering Performance Report: {label}",
        "",
        f"- **Generated at:** {generated_at.isoformat()}",
        f"- **Algorithms considered:** {algorithm_names}",
        "",
        render_summary_section(loaded, best),
        "",
        render_shared_metrics_table(loaded),
        "",
    ]
    for spec in ALGORITHM_SPECS:
        sections.append(render_algorithm_section(spec, loaded[spec.key]))
        sections.append("")
    return "\n".join(sections).rstrip() + "\n"


def generate_report(
    output_path: Path,
    metrics_paths: dict[str, Path | None],
    print_to_console: bool = False,
    generated_at: datetime | None = None,
) -> Path:
    """Generate and write a clustering performance report.

    Args:
        output_path: Where to write the rendered Markdown report.
        metrics_paths: Maps each `AlgorithmSpec.key` to the metrics path
            given for it, or `None` if omitted (see `load_all`).
        print_to_console: If True, also print the rendered Markdown to stdout.
        generated_at: Wall-clock time to record as this report's generation
            time. Defaults to `datetime.now(timezone.utc)`.

    Returns:
        The path the report was written to.
    """
    if generated_at is None:
        generated_at = datetime.now(timezone.utc)

    label = resolve_report_label(metrics_paths)
    loaded = load_all(metrics_paths)
    markdown = render_markdown(label, loaded, generated_at)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")

    n_available = sum(1 for result in loaded.values() if result.status == "ok")
    logger.info(
        "generate_report: wrote report for %r (%d of %d algorithm(s) available) -> %s",
        label, n_available, len(ALGORITHM_SPECS), output_path,
    )

    if print_to_console:
        print(markdown)

    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for this module.

    Returns:
        An ``argparse.ArgumentParser`` accepting a required ``--output-path``
        flag, one optional ``--<algorithm>-metrics-path`` flag per registered
        algorithm (see `ALGORITHM_SPECS`), and a console-printing option.
    """
    parser = argparse.ArgumentParser(
        description="Generate a consolidated clustering performance report across algorithms."
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Output Markdown report path",
    )
    for spec in ALGORITHM_SPECS:
        parser.add_argument(
            metrics_path_flag(spec),
            dest=f"{spec.key}_metrics_path",
            type=Path,
            default=None,
            help=f"{spec.display_name} metrics JSON path (omit if this algorithm hasn't been run)",
        )
    parser.add_argument(
        "--print",
        dest="print_to_console",
        action="store_true",
        help="Also print the generated report to stdout",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = build_arg_parser().parse_args()
    metrics_paths = {spec.key: getattr(args, f"{spec.key}_metrics_path") for spec in ALGORITHM_SPECS}
    generate_report(
        args.output_path,
        metrics_paths,
        print_to_console=args.print_to_console,
    )


if __name__ == "__main__":
    main()
