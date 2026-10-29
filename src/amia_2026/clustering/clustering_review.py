"""Combined report+scatterplot CLI step: generate the report, then visualize the winner.

`clustering_report.py` and `clustering_scatterplot.py` are normally chained
by hand: a person reads the report, decides by eye which methodology
"won," then re-invokes the scatterplot CLI with that methodology's
`--method`/`--metrics-path`/`--assignments-path` re-typed in. This module
automates that hand-off: it generates the report, then picks whichever
methodology has the higher composite objective score (via
`clustering_report.compute_best_algorithm_by_composite_score`, ties going
to PaCMAP+HDBSCAN) and renders its scatterplot — resolving every path
`clustering_scatterplot.run` needs from the metrics JSON the report step
already read, with no new path flags.

The report is always written, even when nothing qualifies for a
scatterplot — a missing/ambiguous "winner" is never a reason to fail the
whole command. Only the scatterplot-rendering call degrades gracefully
(log a clear reason, no scatterplot produced); a genuine defect anywhere
else still surfaces as a crash.

The two existing standalone CLIs (`clustering_report.py`,
`clustering_scatterplot.py`) are untouched and keep working exactly as
today — this module is purely additive, reusing both as-is.
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

from amia_2026.logging_utils import configure_logging
from amia_2026.paths import REPO_ROOT
from amia_2026.clustering import clustering_report, clustering_scatterplot

logger = logging.getLogger(__name__)


def _resolve_repo_relative_path(value: str) -> Path:
    """Inverse of `clustering_io.relative_to_repo_root`.

    Args:
        value: A path string as recorded in a metrics JSON, either
            `REPO_ROOT`-relative or absolute (see `relative_to_repo_root`).

    Returns:
        `value` resolved to an absolute `Path`, joined onto `REPO_ROOT` if
        it wasn't already absolute.
    """
    candidate = Path(value)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def run(
    report_path: Path,
    scatterplot_path: Path,
    metrics_paths: dict[str, Path | None],
    print_to_console: bool = False,
    generated_at: datetime | None = None,
) -> tuple[Path, Path | None]:
    """Generate the clustering report, then render a scatterplot for its composite-score winner.

    Args:
        report_path: Where to write the rendered Markdown report.
        scatterplot_path: Where to write the HTML scatterplot, if one is produced.
        metrics_paths: Maps each `AlgorithmSpec.key` to the metrics path
            given for it, or `None` if omitted (see `clustering_report.load_all`).
        print_to_console: If True, also print the rendered report Markdown to stdout.
        generated_at: Wall-clock time to record as the report's generation
            time. Defaults to `datetime.now(timezone.utc)` (see
            `clustering_report.generate_report`).

    Returns:
        `(report_path, scatterplot_path_or_None)` — the second element is
        `None` whenever no scatterplot was produced, either because no
        methodology qualified for selection or because rendering its
        scatterplot failed.
    """
    clustering_report.generate_report(report_path, metrics_paths, print_to_console, generated_at)

    loaded = clustering_report.load_all(metrics_paths)
    best = clustering_report.compute_best_algorithm_by_composite_score(loaded)
    if best is None:
        logger.warning(
            "run: no methodology produced a usable composite score to select a scatterplot from; "
            "no scatterplot was generated"
        )
        return report_path, None

    spec, score = best
    data = loaded[spec.key].data

    input_path = data.get("input_path")
    embeddings_path = _resolve_repo_relative_path(input_path) if input_path else None
    assignments_output_path = data.get("assignments_output_path")
    assignments_path = _resolve_repo_relative_path(assignments_output_path) if assignments_output_path else None
    reduced_embeddings_output_path = data.get("reduced_embeddings_output_path")
    reduced_embeddings_path = (
        _resolve_repo_relative_path(reduced_embeddings_output_path) if reduced_embeddings_output_path else None
    )

    try:
        clustering_scatterplot.run(
            spec.key,
            metrics_paths[spec.key],
            assignments_path,
            scatterplot_path,
            embeddings_path=embeddings_path,
            reduced_embeddings_path=reduced_embeddings_path,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.warning(
            "run: could not render a scatterplot for %s (composite score %.4f): %s; "
            "no scatterplot was generated",
            spec.display_name, score, exc,
        )
        return report_path, None

    logger.info(
        "run: rendered scatterplot for %s (composite score %.4f) -> %s",
        spec.display_name, score, scatterplot_path,
    )
    return report_path, scatterplot_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for this module.

    Returns:
        An ``argparse.ArgumentParser`` accepting one optional
        ``--<algorithm>-metrics-path`` flag per registered algorithm (see
        `clustering_report.ALGORITHM_SPECS`), a required ``--report-path``
        and ``--scatterplot-path``, and a console-printing option.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate the clustering performance report, then render a scatterplot for whichever "
            "methodology has the higher composite objective score (ties favor PaCMAP+HDBSCAN)."
        )
    )
    for spec in clustering_report.ALGORITHM_SPECS:
        parser.add_argument(
            clustering_report.metrics_path_flag(spec),
            dest=f"{spec.key}_metrics_path",
            type=Path,
            default=None,
            help=f"{spec.display_name} metrics JSON path (omit if this algorithm hasn't been run)",
        )
    parser.add_argument(
        "--report-path",
        type=Path,
        required=True,
        help="Output Markdown report path",
    )
    parser.add_argument(
        "--scatterplot-path",
        type=Path,
        required=True,
        help="Output HTML scatterplot path (written only if a methodology qualifies for selection)",
    )
    parser.add_argument(
        "--print",
        dest="print_to_console",
        action="store_true",
        help="Also print the generated report to stdout",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="If given, also append logs to this file (in addition to the console)",
    )
    return parser


def main() -> None:
    """CLI entry point: generate the report, then render a scatterplot for its composite-score winner."""
    args = build_arg_parser().parse_args()
    configure_logging(args.log_file)
    metrics_paths = {
        spec.key: getattr(args, f"{spec.key}_metrics_path") for spec in clustering_report.ALGORITHM_SPECS
    }
    run(
        args.report_path,
        args.scatterplot_path,
        metrics_paths,
        print_to_console=args.print_to_console,
    )


if __name__ == "__main__":
    main()
