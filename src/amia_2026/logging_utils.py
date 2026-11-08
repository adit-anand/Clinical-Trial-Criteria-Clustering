"""Shared logging setup for the `amia_2026` pipeline CLI entry points.

Every pipeline module's `main()` previously called `logging.basicConfig(level=logging.INFO)`
directly, which only ever logs to the console. Centralizing that call here
means a future change to the shared logging format only needs to be made
in one place.
"""

import logging
import resource
from pathlib import Path


def log_memory_usage(log: logging.Logger, context: str, level: int = logging.INFO) -> None:
    """Log this process's peak resident-set size so far, tagged with `context`.

    A process killed outright by the OS (e.g. the Linux OOM killer sending
    SIGKILL during a large reducer/HDBSCAN/BERTopic fit) never raises a
    Python exception, so the usual exception-logging path records nothing
    about it. Calling this at memory-sensitive checkpoints leaves a trail of
    readings in the log, so the most recent one before such a kill narrows
    down where it happened. `ru_maxrss` is reported in kilobytes on Linux.

    Args:
        log: The caller's own logger.
        context: Short label identifying the checkpoint (e.g. a function name
            and what it just did/is about to do).
        level: Logging level to log at; `logging.ERROR` when called right
            after catching a `MemoryError`, `logging.INFO` for routine
            breadcrumbs.
    """
    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    log.log(level, "%s: peak RSS so far = %.1f MB", context, peak_rss_kb / 1024)


def configure_logging(log_file: Path | None = None) -> None:
    """Configure root logging to log to the console and, optionally, also to a file.

    Args:
        log_file: If given, log records are also appended to this path (its
            parent directories are created as needed), in addition to the
            console. If omitted, logging is console-only, as it was before
            this flag existed.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )
