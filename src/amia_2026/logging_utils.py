"""Shared logging setup for the `amia_2026` pipeline CLI entry points.

Every pipeline module's `main()` previously called `logging.basicConfig(level=logging.INFO)`
directly, which only ever logs to the console. Centralizing that call here
means a future change to the shared logging format only needs to be made
in one place.
"""

import logging
from pathlib import Path


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
