"""Backward-compatible alias for `hdbscan_clustering.py`.

`hdbscan_clustering.py` now performs the hyperparameter sweep this module
used to perform on its own — every invocation of that module sweeps
(a single value per flag is simply a sweep of size one), so there is no
longer a distinct "sweep" step. This module keeps
``uv run python -m src.clustering.hdbscan_sweep ...`` working for existing
callers without duplicating any logic: it just re-exports
`hdbscan_clustering.py`'s CLI and functions under this module's name.

Prefer invoking `hdbscan_clustering.py` directly for new usage.
"""

import sys
from pathlib import Path

# Allow running this file directly (e.g. `uv run src/clustering/hdbscan_sweep.py`)
# as well as via `-m` (e.g. `uv run python -m src.clustering.hdbscan_sweep`).
# Direct execution leaves __package__ unset and never puts the repo root on
# sys.path, so the src.* import below would otherwise fail with
# "ModuleNotFoundError: No module named 'src'"; -m already sets both up
# correctly, so this is a no-op in that case.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.clustering.hdbscan_clustering import (  # noqa: F401
    build_arg_parser,
    main,
    parse_float_list,
    parse_int_list,
    run_sweep,
)

if __name__ == "__main__":
    main()
