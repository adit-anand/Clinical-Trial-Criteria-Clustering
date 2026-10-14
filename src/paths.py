"""Shared repo-root path constant for the pipeline modules under `src/`.

Every pipeline module (`src/preprocessing/`, `src/embeddings/`,
`src/clustering/`) needs the repository root to resolve its own default
input/output paths under `data/`. Centralizing that one derivation here
means a future change in how deep these modules live under `src/` only
needs to be fixed in one place, rather than in each module's own
`Path(__file__).resolve().parents[N]`.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
