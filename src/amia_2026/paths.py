"""Shared repo-root path constant and filename helpers for the `amia_2026` package.

Every pipeline module (`amia_2026.preprocessing`, `amia_2026.embeddings`,
`amia_2026.clustering`) needs the repository root to resolve its own default
input/output paths under `data/`. Centralizing that one derivation here
means a future change in how deep these modules live under `src/amia_2026/`
only needs to be fixed in one place, rather than in each module's own
`Path(__file__).resolve().parents[N]`.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def slug_from_filename(filename: str, suffix: str) -> str:
    """Recover a filesystem-safe slug from a filename by stripping a known suffix.

    Every pipeline stage's CLI now takes its input/output paths explicitly
    (there is no more positional ``condition`` argument to derive them
    from), but a human-readable label is still useful for titles, log lines,
    and metrics-JSON provenance fields. This inverts the naming convention
    those files were written with (``<slug>_trials_...``) so callers can
    recover that slug from whichever path they were actually given.

    Args:
        filename: A file's base name, e.g. "breast_cancer_trials_embeddings.parquet".
        suffix: The expected suffix for this file type, e.g.
            "_trials_embeddings.parquet". If `filename` doesn't end with it
            (an unexpected name), falls back to the name with its extension
            stripped.

    Returns:
        The slug, e.g. "breast_cancer".
    """
    if filename.endswith(suffix):
        return filename[: -len(suffix)]
    return Path(filename).stem


def label_from_slug(slug: str) -> str:
    """Render a slug as a human-readable label, e.g. "breast_cancer" -> "breast cancer"."""
    return slug.replace("_", " ")
