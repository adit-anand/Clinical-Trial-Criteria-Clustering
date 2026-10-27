"""Shared, algorithm-agnostic I/O for the clustering pipeline stages.

`kmeans_clustering.py`, `pacmap_hdbscan_clustering.py`, and
`umap_hdbscan_clustering.py` all read the same embeddings Parquet input
format and write cluster assignments plus a metrics JSON in a similar
shape. This module holds the logic that is genuinely common to all of them
— reading/filtering/validating the input embeddings table and writing
output artifacts — so no algorithm-specific module re-implements it.

Nothing here branches on "which algorithm am I"; algorithm-specific concerns
(fitting, per-algorithm metrics, per-algorithm output columns) stay in each
algorithm's own module.
"""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.paths import REPO_ROOT


def read_embeddings_parquet(input_path: Path) -> pa.Table:
    """Read the embeddings Parquet file produced by `criteria_embedding.py`.

    Args:
        input_path: Path to a Parquet file written by
            `CriteriaEmbedder.write_embeddings_parquet`.

    Returns:
        The full Parquet table, including rows with failed embeddings.

    Raises:
        FileNotFoundError: If `input_path` does not exist.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input embeddings Parquet not found: {input_path}")
    return pq.read_table(input_path)


def filter_successful_rows(table: pa.Table) -> pa.Table:
    """Filter a table down to rows where `embedding_success` is True.

    Args:
        table: A table as returned by `read_embeddings_parquet`.

    Returns:
        The subset of `table` with `embedding_success == True`.

    Raises:
        ValueError: If the result has zero rows, covering both an empty
            input table and an input where every row failed embedding —
            there is nothing to cluster either way.
    """
    filtered = table.filter(pc.field("embedding_success"))
    if filtered.num_rows == 0:
        raise ValueError(
            f"No successfully embedded rows found in input "
            f"(0 of {table.num_rows} row(s) had embedding_success=True); "
            "refusing to cluster an empty dataset"
        )
    return filtered


VALID_CRITERION_TYPES = ("inclusion", "exclusion")


def filter_by_criterion_type(table: pa.Table, criterion_type: str) -> pa.Table:
    """Filter a table down to rows eligible for one side of an inclusion/exclusion split.

    Rows whose `criterion_type` is `"both"` (unstructured eligibility text
    with no inclusion/exclusion headers to split on — see
    `src/preprocessing/criteria_parser.py`) are eligible for *either* side,
    since they cannot be confidently assigned to only one: `criterion_type=
    "inclusion"` selects rows whose column value is `"inclusion"` or
    `"both"`; `criterion_type="exclusion"` selects `"exclusion"` or `"both"`.

    Args:
        table: A table as returned by `read_embeddings_parquet`.
        criterion_type: `"inclusion"` or `"exclusion"`.

    Returns:
        The eligible subset of `table`.

    Raises:
        ValueError: If `criterion_type` isn't one of `VALID_CRITERION_TYPES`,
            or if the result has zero rows — there is nothing to cluster
            either way.
    """
    if criterion_type not in VALID_CRITERION_TYPES:
        raise ValueError(f"Unknown criterion_type {criterion_type!r}; expected one of {VALID_CRITERION_TYPES}")
    filtered = table.filter(pc.field("criterion_type").isin([criterion_type, "both"]))
    if filtered.num_rows == 0:
        raise ValueError(
            f"No rows with criterion_type in {{{criterion_type!r}, 'both'}} found in input "
            f"(0 of {table.num_rows} row(s) matched); refusing to cluster an empty dataset"
        )
    return filtered


def validate_embedding_dimensions(table: pa.Table) -> int:
    """Confirm every row's `embedding` vector has the same length.

    Args:
        table: A table of successfully-embedded rows (e.g. the output of
            `filter_successful_rows`).

    Returns:
        The common embedding dimensionality.

    Raises:
        ValueError: If more than one distinct embedding length is found,
            naming each distinct dimension and an example `nct_id` for it,
            so a mixed-backend/mixed-model input fails with an actionable
            message rather than crashing inside numpy/sklearn.
    """
    nct_ids = table.column("nct_id").to_pylist()
    embeddings = table.column("embedding").to_pylist()

    dim_examples: dict[int, str] = {}
    for nct_id, embedding in zip(nct_ids, embeddings):
        dim_examples.setdefault(len(embedding), nct_id)

    if len(dim_examples) > 1:
        details = ", ".join(
            f"dim={dim} (e.g. nct_id={example!r})" for dim, example in sorted(dim_examples.items())
        )
        raise ValueError(f"Embeddings have inconsistent dimensionality, cannot cluster together: {details}")
    return next(iter(dim_examples))


def extract_embedding_matrix(table: pa.Table) -> np.ndarray:
    """Stack a table's `embedding` column into a 2D float32 matrix.

    Args:
        table: A table of successfully-embedded rows with uniform embedding
            dimensionality.

    Returns:
        An array of shape `(n_samples, embedding_dim)`.

    Raises:
        ValueError: Propagated from `validate_embedding_dimensions` if
            dimensions are not uniform.
    """
    validate_embedding_dimensions(table)
    return np.array(table.column("embedding").to_pylist(), dtype=np.float32)


def extract_traceability_columns(table: pa.Table) -> dict[str, list]:
    """Pull the columns common to both algorithms' assignment output, as parallel lists.

    Args:
        table: A table of successfully-embedded rows (row order defines the
            order every other per-row array must follow).

    Returns:
        A dict with keys `nct_id`, `study_first_posted_date`,
        `criterion_text`, `criterion_type`, `embedding_backend`,
        `embedding_model`, each a list of length `table.num_rows`. Each
        algorithm's own `build_assignment_rows` zips these with its own
        per-row cluster output (e.g. `cluster_id`, and for HDBSCAN also
        `cluster_probability`).
    """
    return {
        "nct_id": table.column("nct_id").to_pylist(),
        "study_first_posted_date": table.column("study_first_posted_date").to_pylist(),
        "criterion_text": table.column("criterion_text").to_pylist(),
        "criterion_type": table.column("criterion_type").to_pylist(),
        "embedding_backend": table.column("embedding_backend").to_pylist(),
        "embedding_model": table.column("embedding_model").to_pylist(),
    }


def relative_to_repo_root(path: Path) -> str:
    """Render `path` as a POSIX-style string relative to `REPO_ROOT` if possible.

    Falls back to the absolute path string when `path` doesn't live under
    `REPO_ROOT` (e.g. an explicit `--input-path` override pointing elsewhere),
    so metrics JSON generation never fails on path formatting alone.
    """
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def compute_cluster_sizes(labels: np.ndarray) -> dict[str, int]:
    """Count how many points fall in each distinct cluster label.

    Args:
        labels: One cluster id per point (may include `-1` for HDBSCAN noise).

    Returns:
        A dict mapping the string form of each distinct label to its count,
        sorted by the label's integer value (so `-1`, if present, sorts first).
    """
    cluster_sizes: dict[str, int] = {}
    for label in labels:
        key = str(int(label))
        cluster_sizes[key] = cluster_sizes.get(key, 0) + 1
    return dict(sorted(cluster_sizes.items(), key=lambda kv: int(kv[0])))


def build_base_metrics_dict(
    label: str,
    criterion_type: str,
    input_path: Path,
    assignments_output_path: Path,
    n_samples: int,
    n_samples_excluded: int,
    embedding_dim: int,
    embedding_backend: str | None,
    embedding_model: str | None,
    cluster_sizes: dict[str, int],
    clustered_at,
) -> dict:
    """Build the metrics-JSON envelope fields common to both algorithms.

    Each algorithm's own `build_metrics_dict` merges this dict with its own
    algorithm-specific keys (e.g. `kmeans_params`/`k_selection` for k-means,
    `hdbscan_params`/`noise_count`/`noise_fraction` for HDBSCAN).

    Args:
        label: Human-readable label for this run, derived from `input_path`'s
            filename (see `src.paths.slug_from_filename`/`label_from_slug`),
            recorded here only for traceability — no longer used to resolve
            any path.
        criterion_type: Which side of the inclusion/exclusion split this run
            covers (see `filter_by_criterion_type`), recorded as its own
            field so downstream consumers don't need to parse `label`.
        input_path: The embeddings Parquet path that was read.
        assignments_output_path: Where cluster assignments were written.
        n_samples: Number of successfully-embedded rows clustered.
        n_samples_excluded: Number of input rows dropped for having failed
            embeddings.
        embedding_dim: Common embedding dimensionality.
        embedding_backend: Backend recorded on the clustered rows.
        embedding_model: Model recorded on the clustered rows.
        cluster_sizes: As produced by `compute_cluster_sizes`.
        clustered_at: Wall-clock time (a `datetime`) this clustering run
            completed.

    Returns:
        A dict with the envelope fields shared by both metrics JSON schemas.
    """
    return {
        "condition": label,
        "criterion_type": criterion_type,
        "input_path": relative_to_repo_root(input_path),
        "assignments_output_path": relative_to_repo_root(assignments_output_path),
        "n_samples": n_samples,
        "n_samples_excluded_failed_embeddings": n_samples_excluded,
        "embedding_dim": embedding_dim,
        "embedding_backend": embedding_backend,
        "embedding_model": embedding_model,
        "cluster_sizes": cluster_sizes,
        "clustered_at": clustered_at.isoformat(),
    }


def write_parquet(rows: list[dict], schema: pa.Schema, output_path: Path) -> None:
    """Write row dicts to a Parquet file against a given schema, overwriting any existing file.

    Args:
        rows: Output row dicts, one per `schema`'s columns.
        schema: The `pa.Schema` to write against.
        output_path: Destination Parquet file path. Parent directories are
            created if missing.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = {name: [row[name] for row in rows] for name in schema.names}
    table = pa.Table.from_pydict(columns, schema=schema)
    pq.write_table(table, output_path)


def write_json(payload: dict, output_path: Path) -> None:
    """Write a JSON-serializable dict to a file, overwriting any existing file.

    Args:
        payload: The dict to serialize.
        output_path: Destination JSON file path. Parent directories are
            created if missing.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
