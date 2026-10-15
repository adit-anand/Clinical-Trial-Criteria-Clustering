"""Embed per-condition eligibility criteria CSVs into per-condition Parquet files.

Reads a criteria CSV produced by `src/preprocessing/criteria_retrieval.py`
(``data/eligibility_criteria/<condition>_trials_eligibility.csv``, with
columns ``nct_id``, ``study_first_posted_date``, ``criterion_text``,
``criterion_type``), embeds each row's ``criterion_text`` via a configurable
`EmbeddingModel` backend (see `EmbeddingModel.py`), and writes the result as
one row per input row to
``data/embeddings/<condition>_trials_embeddings.parquet``.

Output schema (one row per input criterion row, including failed ones, so a
downstream clustering step can load the file directly without guesswork):

    nct_id                  string
    study_first_posted_date string
    criterion_text          string
    criterion_type          string
    embedding                list<float32>, null when embedding_success is False
    embedding_success       bool
    embedding_dim           int32, null on failure
    embedding_backend       string ("vllm" or "huggingface")
    embedding_model         string (e.g. "gpt-oss-20b")
    embedded_at             timestamp("us", tz="UTC")

Re-running this step on the same condition overwrites the existing output
file wholesale — there is no append/skip-if-exists/merge behavior.
"""

import argparse
import csv
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# Allow running this file directly (e.g. `uv run src/embeddings/criteria_embedding.py`)
# as well as via `-m` (e.g. `uv run python -m src.embeddings.criteria_embedding`).
# Direct execution leaves __package__ unset and never puts the repo root on
# sys.path, so the src.* imports below would otherwise fail with
# "ModuleNotFoundError: No module named 'src'"; -m already sets both up
# correctly, so this is a no-op in that case.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.paths import REPO_ROOT
from src.preprocessing.criteria_retrieval import EligibilityCriteriaFetcher
from src.embeddings.embedding_model import EmbeddingModel, build_embedding_model

INPUT_DIR = REPO_ROOT / "data" / "eligibility_criteria"
OUTPUT_DIR = REPO_ROOT / "data" / "embeddings"

logger = logging.getLogger(__name__)

PARQUET_SCHEMA = pa.schema(
    [
        ("nct_id", pa.string()),
        ("study_first_posted_date", pa.string()),
        ("criterion_text", pa.string()),
        ("criterion_type", pa.string()),
        ("embedding", pa.list_(pa.float32())),
        ("embedding_success", pa.bool_()),
        ("embedding_dim", pa.int32()),
        ("embedding_backend", pa.string()),
        ("embedding_model", pa.string()),
        ("embedded_at", pa.timestamp("us", tz="UTC")),
    ]
)


class CriteriaEmbedder:
    """Orchestrates the read -> embed -> write pipeline for one condition's criteria.

    Holds a constructed `EmbeddingModel` instance across calls the same way
    `EligibilityCriteriaFetcher` holds its HTTP session, so backend setup
    (model load / connection config) happens once per run rather than once
    per row.

    Typical usage:
        embedder = CriteriaEmbedder(build_embedding_model("huggingface"))
        output_path = embedder.run("breast cancer")
    """

    def __init__(self, embedding_model: EmbeddingModel, batch_size: int = 32):
        """Initialize the embedder with a backend and batching configuration.

        Args:
            embedding_model: The backend to embed criterion text with.
            batch_size: Number of criteria to pass to `embedding_model`'s
                `embed_batch` per call.
        """
        self.embedding_model = embedding_model
        self.batch_size = batch_size

    def read_criteria_csv(self, input_path: Path) -> list[dict]:
        """Read a criteria CSV into row dicts.

        Args:
            input_path: Path to a CSV written by
                `EligibilityCriteriaFetcher.write_eligibility_csv`.

        Returns:
            One dict per data row, keyed by the CSV's header columns
            (``nct_id``, ``study_first_posted_date``, ``criterion_text``,
            ``criterion_type``). Returns an empty list for a header-only
            (or otherwise empty) file.

        Raises:
            FileNotFoundError: If `input_path` does not exist.
        """
        if not input_path.exists():
            raise FileNotFoundError(f"Input criteria CSV not found: {input_path}")
        with open(input_path, encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    def batch_embed(self, records: list[dict]) -> list[list[float] | None]:
        """Embed every record's `criterion_text`, `batch_size` rows at a time.

        Args:
            records: Row dicts as produced by `read_criteria_csv`.

        Returns:
            One vector per record, in the same order as `records`, with
            `None` for any row whose batch failed.
        """
        embeddings: list[list[float] | None] = []
        success_count = 0
        failure_count = 0

        for start in range(0, len(records), self.batch_size):
            chunk = records[start : start + self.batch_size]
            texts = [record["criterion_text"] for record in chunk]
            chunk_embeddings = self.embedding_model.embed_batch(texts)
            embeddings.extend(chunk_embeddings)

            chunk_successes = sum(1 for embedding in chunk_embeddings if embedding is not None)
            success_count += chunk_successes
            failure_count += len(chunk_embeddings) - chunk_successes
            logger.info(
                "batch_embed: processed %d/%d rows (%d succeeded, %d failed so far)",
                min(start + self.batch_size, len(records)),
                len(records),
                success_count,
                failure_count,
            )

        return embeddings

    def build_output_rows(
        self, records: list[dict], embeddings: list[list[float] | None]
    ) -> list[dict]:
        """Zip source rows with their embeddings into output rows.

        Args:
            records: Row dicts as produced by `read_criteria_csv`.
            embeddings: One vector (or `None`) per record, as produced by
                `batch_embed`.

        Returns:
            One output dict per record, matching `PARQUET_SCHEMA`'s columns.
        """
        embedded_at = datetime.now(timezone.utc)
        rows = []
        for record, embedding in zip(records, embeddings):
            success = embedding is not None
            rows.append(
                {
                    "nct_id": record["nct_id"],
                    "study_first_posted_date": record["study_first_posted_date"],
                    "criterion_text": record["criterion_text"],
                    "criterion_type": record["criterion_type"],
                    "embedding": embedding,
                    "embedding_success": success,
                    "embedding_dim": len(embedding) if success else None,
                    "embedding_backend": self.embedding_model.backend_name,
                    "embedding_model": self.embedding_model.model_name,
                    "embedded_at": embedded_at,
                }
            )
        return rows

    def write_embeddings_parquet(self, rows: list[dict], output_path: Path) -> None:
        """Write output rows to a Parquet file, overwriting any existing file.

        Creates any missing parent directories for `output_path`. Always
        overwrites `output_path` wholesale — there is no append or
        skip-if-exists behavior, matching `write_eligibility_csv`'s existing
        overwrite convention.

        Args:
            rows: Output row dicts as produced by `build_output_rows`.
            output_path: Destination Parquet file path.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        columns = {name: [row[name] for row in rows] for name in PARQUET_SCHEMA.names}
        table = pa.Table.from_pydict(columns, schema=PARQUET_SCHEMA)
        pq.write_table(table, output_path)

    def run(
        self,
        condition: str,
        input_path: Path | None = None,
        output_path: Path | None = None,
    ) -> Path:
        """Run the full read-embed-write pipeline for one condition.

        Args:
            condition: Condition string, e.g. "breast cancer" — the same
                string passed to `EligibilityCriteriaFetcher.run`. Used to
                derive default input/output paths via
                `EligibilityCriteriaFetcher.condition_to_slug`.
            input_path: Criteria CSV to read. If omitted, defaults to
                ``INPUT_DIR / f"{slug}_trials_eligibility.csv"``.
            output_path: Where to write the resulting Parquet file. If
                omitted, defaults to
                ``OUTPUT_DIR / f"{slug}_trials_embeddings.parquet"``.

        Returns:
            The path the Parquet file was written to.

        Raises:
            FileNotFoundError: If the resolved input CSV does not exist.
            ValueError: If the input CSV has no data rows.
        """
        slug = EligibilityCriteriaFetcher.condition_to_slug(condition)
        if input_path is None:
            input_path = INPUT_DIR / f"{slug}_trials_eligibility.csv"
        if output_path is None:
            output_path = OUTPUT_DIR / f"{slug}_trials_embeddings.parquet"

        records = self.read_criteria_csv(input_path)
        if not records:
            raise ValueError(f"No criterion rows found in {input_path}; refusing to write an empty output")

        embeddings = self.batch_embed(records)
        rows = self.build_output_rows(records, embeddings)
        self.write_embeddings_parquet(rows, output_path)

        failure_count = sum(1 for row in rows if not row["embedding_success"])
        log_level = logging.ERROR if failure_count == len(rows) else logging.INFO
        logger.log(
            log_level,
            "run: wrote %d row(s) to %s (backend=%s, model=%s, dim=%s, %d failed)",
            len(rows),
            output_path,
            self.embedding_model.backend_name,
            self.embedding_model.model_name,
            self.embedding_model.embedding_dim(),
            failure_count,
        )
        return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for this module.

    Returns:
        An ``argparse.ArgumentParser`` accepting a positional ``condition``
        argument and backend/model/path/batch-size options.
    """
    parser = argparse.ArgumentParser(
        description="Embed a condition's eligibility criteria CSV into a Parquet vector file."
    )
    parser.add_argument(
        "condition",
        help='Condition to embed, e.g. "breast cancer" (same string passed to criteria_retrieval.py)',
    )
    parser.add_argument(
        "--backend",
        required=True,
        choices=["vllm", "huggingface"],
        help="Embedding backend to use",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the backend's default model name",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="Override the default input CSV path",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Override the default output Parquet path",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of criteria to embed per backend call (default: 32)",
    )
    return parser


def main() -> None:
    """CLI entry point: embed and write eligibility criteria for one condition."""
    logging.basicConfig(level=logging.INFO)

    args = build_arg_parser().parse_args()

    embedding_model = build_embedding_model(args.backend, args.model)
    embedder = CriteriaEmbedder(embedding_model, batch_size=args.batch_size)
    embedder.run(args.condition, input_path=args.input_path, output_path=args.output_path)


if __name__ == "__main__":
    main()
