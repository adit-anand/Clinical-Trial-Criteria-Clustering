"""Content-hash-keyed, Parquet-backed cache for embedding vectors.

One cache file is intended per input CSV, shared across backends/models/
pooling strategies/schema versions for that file — the cache key itself
(see `build_cache_key`) is what disambiguates entries, not the file layout.

Deliberate exception to this repo's conventions: every other pipeline output
(criteria CSV, embeddings Parquet, clustering assignments/metrics) is
wholesale-overwritten on rerun. This cache instead *accumulates* across
calls/runs — `EmbeddingCache.put_many` reads whatever already exists at
`cache_path`, merges in newly-successful rows, and rewrites the whole merged
table back immediately. That rewrite-on-every-call behavior is intentional:
it's what makes a crash mid-run safe (every already-completed batch is
durably on disk before the next batch starts), and it's what lets a later
run with a different `--model`/`--pooling-strategy` add new rows without
disturbing or deleting old ones. Do not "fix" this toward the rest of the
repo's overwrite-on-rerun style — that would defeat the entire point of
having a cache.

Only successful embeddings are ever stored. A cache key's mere presence in
the file is the success signal — there is no separate failure column, so a
previously-failed row is indistinguishable from "never attempted" and is
always retried on the next run.
"""

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

CACHE_SCHEMA = pa.schema(
    [
        ("cache_key", pa.string()),
        ("criterion_text", pa.string()),  # debug-only, not used for lookup
        ("model_name", pa.string()),
        ("backend_name", pa.string()),
        ("pooling_strategy", pa.string()),
        ("cache_schema_version", pa.int32()),
        ("embedding", pa.list_(pa.float32())),
        ("embedding_dim", pa.int32()),
        ("cached_at", pa.timestamp("us", tz="UTC")),
    ]
)


def build_cache_key(
    criterion_text: str,
    model_name: str,
    backend_name: str,
    pooling_strategy: str,
    cache_schema_version: int,
) -> str:
    """Derive a content-based cache key from a row's embedding inputs.

    Deliberately excludes `nct_id`/`criterion_type` (not part of
    `embed_batch`'s shared interface, and not unique per row anyway) and
    batch size (architecturally irrelevant to a transformer's per-row
    output). `cache_schema_version` is baked into the key itself so a future
    pooling/normalization change invalidates old entries automatically, with
    no explicit migration step — old rows simply become unreachable.

    Each component is length-prefixed before hashing so that, e.g.,
    ``("ab", "c")`` and ``("a", "bc")`` can never collide via naive
    concatenation.

    Args:
        criterion_text: The raw criterion text being embedded.
        model_name: The embedding model identifier.
        backend_name: The embedding backend identifier (e.g. ``"bert"``).
        pooling_strategy: The pooling strategy used (e.g. ``"mean"``).
        cache_schema_version: The cache schema version in effect.

    Returns:
        A hex-encoded SHA-256 digest uniquely identifying this combination.
    """
    hasher = hashlib.sha256()
    components = (
        criterion_text,
        model_name,
        backend_name,
        pooling_strategy,
        str(cache_schema_version),
    )
    for component in components:
        encoded = component.encode("utf-8")
        hasher.update(len(encoded).to_bytes(8, "big"))
        hasher.update(encoded)
    return hasher.hexdigest()


class EmbeddingCache:
    """An in-memory-mirrored, Parquet-backed cache of successful embeddings.

    Loads any existing cache file fully into memory at construction, then
    keeps that in-memory dict as the source of truth for the lifetime of the
    instance — `get_many` never touches disk, and `put_many` writes the
    whole merged dict back to disk immediately after updating it.
    """

    def __init__(self, cache_path: Path):
        """Load an existing cache file into memory, or start empty.

        Args:
            cache_path: Parquet file path this cache reads from/writes to.
                Need not exist yet — a missing file just means an empty
                starting cache.
        """
        self.cache_path = cache_path
        self._entries: dict[str, dict] = {}

        if cache_path.exists():
            table = pq.read_table(cache_path)
            for row in table.to_pylist():
                self._entries[row["cache_key"]] = row
            logger.info("EmbeddingCache: loaded %d cached row(s) from %s", len(self._entries), cache_path)

    def get_many(
        self,
        texts: list[str],
        model_name: str,
        backend_name: str,
        pooling_strategy: str,
        cache_schema_version: int,
    ) -> list[list[float] | None]:
        """Look up cached embeddings for a batch of texts.

        Args:
            texts: The strings to look up, in order.
            model_name: The embedding model identifier.
            backend_name: The embedding backend identifier.
            pooling_strategy: The pooling strategy in effect.
            cache_schema_version: The cache schema version in effect.

        Returns:
            One vector per input, in the same order as `texts`. An entry is
            ``None`` if that text isn't cached under this exact
            model/backend/pooling/schema-version combination — meaning
            "needs computing," never "cached as a failure."
        """
        results: list[list[float] | None] = []
        for text in texts:
            key = build_cache_key(text, model_name, backend_name, pooling_strategy, cache_schema_version)
            entry = self._entries.get(key)
            results.append(list(entry["embedding"]) if entry is not None else None)
        return results

    def put_many(
        self,
        texts: list[str],
        embeddings: list[list[float] | None],
        model_name: str,
        backend_name: str,
        pooling_strategy: str,
        cache_schema_version: int,
    ) -> None:
        """Upsert newly-successful embeddings and persist the merged cache.

        Only entries with a non-``None`` embedding are stored — a failed row
        is simply skipped, so it remains absent (and therefore retryable)
        rather than being recorded as a cached failure.

        Args:
            texts: The strings that were just embedded, in order.
            embeddings: One vector (or `None` for a failure) per text, in
                the same order as `texts`.
            model_name: The embedding model identifier used to produce
                `embeddings`.
            backend_name: The embedding backend identifier used.
            pooling_strategy: The pooling strategy used.
            cache_schema_version: The cache schema version these embeddings
                were produced under.
        """
        cached_at = datetime.now(timezone.utc)
        new_row_count = 0

        for text, embedding in zip(texts, embeddings):
            if embedding is None:
                continue
            key = build_cache_key(text, model_name, backend_name, pooling_strategy, cache_schema_version)
            self._entries[key] = {
                "cache_key": key,
                "criterion_text": text,
                "model_name": model_name,
                "backend_name": backend_name,
                "pooling_strategy": pooling_strategy,
                "cache_schema_version": cache_schema_version,
                "embedding": embedding,
                "embedding_dim": len(embedding),
                "cached_at": cached_at,
            }
            new_row_count += 1

        if new_row_count == 0:
            return

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        columns = {name: [entry[name] for entry in self._entries.values()] for name in CACHE_SCHEMA.names}
        table = pa.Table.from_pydict(columns, schema=CACHE_SCHEMA)
        pq.write_table(table, self.cache_path)
        logger.info(
            "EmbeddingCache: wrote %d new row(s) (%d total) to %s",
            new_row_count,
            len(self._entries),
            self.cache_path,
        )
