"""Text embedding backends for eligibility criteria.

Defines a single `EmbeddingModel` interface implemented by two interchangeable
backends:

* `VLLMEmbeddingModel` — calls an already-running, externally-managed vLLM
  server's OpenAI-compatible ``/v1/embeddings`` endpoint over HTTP.
* `HuggingFaceEmbeddingModel` — runs a `sentence-transformers` model
  in-process.

Both backends return pooled + normalized vectors on their own (vLLM
server-side; `sentence-transformers` via ``normalize_embeddings=True``), so no
manual pooling/normalization is implemented here. Orchestration code (see
`criteria_embedding.py`) should depend only on `EmbeddingModel` and
`build_embedding_model`, never on a concrete backend class, so adding a third
backend never requires touching the orchestration layer.
"""

import logging
import os
import time
from abc import ABC, abstractmethod
from typing import ClassVar

import requests
from dotenv import load_dotenv

from src.paths import REPO_ROOT

logger = logging.getLogger(__name__)


class EmbeddingModel(ABC):
    """Shared contract for every embedding backend.

    Orchestration code should depend only on this interface, never branch on
    which concrete backend is in use.
    """

    model_name: str
    backend_name: ClassVar[str]

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        """Embed a batch of strings in one backend call.

        Args:
            texts: The strings to embed, in order.

        Returns:
            One vector per input, in the same order as `texts`. An entry is
            ``None`` if that row's embedding could not be produced, so
            partial failures are visible rather than silently dropped.
        """
        raise NotImplementedError

    @abstractmethod
    def embedding_dim(self) -> int | None:
        """Report this instance's embedding dimensionality.

        Returns:
            The vector length produced by `embed_batch`, or None if it
            cannot be determined (e.g. no successful call has been made yet
            for a backend that infers it lazily).
        """
        raise NotImplementedError


class VLLMEmbeddingModel(EmbeddingModel):
    """Calls an externally-managed vLLM server's OpenAI-compatible embeddings endpoint.

    An instance holds the HTTP session, base URL, and API key used across
    calls, so a single instance can be reused across many batches without
    re-resolving configuration each time.
    """

    backend_name: ClassVar[str] = "vllm"

    DEFAULT_MODEL_NAME = "gpt-oss-20b"
    DEFAULT_BASE_URL = "http://localhost:8000/v1"

    MAX_RETRIES = 3
    RETRY_BACKOFF_SECONDS = 2
    REQUEST_TIMEOUT_SECONDS = 60

    def __init__(
        self,
        model_name: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        session: requests.Session | None = None,
        load_env: bool = True,
    ):
        """Initialize the vLLM client's connection settings.

        Args:
            model_name: Model identifier to send in each request's
                ``"model"`` field. Defaults to `DEFAULT_MODEL_NAME`.
            base_url: Base URL of the vLLM server's OpenAI-compatible API
                (e.g. ``"http://localhost:8000/v1"``). If not provided, falls
                back to the ``VLLM_BASE_URL`` environment variable, then
                `DEFAULT_BASE_URL`.
            api_key: API key to send as a bearer token, if the server
                requires one. If not provided, falls back to the
                ``VLLM_API_KEY`` environment variable. May be left unset if
                the server doesn't require authentication.
            session: A ``requests.Session`` to reuse for all HTTP calls. If
                omitted, a new session is created.
            load_env: If True, load variables from a ``.env`` file at the
                repo root (via ``python-dotenv``) before reading
                ``VLLM_BASE_URL``/``VLLM_API_KEY`` from the environment. Has
                no effect on any argument given explicitly.
        """
        if load_env and (base_url is None or api_key is None):
            load_dotenv(REPO_ROOT / ".env")

        self.model_name = model_name or self.DEFAULT_MODEL_NAME
        self.base_url = (base_url or os.environ.get("VLLM_BASE_URL") or self.DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or os.environ.get("VLLM_API_KEY")
        self.session = session or requests.Session()
        self._embedding_dim: int | None = None

    def build_headers(self) -> dict:
        """Build request headers, including bearer auth only if configured.

        Returns:
            A headers dict with ``Content-Type: application/json``, plus an
            ``Authorization: Bearer <api_key>`` entry if `api_key` is set.
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def post_embeddings_request(self, texts: list[str]) -> dict:
        """POST a batch of texts to the server's ``/embeddings`` endpoint.

        Retries on timeout, HTTP 429, and HTTP 5xx responses with linear
        backoff, up to `MAX_RETRIES` attempts.

        Args:
            texts: The strings to embed, in order.

        Returns:
            The parsed JSON response body, expected to contain a
            ``"data"`` list of ``{"embedding": [...], "index": ...}`` dicts.

        Raises:
            requests.RequestException: If every retry attempt fails (timeout
                or a retryable status code), or if the server returns a
                non-retryable error status.
        """
        url = f"{self.base_url}/embeddings"
        payload = {"model": self.model_name, "input": texts}
        headers = self.build_headers()

        last_error: requests.RequestException | None = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                response = self.session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=self.REQUEST_TIMEOUT_SECONDS,
                )
            except requests.Timeout as exc:
                last_error = exc
                logger.warning(
                    "vLLM embeddings request timed out (attempt %d/%d), retrying",
                    attempt,
                    self.MAX_RETRIES,
                )
                time.sleep(self.RETRY_BACKOFF_SECONDS * attempt)
                continue

            if response.status_code == 429 or response.status_code >= 500:
                last_error = requests.HTTPError(
                    f"vLLM server returned {response.status_code}", response=response
                )
                logger.warning(
                    "vLLM embeddings request returned %d (attempt %d/%d), retrying",
                    response.status_code,
                    attempt,
                    self.MAX_RETRIES,
                )
                time.sleep(self.RETRY_BACKOFF_SECONDS * attempt)
                continue

            response.raise_for_status()
            return response.json()

        logger.error("vLLM embeddings request failed after %d attempts", self.MAX_RETRIES)
        raise last_error

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        """Embed a batch of strings via the vLLM server.

        Args:
            texts: The strings to embed, in order.

        Returns:
            One vector per input, in the same order as `texts`. If the
            request fails after retries, returns ``[None] * len(texts)`` so
            one bad batch doesn't lose progress from earlier batches.
        """
        try:
            response_json = self.post_embeddings_request(texts)
        except requests.RequestException:
            logger.error("vLLM embed_batch: giving up on a batch of %d text(s)", len(texts))
            return [None] * len(texts)

        entries = sorted(response_json["data"], key=lambda entry: entry["index"])
        embeddings = [entry["embedding"] for entry in entries]

        if embeddings and self._embedding_dim is None:
            self._embedding_dim = len(embeddings[0])

        return embeddings

    def embedding_dim(self) -> int | None:
        """Report this model's embedding dimensionality.

        Returns:
            The vector length observed in the most recent successful
            `embed_batch` call, or None if no batch has succeeded yet.
        """
        return self._embedding_dim


class HuggingFaceEmbeddingModel(EmbeddingModel):
    """Runs an embedding model in-process via `sentence-transformers`."""

    backend_name: ClassVar[str] = "huggingface"

    DEFAULT_MODEL_NAME = "Qwen/Qwen3-Embedding-8B"

    def __init__(self, model_name: str | None = None, device: str | None = None):
        """Load the `sentence-transformers` model once at construction.

        The expensive model/weights load happens here, not per batch, so a
        single instance should be reused across all batches for a run.

        Args:
            model_name: HuggingFace model identifier to load. Defaults to
                `DEFAULT_MODEL_NAME`.
            device: Device to load the model onto (e.g. ``"cpu"``,
                ``"cuda"``). If omitted, `sentence-transformers` picks a
                default.

        Raises:
            Exception: Whatever `sentence_transformers.SentenceTransformer`
                raises if the model can't be resolved/downloaded/loaded;
                allowed to propagate directly for a fast, clear failure.
        """
        from sentence_transformers import SentenceTransformer

        load_dotenv(REPO_ROOT / ".env")

        self.model_name = model_name or self.DEFAULT_MODEL_NAME
        self.model = SentenceTransformer(
            self.model_name,
            device=device,
            token=os.environ.get("HF_TOKEN"),
        )

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        """Embed a batch of strings via the loaded `sentence-transformers` model.

        Relies on the model's own tokenizer truncation (``max_seq_length``)
        for long text — no manual truncation is implemented here.

        Args:
            texts: The strings to embed, in order.

        Returns:
            One vector per input, in the same order as `texts`. If
            `SentenceTransformer.encode` raises, returns
            ``[None] * len(texts)``, matching the vLLM backend's
            all-or-nothing-per-batch failure contract.
        """
        try:
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception:
            logger.exception("HuggingFace embed_batch: failed on a batch of %d text(s)", len(texts))
            return [None] * len(texts)

        return [embedding.tolist() for embedding in embeddings]

    def embedding_dim(self) -> int | None:
        """Report this model's embedding dimensionality.

        Returns:
            The dimensionality reported by the underlying
            `SentenceTransformer` instance.
        """
        return self.model.get_embedding_dimension()


def build_embedding_model(backend: str, model_name: str | None = None, **kwargs) -> EmbeddingModel:
    """Construct an `EmbeddingModel` for the named backend.

    The single place a future third backend gets registered.

    Args:
        backend: ``"vllm"`` or ``"huggingface"``.
        model_name: Model identifier to pass through to the backend's
            constructor. If omitted, the backend uses its own default.
        **kwargs: Additional backend-specific constructor arguments (e.g.
            `base_url`/`api_key` for vLLM, `device` for HuggingFace).

    Returns:
        A constructed `EmbeddingModel` instance.

    Raises:
        ValueError: If `backend` is not a recognized backend name.
    """
    if backend == "vllm":
        return VLLMEmbeddingModel(model_name=model_name, **kwargs)
    if backend == "huggingface":
        return HuggingFaceEmbeddingModel(model_name=model_name, **kwargs)
    raise ValueError(f"Unknown embedding backend: {backend!r} (expected 'vllm' or 'huggingface')")
