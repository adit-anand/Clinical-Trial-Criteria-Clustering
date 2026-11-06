"""Two-stage Anthropic-LLM cluster relabeling for the winning trial's final output only.

`reduction_hdbscan_pipeline.run` calls `relabel_topics_with_llm` exactly once,
after the Optuna search has already picked and scored its winning trial —
never once per trial. The per-trial search itself keeps fitting BERTopic with
its own `ZeroShotClassification` representation model exactly as before (see
`bertopic_model.fit_bertopic_topics`), so the Optuna objective's
`zeroshot_label_confidence_rate` term is entirely unaffected by this module.

This module only ever overwrites `TopicRepresentation.label` for the winning
trial's non-noise topics; `.keywords`/`.topic_id`/`.count` are left untouched,
so every other consumer of a `TopicRepresentation` keeps working unmodified.

Two stages, run in sequence:
    1. Primary summarization: one LLM call per non-noise cluster, over that
       cluster's own top-N (by `cluster_probability`) eligibility criteria,
       producing a short overarching theme.
    2. Secondary summarization: one LLM call across every cluster's primary
       theme at once (semantic distinctness between clusters cannot be judged
       one cluster at a time), producing the final per-cluster label.
"""

import logging
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from amia_2026.paths import REPO_ROOT
from amia_2026.clustering.algorithms.bertopic_model import TopicRepresentation

logger = logging.getLogger(__name__)

DEFAULT_LLM_MODEL = "claude-haiku-5-5"
TOP_N_CRITERIA_PER_CLUSTER = 50
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2
MAX_OUTPUT_TOKENS = 128

DEFAULT_PRIMARY_PROMPT_PATH = REPO_ROOT / "config" / "llm_prompts" / "primary_summary_prompt.txt"
DEFAULT_SECONDARY_PROMPT_PATH = REPO_ROOT / "config" / "llm_prompts" / "secondary_summary_prompt.txt"


def _build_client():
    """Build an Anthropic client, reading `ANTHROPIC_API_KEY` from `.env` if not already set.

    Follows the same lazy `load_dotenv(REPO_ROOT / ".env")` convention used
    elsewhere in this codebase (e.g. `embedding_model.VLLMEmbeddingModel`,
    `criteria_retrieval.EligibilityCriteriaFetcher`), rather than requiring
    the caller to load the key itself.

    Returns:
        An `anthropic.Anthropic` client.
    """
    import anthropic

    load_dotenv(REPO_ROOT / ".env")
    return anthropic.Anthropic()


def _call_llm(client, model: str, system_prompt: str, user_content: str) -> str:
    """Make one retried Anthropic Messages API call and return its text response.

    Args:
        client: An `anthropic.Anthropic` client (see `_build_client`).
        model: Anthropic model id (see `DEFAULT_LLM_MODEL`).
        system_prompt: The caller-supplied system prompt (verbatim contents of
            a `--primary-summary-prompt-path`/`--secondary-summary-prompt-path` file).
        user_content: The fully-assembled user-turn content (data plus a
            fixed, code-supplied output-format instruction — see
            `_build_primary_user_content`/`_build_secondary_user_content`).

    Returns:
        The response's first `text`-type content block, stripped.

    Raises:
        RuntimeError: If the response has no `text`-type content block (e.g.
            `stop_reason == "refusal"`).
        anthropic.APIStatusError: A non-retryable (non-429, non-5xx) API error.
        anthropic.RateLimitError | anthropic.APIConnectionError: If every
            retry attempt is exhausted.
    """
    import anthropic

    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=MAX_OUTPUT_TOKENS,
                system=system_prompt,
                messages=[{"role": "user", "content": user_content}],
            )
        except anthropic.APIStatusError as exc:
            # 4xx-non-429 (bad model id, auth failure, malformed request) won't
            # resolve itself on retry -- only back off for 429/5xx.
            if not isinstance(exc, anthropic.RateLimitError) and exc.status_code < 500:
                raise
            last_exc = exc
            logger.warning("_call_llm: attempt %d/%d failed (%s); retrying", attempt, MAX_RETRIES, exc)
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
            continue
        except anthropic.APIConnectionError as exc:
            last_exc = exc
            logger.warning("_call_llm: attempt %d/%d failed (%s); retrying", attempt, MAX_RETRIES, exc)
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
            continue

        text_block = next((block for block in response.content if block.type == "text"), None)
        if text_block is None:
            raise RuntimeError(f"_call_llm: LLM call returned no text content (stop_reason={response.stop_reason!r})")
        return text_block.text.strip()

    logger.error("_call_llm: giving up after %d attempts", MAX_RETRIES)
    raise RuntimeError(f"_call_llm: giving up after {MAX_RETRIES} attempts") from last_exc


def select_top_criteria(
    criterion_texts: list[str], labels: np.ndarray, probabilities: np.ndarray, topic_id: int, top_n: int
) -> list[str]:
    """The top-`top_n` `criterion_texts` entries for one cluster, ranked by `cluster_probability` descending.

    Args:
        criterion_texts: One criterion string per row (same row order as `labels`/`probabilities`).
        labels: One cluster id per row (`BERTopicFitResult.labels`).
        probabilities: One cluster-membership probability per row (`BERTopicFitResult.probabilities`).
        topic_id: The cluster to select criteria for.
        top_n: Maximum number of criteria to return.

    Returns:
        Up to `top_n` criterion strings for `topic_id`, highest `cluster_probability` first.
        Fewer than `top_n` if the cluster itself has fewer members.
    """
    indices = [i for i in range(len(criterion_texts)) if int(labels[i]) == topic_id]
    indices.sort(key=lambda i: probabilities[i], reverse=True)
    return [criterion_texts[i] for i in indices[:top_n]]


def _build_primary_user_content(criteria: list[str], guide_labels: list[str]) -> str:
    """Assemble one cluster's primary-summarization user-turn content."""
    guide_block = ", ".join(guide_labels)
    criteria_block = "\n".join(f"{i + 1}. {text}" for i, text in enumerate(criteria))
    return (
        f"Reference labels (optional guidance, not a strict list): {guide_block}\n\n"
        f"Eligibility criteria in this cluster:\n{criteria_block}\n\n"
        "Respond with ONLY the overarching theme, 5 words or fewer. No numbering, "
        "quotation marks, or additional commentary."
    )


def primary_summarize(
    client, model: str, system_prompt: str, criteria_by_topic: dict[int, list[str]], guide_labels: list[str]
) -> dict[int, str]:
    """One Anthropic call per non-noise topic, each over that topic's own top criteria.

    Args:
        client: An `anthropic.Anthropic` client.
        model: Anthropic model id.
        system_prompt: The `--primary-summary-prompt-path` file's contents.
        criteria_by_topic: `{topic_id: top-N criterion strings}` (see `select_top_criteria`).
        guide_labels: The `--candidate-labels` list, passed through as soft guidance.

    Returns:
        `{topic_id: primary theme string}`, one entry per key of `criteria_by_topic`.
    """
    primary_labels: dict[int, str] = {}
    for topic_id, criteria in criteria_by_topic.items():
        user_content = _build_primary_user_content(criteria, guide_labels)
        primary_labels[topic_id] = _call_llm(client, model, system_prompt, user_content)
    return primary_labels


def _build_secondary_user_content(primary_labels: dict[int, str], guide_labels: list[str]) -> str:
    """Assemble the single secondary-summarization user-turn content covering every cluster at once."""
    guide_block = ", ".join(guide_labels)
    ordered_ids = sorted(primary_labels)
    themes_block = "\n".join(f"{topic_id}: {primary_labels[topic_id]}" for topic_id in ordered_ids)
    return (
        f"Reference labels (optional guidance, not a strict list): {guide_block}\n\n"
        f"Primary per-cluster themes (format '<topic_id>: <theme>'):\n{themes_block}\n\n"
        "For EACH line above, output one line in the exact format '<topic_id>: <final label>', "
        "where <final label> is 5 words or fewer, semantically distinct from every other cluster's "
        "final label, and clearly related to that cluster's own primary theme. Output exactly one "
        "line per cluster listed above, in the same order, with no extra commentary."
    )


def _parse_secondary_response(text: str, expected_ids: list[int]) -> dict[int, str]:
    """Parse `'<topic_id>: <label>'` lines; silently skip anything unparseable or unexpected.

    Args:
        text: The secondary-summarization call's raw text response.
        expected_ids: The topic ids that were actually sent to this call.

    Returns:
        `{topic_id: final label}` for every line that parsed to one of `expected_ids`.
        May have fewer entries than `expected_ids` if the model dropped or
        malformed a line -- callers fill the gap from `primary_labels`.
    """
    expected = set(expected_ids)
    parsed: dict[int, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        prefix, _, rest = line.partition(":")
        try:
            topic_id = int(prefix.strip())
        except ValueError:
            continue
        if topic_id in expected:
            parsed[topic_id] = rest.strip()
    return parsed


def secondary_summarize(
    client, model: str, system_prompt: str, primary_labels: dict[int, str], guide_labels: list[str]
) -> dict[int, str]:
    """One Anthropic call across every cluster's primary label, for cross-cluster distinctness.

    Args:
        client: An `anthropic.Anthropic` client.
        model: Anthropic model id.
        system_prompt: The `--secondary-summary-prompt-path` file's contents.
        primary_labels: `{topic_id: primary theme string}` (see `primary_summarize`).
        guide_labels: The `--candidate-labels` list, passed through as soft guidance.

    Returns:
        `{topic_id: final label}`, one entry per key of `primary_labels`. Any
        topic_id the model's response didn't address falls back to its own
        `primary_labels` entry (logged as a warning) -- never raises for a
        partial/malformed response.
    """
    if not primary_labels:
        return {}
    user_content = _build_secondary_user_content(primary_labels, guide_labels)
    response_text = _call_llm(client, model, system_prompt, user_content)
    parsed = _parse_secondary_response(response_text, list(primary_labels))

    missing = set(primary_labels) - set(parsed)
    for topic_id in missing:
        logger.warning(
            "secondary_summarize: no parsed label for topic_id=%d; falling back to its primary label", topic_id
        )
        parsed[topic_id] = primary_labels[topic_id]
    return parsed


def relabel_topics_with_llm(
    criterion_texts: list[str],
    labels: np.ndarray,
    probabilities: np.ndarray,
    topics: list[TopicRepresentation],
    guide_labels: list[str],
    model: str,
    primary_prompt_path: Path,
    secondary_prompt_path: Path,
) -> list[TopicRepresentation]:
    """Replace every non-noise topic's `.label` with a two-stage LLM-generated label.

    Noise (`topic_id == -1`) is left completely untouched -- it's never fed
    to either LLM call and its existing `TopicRepresentation` is passed
    through unchanged.

    Args:
        criterion_texts: One criterion string per row (same row order as `labels`/`probabilities`).
        labels: One cluster id per row (`BERTopicFitResult.labels`).
        probabilities: One cluster-membership probability per row (`BERTopicFitResult.probabilities`).
        topics: The winning trial's zero-shot-produced topics (`BERTopicFitResult.topics`).
        guide_labels: The `--candidate-labels` list, reused here as soft
            guidance for both LLM prompts (it keeps driving the unrelated
            per-trial `ZeroShotClassification` call too -- see
            `bertopic_model.fit_bertopic_topics`).
        model: Anthropic model id (see `DEFAULT_LLM_MODEL`).
        primary_prompt_path: `--primary-summary-prompt-path` file.
        secondary_prompt_path: `--secondary-summary-prompt-path` file.

    Returns:
        A new `topics` list, same length/order, with every non-noise entry's
        `.label` replaced. `.keywords`/`.topic_id`/`.count` are unchanged.
    """
    primary_prompt = primary_prompt_path.read_text(encoding="utf-8")
    secondary_prompt = secondary_prompt_path.read_text(encoding="utf-8")
    client = _build_client()

    criteria_by_topic = {
        topic.topic_id: select_top_criteria(
            criterion_texts, labels, probabilities, topic.topic_id, TOP_N_CRITERIA_PER_CLUSTER
        )
        for topic in topics
        if topic.topic_id != -1
    }

    primary_labels = primary_summarize(client, model, primary_prompt, criteria_by_topic, guide_labels)

    try:
        final_labels = secondary_summarize(client, model, secondary_prompt, primary_labels, guide_labels)
    except Exception:
        logger.exception(
            "relabel_topics_with_llm: secondary summarization failed; falling back to primary labels only"
        )
        final_labels = primary_labels

    return [
        replace(topic, label=final_labels[topic.topic_id]) if topic.topic_id in final_labels else topic
        for topic in topics
    ]
