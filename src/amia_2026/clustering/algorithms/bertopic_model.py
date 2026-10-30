"""BERTopic integration: the per-trial topic-representation/clustering layer.

`reduction_hdbscan_pipeline.run` fits this once per Optuna trial, on that
trial's already-selected representative repeat (see `reduction_sweep.
representative_repeat`) — never once per seed/repeat, and never as a
separate step after the sweep finishes. BERTopic is configured to skip its
own internal UMAP step (the embeddings it's given are already reduced) and
to cluster with the exact same, already-hyperparameterized HDBSCAN estimator
the trial already committed to (`HDBSCANClusterer.build_estimator`), so this
module adds topic representations on top of a clustering decision whose
reduction and hyperparameter selection remain exactly as they were before
this feature.

BERTopic renumbers non-noise cluster/topic ids after fitting the given
HDBSCAN estimator (confirmed empirically against the installed version:
`fit_transform`'s returned `topics` is the same partition as the fitted
estimator's own `.labels_`, just permuted on non-noise ids — noise stays
`-1` in both, and `.probabilities_` is passed through unchanged). `topics`/
`probs` — not the estimator's own post-fit attributes — are therefore the
only labels/probabilities this module returns, so every topic id this
module reports always matches `get_topic_info()`'s own `Topic` column.
"""

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import HDBSCAN

logger = logging.getLogger(__name__)

DEFAULT_ZEROSHOT_MODEL = "facebook/bart-large-mnli"


def parse_candidate_labels(raw: str) -> list[str]:
    """Parse a comma-separated `--candidate-labels` CLI value into a label list.

    Args:
        raw: The raw `--candidate-labels` flag value, e.g. `"oncology eligibility,treatment history"`.

    Returns:
        One label string per comma-separated entry, stripped of surrounding whitespace.

    Raises:
        ValueError: If the result is empty (nothing left after splitting/stripping) —
            `ZeroShotClassification` has nothing to classify against otherwise.
    """
    labels = [label.strip() for label in raw.split(",") if label.strip()]
    if not labels:
        raise ValueError(f"--candidate-labels {raw!r} produced no usable labels after parsing")
    return labels


@dataclass(frozen=True)
class TopicRepresentation:
    """One cluster/topic's representation, as produced by one `fit_bertopic_topics` call.

    Attributes:
        topic_id: `-1` for BERTopic's own outlier/noise topic, matching
            HDBSCAN's own noise label; a non-negative id otherwise.
        keywords: BERTopic's own per-topic `Representation` (its c-TF-IDF
            keywords), stripped of the empty-string padding
            `get_topic_info()` returns it with. Exactly `[label]` (a single
            entry equal to `label`) when the zero-shot classification layer
            found a confident match for this topic; otherwise the raw
            keyword list.
        label: The zero-shot-assigned candidate label if one cleared
            `ZeroShotClassification`'s `min_prob` threshold, otherwise a
            space-joined fallback built from `keywords`.
        count: This topic's member count, from `get_topic_info()`'s `Count` column.
    """

    topic_id: int
    keywords: list[str]
    label: str
    count: int


@dataclass(frozen=True)
class BERTopicFitResult:
    """The outcome of one `fit_bertopic_topics` call.

    Attributes:
        labels: One cluster/topic id per document (`fit_transform`'s own
            `topics` return value — see this module's docstring for why this,
            not the fitted HDBSCAN estimator's own `.labels_`, is authoritative).
        probabilities: One cluster-membership probability per document.
        topics: One `TopicRepresentation` per distinct id present in `labels`
            (including `-1`).
        topic_embeddings: Each topic's centroid embedding, keyed by
            `topic_id` (including `-1`) rather than stored as a bare
            position-ordered array. BERTopic's own `topic_embeddings_`
            *is* a bare array, row-ordered by an internal sort that happens
            to agree with `get_topic_info()`'s own row order today (both
            ultimately sort the same topic-id universe ascending) but isn't
            a documented public contract between the two — keying this dict
            explicitly by id (see `fit_bertopic_topics`) means a consumer
            can never silently misattribute one topic's embedding to
            another if that coincidence ever stops holding.
    """

    labels: np.ndarray
    probabilities: np.ndarray
    topics: list[TopicRepresentation]
    topic_embeddings: dict[int, np.ndarray]


def same_partition(expected: np.ndarray, actual: np.ndarray) -> bool:
    """Check whether two label arrays describe the same partition, up to a relabeling of non-noise ids.

    Used to sanity-check BERTopic's own HDBSCAN refit against the seed/repeat
    sweep's earlier fit over the identical reduced matrix/hyperparameters:
    both are expected to group points identically, but BERTopic renumbers
    non-noise ids (see this module's docstring), so a raw `np.array_equal`
    would almost always report a spurious mismatch.

    Args:
        expected: One label array (e.g. `representative.labels`).
        actual: Another label array of the same length (e.g. a `BERTopicFitResult.labels`).

    Returns:
        `True` if every point grouped together in `expected` is grouped
        together in `actual` (and vice versa) and noise (`-1`) points match
        exactly in both; `False` otherwise.
    """
    if expected.shape != actual.shape:
        return False
    if (expected == -1).tolist() != (actual == -1).tolist():
        return False
    mapping: dict[int, int] = {}
    for e, a in zip(expected.tolist(), actual.tolist()):
        if e == -1:
            continue
        if e in mapping:
            if mapping[e] != a:
                return False
        else:
            mapping[e] = a
    return True


def fit_bertopic_topics(
    criterion_texts: list[str],
    reduced_matrix: np.ndarray,
    hdbscan_estimator: HDBSCAN,
    candidate_labels: list[str],
    zeroshot_model: str,
) -> BERTopicFitResult:
    """Fit BERTopic once over an already-reduced embedding matrix, reusing a preconfigured HDBSCAN estimator.

    Args:
        criterion_texts: One document string per row of `reduced_matrix`
            (this trial's surviving criteria' `criterion_text`), used as the
            document text for computing topic representations.
        reduced_matrix: The reducer's already-reduced output for this
            trial's representative repeat — never re-reduced by BERTopic.
        hdbscan_estimator: An *unfit* estimator built from this trial's own
            hyperparameters (see `HDBSCANClusterer.build_estimator`).
            BERTopic fits this object internally.
        candidate_labels: The caller-supplied candidate topic labels (see
            `parse_candidate_labels`) `ZeroShotClassification` scores each
            topic against.
        zeroshot_model: The HuggingFace zero-shot-classification model id or
            an already-initialized `transformers` pipeline to pass to
            `ZeroShotClassification`.

    Returns:
        A `BERTopicFitResult`.
    """
    from bertopic import BERTopic
    from bertopic.dimensionality import BaseDimensionalityReduction
    from bertopic.representation import ZeroShotClassification
    from sklearn.feature_extraction.text import CountVectorizer

    umap_model = BaseDimensionalityReduction()
    representation_model = ZeroShotClassification(candidate_topics=candidate_labels, model=zeroshot_model)
    vectorizer_model = CountVectorizer(stop_words="english")

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_estimator,
        representation_model=representation_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=False,
    )
    topics, probs = topic_model.fit_transform(criterion_texts, embeddings=reduced_matrix)
    labels = np.asarray(topics)
    probabilities = np.asarray(probs)

    topic_info = topic_model.get_topic_info()
    topic_representations = []
    for row in topic_info.itertuples(index=False):
        keywords = [word for word in row.Representation if word]
        label = " ".join(keywords) if keywords else ""
        topic_representations.append(
            TopicRepresentation(topic_id=int(row.Topic), keywords=keywords, label=label, count=int(row.Count))
        )

    # `topic_embeddings_`'s row order is never trusted positionally (see
    # `BERTopicFitResult.topic_embeddings`'s docstring) -- reconstruct the
    # id-to-row correspondence from `topic_sizes_`, the same topic-id
    # universe `topic_embeddings_` was itself built from (both ultimately
    # derive from the same fitted `documents` frame's `Topic` column), and
    # fail loudly rather than silently misalign if the two ever disagree.
    topic_ids_in_order = sorted(topic_model.topic_sizes_.keys())
    raw_topic_embeddings = np.asarray(topic_model.topic_embeddings_)
    if raw_topic_embeddings.shape[0] != len(topic_ids_in_order):
        raise RuntimeError(
            f"fit_bertopic_topics: topic_embeddings_ has {raw_topic_embeddings.shape[0]} row(s) but "
            f"{len(topic_ids_in_order)} topic id(s) are present in topic_sizes_; cannot safely align"
        )
    topic_embeddings = {tid: raw_topic_embeddings[i] for i, tid in enumerate(topic_ids_in_order)}

    return BERTopicFitResult(
        labels=labels, probabilities=probabilities, topics=topic_representations, topic_embeddings=topic_embeddings
    )
