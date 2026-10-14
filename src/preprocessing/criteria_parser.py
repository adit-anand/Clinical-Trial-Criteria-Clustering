"""Parse CTRP eligibility criteria text into atomic inclusion/exclusion criteria.

CTRP returns eligibility criteria in one of two shapes: already-atomic
per-criterion entries, or one undifferentiated block of text with
"Inclusion Criteria:"/"Exclusion Criteria:" headers embedded in the prose.
This module handles the second case — splitting a raw text block into
sections, then into individual criteria — plus cleanup shared by both
shapes. These are pure text-transformation functions with no network or
instance state, so they're module-level rather than methods on a class.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Header lines like "Inclusion Criteria:" or "Exclusion Criteria for fMRI
# component only:" that mark the start of a new section within an
# unstructured criteria block. Lines that themselves start with a
# bullet/numbered marker are excluded so bullet text mentioning
# "inclusion"/"exclusion" isn't mistaken for a header.
HEADER_PATTERN = re.compile(
    r"^(?!\s*(?:[-•]|\d+\.)\s)\s*.*\b(inclusion|exclusion)\b.*:\s*$",
    re.IGNORECASE,
)
NUMBERED_BULLET_PATTERN = re.compile(r"^\d+\.\s*")
UNORDERED_BULLET_PATTERN = re.compile(r"^[-•]\s*")

# Matches a leading all-caps domain-specific sub-header glued onto a
# criterion's text (e.g. "BREAST CANCER SURVIVORS:", "CAREGIVER:"), which
# CTRP sometimes includes even though it isn't an inclusion/exclusion
# header. The lookahead requires at least one actual letter so a stray
# numeric marker like "12:" isn't mistaken for a header.
ALLCAPS_HEADER_PREFIX_PATTERN = re.compile(
    r"^(?=[A-Z0-9 \-/&'(),]*[A-Z])[A-Z0-9][A-Z0-9 \-/&'(),]*:\s*"
)


def split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split a block of text on embedded inclusion/exclusion headers.

    Scans line by line for header lines like "Inclusion Criteria:" or
    "Exclusion Criteria for fMRI component only:" (matched by
    `HEADER_PATTERN`), and groups the lines that follow each header into
    that header's section.

    Args:
        text: The raw unstructured criteria text for a trial.

    Returns:
        A list of ``(criterion_type, section_text)`` pairs in document
        order, where ``criterion_type`` is ``"inclusion"`` or
        ``"exclusion"`` (lowercased from the matched header) and
        ``section_text`` is the newline-joined body of that section. Any
        lines before the first header are discarded as preamble, so this
        returns an empty list if `text` contains no recognizable headers.
    """
    sections: list[tuple[str, list[str]]] = []
    current_type: str | None = None

    for line in text.splitlines():
        if not line.strip():
            continue
        header_match = HEADER_PATTERN.match(line)
        if header_match:
            current_type = header_match.group(1).lower()
            sections.append((current_type, []))
            continue
        if current_type is not None:
            sections[-1][1].append(line)

    return [(criterion_type, "\n".join(lines)) for criterion_type, lines in sections]


def split_atomic_criteria(text: str, use_numbered_bullets: bool) -> list[str]:
    """Split one section's text into individual atomic criteria.

    Splits on a single bullet style — either numbered ("1.", "2.") or
    unordered ("-", "•") — chosen by the caller via `use_numbered_bullets`.
    Lines that don't start with a bullet of the chosen style are treated as
    a wrapped continuation of the previous criterion, since
    ClinicalTrials.gov wraps long criteria across multiple lines.

    Args:
        text: The text of a single section (e.g. everything under one
            "Inclusion Criteria:" header), as produced by
            `split_into_sections`.
        use_numbered_bullets: If True, split on `NUMBERED_BULLET_PATTERN`;
            if False, split on `UNORDERED_BULLET_PATTERN`.

    Returns:
        A list of criterion strings with bullet markers stripped and
        wrapped lines joined with a single space. If no bullets of the
        chosen style are found, the entire text is returned as one
        criterion (or an empty list if `text` is blank).
    """
    bullet_pattern = NUMBERED_BULLET_PATTERN if use_numbered_bullets else UNORDERED_BULLET_PATTERN

    items: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if bullet_pattern.match(stripped):
            if current:
                items.append(" ".join(current))
            current = [bullet_pattern.sub("", stripped, count=1)]
        elif current:
            current.append(stripped)
        # else: text preceding the first bullet of the chosen style; discarded.
    if current:
        items.append(" ".join(current))

    if not items:
        whole = " ".join(line.strip() for line in text.splitlines() if line.strip())
        return [whole] if whole else []
    return items


def criteria_use_numbered_bullets(text: str) -> bool:
    """Decide which bullet style to split on for an unstructured criteria block.

    Numbered bullets are preferred over unordered ones whenever any line in
    the text starts with a numbered marker, since trials that mix styles
    overwhelmingly use numbers for the real criteria and dashes only
    incidentally (e.g. within a single criterion's prose).

    Args:
        text: The full unstructured criteria text for a trial.

    Returns:
        True if `split_atomic_criteria` should be called with
        numbered-bullet splitting, False for unordered-bullet splitting.
    """
    return any(
        NUMBERED_BULLET_PATTERN.match(line.strip())
        for line in text.splitlines()
        if line.strip()
    )


def parse_unstructured_block(text: str, nct_id: str) -> list[tuple[str, str]]:
    """Parse a single undifferentiated block of eligibility criteria text.

    Used when the CTRP API returns eligibility criteria as one blob with
    inclusion/exclusion headers embedded in the text itself (e.g.
    NCT02111850), rather than pre-split per-criterion entries. The text is
    first divided into inclusion/exclusion sections by `split_into_sections`,
    then each section is split into atomic criteria by
    `split_atomic_criteria`, preferring numbered bullets when both styles
    appear (see `criteria_use_numbered_bullets`).

    Args:
        text: The single unstructured criteria string for this trial.
        nct_id: The trial's NCT ID, used only for log messages.

    Returns:
        A list of ``(criterion_text, criterion_type)`` pairs. If no
        inclusion/exclusion headers are found at all, the whole text is
        split into atomic criteria and each is labeled ``"both"`` rather
        than being discarded.
    """
    use_numbered_bullets = criteria_use_numbered_bullets(text)
    sections = split_into_sections(text)

    if not sections:
        logger.warning(
            "%s: unstructured eligibility text has no inclusion/exclusion headers",
            nct_id,
        )
        return [
            (item, "both")
            for item in split_atomic_criteria(text, use_numbered_bullets)
        ]

    return [
        (item, criterion_type)
        for criterion_type, section_text in sections
        for item in split_atomic_criteria(section_text, use_numbered_bullets)
    ]


def strip_allcaps_header(text: str) -> str:
    """Remove a leading all-caps section-header label from criterion text.

    CTRP sometimes glues a domain-specific sub-header (e.g.
    "BREAST CANCER SURVIVORS:", "CAREGIVER:") directly onto the front of a
    criterion's text, or returns a criterion row that is nothing but such a
    header with no actual content. Unlike `HEADER_PATTERN`, which only
    recognizes headers containing the words "inclusion"/"exclusion" and is
    used to split a whole block into sections, this recognizes *any*
    all-caps "LABEL:" prefix and is used to clean up individual
    already-split criterion strings. This strips exactly one leading match
    of `ALLCAPS_HEADER_PREFIX_PATTERN` from `text`, so a glued-prefix
    criterion is reduced to its real content, and a header-only row is
    reduced to an empty string so the caller can drop it.

    Args:
        text: A single criterion's raw text.

    Returns:
        `text` with any leading all-caps "LABEL:" prefix removed. Returns
        an empty string if `text` was nothing but such a label.
    """
    return ALLCAPS_HEADER_PREFIX_PATTERN.sub("", text, count=1)


def clean_parsed_criteria(parsed: list[tuple[str, str]], nct_id: str) -> list[tuple[str, str]]:
    """Strip glued all-caps headers and drop header-only criterion rows.

    Applies `strip_allcaps_header` to every criterion's text. A criterion
    that was nothing but a header (e.g. "PRACTICE ELIGIBILITY:") strips down
    to an empty string and is dropped; a criterion with a glued prefix (e.g.
    "CAREGIVER: A primary caregiver...") keeps its real content with the
    prefix removed.

    Args:
        parsed: ``(criterion_text, criterion_type)`` pairs as produced by
            either branch of `parse_eligibility_criteria`.
        nct_id: The trial's NCT ID, used only for log messages.

    Returns:
        The same pairs with headers stripped from ``criterion_text`` and any
        now-empty entries removed.
    """
    cleaned: list[tuple[str, str]] = []
    dropped = 0
    for criterion_text, criterion_type in parsed:
        stripped = strip_allcaps_header(criterion_text).strip()
        if not stripped:
            dropped += 1
            continue
        cleaned.append((stripped, criterion_type))

    if dropped:
        logger.info("%s: dropped %d header-only criterion row(s)", nct_id, dropped)
    return cleaned


def parse_eligibility_criteria(raw: list[dict], nct_id: str) -> list[tuple[str, str]]:
    """Normalize a CTRP eligibility response into atomic criterion rows.

    CTRP returns eligibility criteria in one of two shapes:

    * Multiple entries, each already an atomic criterion with its own
      ``display_order`` and ``inclusion_indicator`` — these are simply
      sorted and relabeled.
    * A single entry containing one undifferentiated block of text with
      "Inclusion Criteria:"/"Exclusion Criteria:" headers embedded in the
      prose — this is delegated to `parse_unstructured_block`, and the
      entry's own ``inclusion_indicator`` is ignored since it's effectively
      meaningless for a mixed block.

    Either way, the resulting pairs are passed through
    `clean_parsed_criteria` before being returned, which strips any
    glued-on all-caps sub-header (e.g. "CAREGIVER:") from each criterion's
    text and drops any criterion that turns out to be nothing but such a
    header (e.g. "PRACTICE ELIGIBILITY:").

    Args:
        raw: The raw criterion dicts as returned by
            `CTRPClient.fetch_eligibility_criteria` (i.e.
            ``data[0]["eligibility"]["unstructured"]``).
        nct_id: The trial's NCT ID, used only for log messages.

    Returns:
        A list of ``(criterion_text, criterion_type)`` pairs, where
        ``criterion_type`` is one of ``"inclusion"``, ``"exclusion"``, or
        ``"both"``. Returns an empty list if `raw` is empty.
    """
    if not raw:
        logger.warning("%s: empty eligibility criteria list", nct_id)
        return []

    if len(raw) == 1:
        parsed = parse_unstructured_block(raw[0]["description"], nct_id)
    else:
        ordered = sorted(raw, key=lambda criterion: criterion["display_order"])
        parsed = [
            (c["description"], "inclusion" if c["inclusion_indicator"] else "exclusion")
            for c in ordered
        ]

    return clean_parsed_criteria(parsed, nct_id)
