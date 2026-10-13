"""Fetch NCI-affiliated clinical trials and their eligibility criteria.

This module wraps two public APIs behind a single ``EligibilityCriteriaFetcher``
class:

* ClinicalTrials.gov (``/api/v2/studies``) — used to search for interventional
  trials matching a condition, sponsored/affiliated with the National Cancer
  Institute (NCI).
* The NCI Clinical Trials Reporting Program (CTRP) API
  (``/api/v2/trials``) — used to look up a trial's eligibility criteria, which
  may come back either pre-split per-criterion or as one unstructured block of
  text that still needs to be parsed into atomic inclusion/exclusion criteria.
"""

import argparse
import csv
import logging
import os
import re
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "data" / "eligibility_criteria"

logger = logging.getLogger(__name__)


class EligibilityCriteriaFetcher:
    """Searches ClinicalTrials.gov and the CTRP API for NCI trial eligibility criteria.

    An instance holds the HTTP session and CTRP API key used across calls, so
    that a single fetcher can be reused to process many conditions without
    re-authenticating or re-opening connections each time.

    Typical usage:
        fetcher = EligibilityCriteriaFetcher()
        output_path = fetcher.run("breast cancer")
    """

    # --- API endpoints -----------------------------------------------------
    CT_GOV_STUDIES_URL = "https://clinicaltrials.gov/api/v2/studies"
    CTRP_TRIALS_URL = "https://clinicaltrialsapi.cancer.gov/api/v2/trials"

    # --- ClinicalTrials.gov search configuration ----------------------------
    SEARCH_FIELDS = "NCTId,StudyFirstPostDate,LeadSponsorName,CollaboratorName"
    SEARCH_PAGE_SIZE = 1000
    NCI_NAME_FRAGMENTS = ("national cancer institute", "nci")

    # --- CTRP request retry/pacing configuration ----------------------------
    MAX_RETRIES = 3
    RETRY_BACKOFF_SECONDS = 2
    REQUEST_PACING_SECONDS = 0.15

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

    def __init__(
        self,
        api_key: str | None = None,
        session: requests.Session | None = None,
        load_env: bool = True,
    ):
        """Initialize the fetcher's HTTP session and CTRP credentials.

        Args:
            api_key: CTRP API key to send as the ``X-API-KEY`` header. If not
                provided, falls back to the ``CTRP_API_KEY`` environment
                variable. May be left unset if the caller only intends to use
                `fetch_matching_trials`, which does not require a key;
                `fetch_eligibility_criteria` will raise if no key is
                available when it's actually called.
            session: A ``requests.Session`` to reuse for all HTTP calls (e.g.
                to share connection pooling with other code). If omitted, a
                new session is created.
            load_env: If True, load variables from a ``.env`` file at the repo
                root (via ``python-dotenv``) before reading ``CTRP_API_KEY``
                from the environment. Has no effect if `api_key` is given
                explicitly.
        """
        if load_env and api_key is None:
            load_dotenv(REPO_ROOT / ".env")
        self.api_key = api_key or os.environ.get("CTRP_API_KEY")
        self.session = session or requests.Session()

    # ------------------------------------------------------------------
    # ClinicalTrials.gov search
    # ------------------------------------------------------------------

    def build_trial_search_params(self, condition: str) -> dict:
        """Build the ClinicalTrials.gov query parameters for a condition search.

        Restricts results to interventional trials sponsored/affiliated with
        the National Cancer Institute, posted on or after 2014-01-01, with an
        overall status of Active-not-recruiting or Completed.

        Args:
            condition: Condition to search for, e.g. "breast cancer". Passed
                through to the API's ``query.cond`` parameter.

        Returns:
            A dict of query parameters suitable for passing as the ``params``
            argument of a GET request to `CT_GOV_STUDIES_URL`. Does not
            include a page token; callers add one when paginating.
        """
        return {
            "query.cond": condition,
            "query.spons": "National Cancer Institute",
            "filter.overallStatus": "ACTIVE_NOT_RECRUITING,COMPLETED",
            "filter.advanced": "AREA[StudyType]INTERVENTIONAL",
            "query.term": "AREA[StudyFirstPostDate]RANGE[2014-01-01,MAX]",
            "fields": self.SEARCH_FIELDS,
            "pageSize": self.SEARCH_PAGE_SIZE,
            "countTotal": "true",
        }

    def is_nci_affiliated(self, sponsor_collaborators_module: dict) -> bool:
        """Check whether a study's lead sponsor or any collaborator is the NCI.

        ClinicalTrials.gov's ``query.spons`` filter is a loose text match, so
        it can return false positives (e.g. sponsors that merely mention
        "National Cancer Institute" in passing). This performs a stricter
        substring check against the lead sponsor and all collaborator names.

        Args:
            sponsor_collaborators_module: The ``sponsorCollaboratorsModule``
                dict from a study's ``protocolSection``, as returned by the
                ClinicalTrials.gov API.

        Returns:
            True if any sponsor/collaborator name contains one of
            `NCI_NAME_FRAGMENTS` (case-insensitive), False otherwise.
        """
        sponsor_names = [
            sponsor_collaborators_module.get("leadSponsor", {}).get("name", "")
        ]
        sponsor_names.extend(
            collaborator.get("name", "")
            for collaborator in sponsor_collaborators_module.get("collaborators", [])
        )
        return any(
            fragment in name.lower()
            for name in sponsor_names
            for fragment in self.NCI_NAME_FRAGMENTS
        )

    def extract_trial_summary(self, study: dict) -> tuple[str, str]:
        """Pull the NCT ID and first-posted date out of one study record.

        Args:
            study: A single element of the ``studies`` array from a
                ClinicalTrials.gov search response.

        Returns:
            A ``(nct_id, study_first_posted_date)`` tuple.
        """
        protocol = study["protocolSection"]
        nct_id = protocol["identificationModule"]["nctId"]
        posted_date = protocol["statusModule"]["studyFirstPostDateStruct"]["date"]
        return nct_id, posted_date

    def fetch_matching_trials(self, condition: str) -> list[dict]:
        """Page through ClinicalTrials.gov for NCI trials matching a condition.

        Searches for interventional trials matching `condition`, filters out
        false-positive sponsor matches via `_is_nci_affiliated`, and
        de-duplicates by NCT ID across pages.

        Args:
            condition: Condition to search for, e.g. "breast cancer".

        Returns:
            A list of dicts, one per matching trial, each with keys
            ``"nct_id"`` and ``"study_first_posted_date"``.
        """
        params = self.build_trial_search_params(condition)

        trials_by_nct_id: dict[str, dict] = {}
        reported_total: int | None = None
        page_token: str | None = None

        while True:
            page_params = dict(params)
            if page_token:
                page_params["pageToken"] = page_token

            response = self.session.get(self.CT_GOV_STUDIES_URL, params=page_params)
            response.raise_for_status()
            payload = response.json()

            if reported_total is None:
                reported_total = payload.get("totalCount")

            for study in payload.get("studies", []):
                nct_id, posted_date = self.extract_trial_summary(study)
                sponsor_module = study["protocolSection"].get(
                    "sponsorCollaboratorsModule", {}
                )

                if not self.is_nci_affiliated(sponsor_module):
                    logger.info(
                        "%s: dropping false positive, no NCI sponsor/collaborator match",
                        nct_id,
                    )
                    continue

                if nct_id in trials_by_nct_id:
                    continue
                trials_by_nct_id[nct_id] = {
                    "nct_id": nct_id,
                    "study_first_posted_date": posted_date,
                }

            page_token = payload.get("nextPageToken")
            if not page_token:
                break

        logger.info(
            "fetch_matching_trials: found %d trials (API countTotal=%s)",
            len(trials_by_nct_id),
            reported_total,
        )
        return list(trials_by_nct_id.values())

    # ------------------------------------------------------------------
    # CTRP eligibility criteria retrieval
    # ------------------------------------------------------------------

    def fetch_eligibility_criteria(self, nct_id: str) -> list[dict] | None:
        """Look up a single trial's unstructured eligibility criteria from CTRP.

        Retries on HTTP 429 (rate limited) and 5xx (server error) responses
        with linear backoff, up to `MAX_RETRIES` attempts. Successful
        responses are paced by `REQUEST_PACING_SECONDS` to avoid tripping the
        API's rate limit on subsequent calls.

        Args:
            nct_id: The trial's ClinicalTrials.gov identifier (e.g.
                "NCT01234567").

        Returns:
            The list of raw criterion dicts from the CTRP response's
            ``data[0]["eligibility"]["unstructured"]`` field, or None if the
            API returned no data for this trial or all retry attempts were
            exhausted.

        Raises:
            RuntimeError: If no CTRP API key was configured on this fetcher.
            requests.HTTPError: If the API returns a non-retryable error
                status (anything other than 429/5xx).
        """
        if not self.api_key:
            raise RuntimeError(
                "No CTRP API key configured. Pass api_key= to "
                "EligibilityCriteriaFetcher() or set CTRP_API_KEY."
            )

        headers = {"X-API-KEY": self.api_key}
        params = {"nct_id": nct_id}

        for attempt in range(1, self.MAX_RETRIES + 1):
            response = self.session.get(
                self.CTRP_TRIALS_URL, headers=headers, params=params
            )

            if response.status_code == 429 or response.status_code >= 500:
                logger.warning(
                    "%s: CTRP API returned %d (attempt %d/%d), retrying",
                    nct_id,
                    response.status_code,
                    attempt,
                    self.MAX_RETRIES,
                )
                time.sleep(self.RETRY_BACKOFF_SECONDS * attempt)
                continue

            response.raise_for_status()
            payload = response.json()
            time.sleep(self.REQUEST_PACING_SECONDS)

            data = payload.get("data", [])
            if not data:
                logger.warning("%s: no eligibility data returned by CTRP API", nct_id)
                return None
            return data[0].get("eligibility", {}).get("unstructured", [])

        logger.warning("%s: giving up on CTRP API after %d attempts", nct_id, self.MAX_RETRIES)
        return None

    # ------------------------------------------------------------------
    # Unstructured criteria parsing
    # ------------------------------------------------------------------

    def split_into_sections(self, text: str) -> list[tuple[str, str]]:
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
            header_match = self.HEADER_PATTERN.match(line)
            if header_match:
                current_type = header_match.group(1).lower()
                sections.append((current_type, []))
                continue
            if current_type is not None:
                sections[-1][1].append(line)

        return [(criterion_type, "\n".join(lines)) for criterion_type, lines in sections]

    def split_atomic_criteria(self, text: str, use_numbered_bullets: bool) -> list[str]:
        """Split one section's text into individual atomic criteria.

        Splits on a single bullet style — either numbered ("1.", "2.") or
        unordered ("-", "•") — chosen by the caller via
        `use_numbered_bullets`. Lines that don't start with a bullet of the
        chosen style are treated as a wrapped continuation of the previous
        criterion, since ClinicalTrials.gov wraps long criteria across
        multiple lines.

        Args:
            text: The text of a single section (e.g. everything under one
                "Inclusion Criteria:" header), as produced by
                `_split_into_sections`.
            use_numbered_bullets: If True, split on `NUMBERED_BULLET_PATTERN`;
                if False, split on `UNORDERED_BULLET_PATTERN`.

        Returns:
            A list of criterion strings with bullet markers stripped and
            wrapped lines joined with a single space. If no bullets of the
            chosen style are found, the entire text is returned as one
            criterion (or an empty list if `text` is blank).
        """
        bullet_pattern = (
            self.NUMBERED_BULLET_PATTERN if use_numbered_bullets else self.UNORDERED_BULLET_PATTERN
        )

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

    def criteria_use_numbered_bullets(self, text: str) -> bool:
        """Decide which bullet style to split on for an unstructured criteria block.

        Numbered bullets are preferred over unordered ones whenever any line
        in the text starts with a numbered marker, since trials that mix
        styles overwhelmingly use numbers for the real criteria and dashes
        only incidentally (e.g. within a single criterion's prose).

        Args:
            text: The full unstructured criteria text for a trial.

        Returns:
            True if `_split_atomic_criteria` should be called with
            numbered-bullet splitting, False for unordered-bullet splitting.
        """
        return any(
            self.NUMBERED_BULLET_PATTERN.match(line.strip())
            for line in text.splitlines()
            if line.strip()
        )

    def parse_unstructured_block(self, text: str, nct_id: str) -> list[tuple[str, str]]:
        """Parse a single undifferentiated block of eligibility criteria text.

        Used when the CTRP API returns eligibility criteria as one blob with
        inclusion/exclusion headers embedded in the text itself (e.g.
        NCT02111850), rather than pre-split per-criterion entries. The text is
        first divided into inclusion/exclusion sections by
        `split_into_sections`, then each section is split into atomic
        criteria by `split_atomic_criteria`, preferring numbered bullets when
        both styles appear (see `criteria_use_numbered_bullets`).

        Args:
            text: The single unstructured criteria string for this trial.
            nct_id: The trial's NCT ID, used only for log messages.

        Returns:
            A list of ``(criterion_text, criterion_type)`` pairs. If no
            inclusion/exclusion headers are found at all, the whole text is
            split into atomic criteria and each is labeled ``"both"`` rather
            than being discarded.
        """
        use_numbered_bullets = self.criteria_use_numbered_bullets(text)
        sections = self.split_into_sections(text)

        if not sections:
            logger.warning(
                "%s: unstructured eligibility text has no inclusion/exclusion headers",
                nct_id,
            )
            return [
                (item, "both")
                for item in self.split_atomic_criteria(text, use_numbered_bullets)
            ]

        return [
            (item, criterion_type)
            for criterion_type, section_text in sections
            for item in self.split_atomic_criteria(section_text, use_numbered_bullets)
        ]

    def parse_eligibility_criteria(self, raw: list[dict], nct_id: str) -> list[tuple[str, str]]:
        """Normalize a CTRP eligibility response into atomic criterion rows.

        CTRP returns eligibility criteria in one of two shapes:

        * Multiple entries, each already an atomic criterion with its own
          ``display_order`` and ``inclusion_indicator`` — these are simply
          sorted and relabeled.
        * A single entry containing one undifferentiated block of text with
          "Inclusion Criteria:"/"Exclusion Criteria:" headers embedded in the
          prose — this is delegated to `parse_unstructured_block`, and the
          entry's own ``inclusion_indicator`` is ignored since it's
          effectively meaningless for a mixed block.

        Either way, the resulting pairs are passed through
        `clean_parsed_criteria` before being returned, which strips any
        glued-on all-caps sub-header (e.g. "CAREGIVER:") from each
        criterion's text and drops any criterion that turns out to be
        nothing but such a header (e.g. "PRACTICE ELIGIBILITY:").

        Args:
            raw: The raw criterion dicts as returned by
                `fetch_eligibility_criteria` (i.e.
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
            parsed = self.parse_unstructured_block(raw[0]["description"], nct_id)
        else:
            ordered = sorted(raw, key=lambda criterion: criterion["display_order"])
            parsed = [
                (c["description"], "inclusion" if c["inclusion_indicator"] else "exclusion")
                for c in ordered
            ]

        return self.clean_parsed_criteria(parsed, nct_id)

    def strip_allcaps_header(self, text: str) -> str:
        """Remove a leading all-caps section-header label from criterion text.

        CTRP sometimes glues a domain-specific sub-header (e.g.
        "BREAST CANCER SURVIVORS:", "CAREGIVER:") directly onto the front of
        a criterion's text, or returns a criterion row that is nothing but
        such a header with no actual content. Unlike `HEADER_PATTERN`, which
        only recognizes headers containing the words "inclusion"/"exclusion"
        and is used to split a whole block into sections, this recognizes
        *any* all-caps "LABEL:" prefix and is used to clean up individual
        already-split criterion strings. This strips exactly one leading
        match of `ALLCAPS_HEADER_PREFIX_PATTERN` from `text`, so a
        glued-prefix criterion is reduced to its real content, and a
        header-only row is reduced to an empty string so the caller can drop
        it.

        Args:
            text: A single criterion's raw text.

        Returns:
            `text` with any leading all-caps "LABEL:" prefix removed.
            Returns an empty string if `text` was nothing but such a label.
        """
        return self.ALLCAPS_HEADER_PREFIX_PATTERN.sub("", text, count=1)

    def clean_parsed_criteria(
        self, parsed: list[tuple[str, str]], nct_id: str
    ) -> list[tuple[str, str]]:
        """Strip glued all-caps headers and drop header-only criterion rows.

        Applies `strip_allcaps_header` to every criterion's text. A
        criterion that was nothing but a header (e.g.
        "PRACTICE ELIGIBILITY:") strips down to an empty string and is
        dropped; a criterion with a glued prefix (e.g. "CAREGIVER: A primary
        caregiver...") keeps its real content with the prefix removed.

        Args:
            parsed: ``(criterion_text, criterion_type)`` pairs as produced
                by either branch of `parse_eligibility_criteria`.
            nct_id: The trial's NCT ID, used only for log messages.

        Returns:
            The same pairs with headers stripped from ``criterion_text`` and
            any now-empty entries removed.
        """
        cleaned: list[tuple[str, str]] = []
        dropped = 0
        for criterion_text, criterion_type in parsed:
            stripped = self.strip_allcaps_header(criterion_text).strip()
            if not stripped:
                dropped += 1
                continue
            cleaned.append((stripped, criterion_type))

        if dropped:
            logger.info("%s: dropped %d header-only criterion row(s)", nct_id, dropped)
        return cleaned

    # ------------------------------------------------------------------
    # Output assembly
    # ------------------------------------------------------------------

    def build_criterion_records(
        self, trial: dict, parsed_criteria: list[tuple[str, str]]
    ) -> list[dict]:
        """Combine trial metadata with its parsed criteria into output rows.

        Args:
            trial: A trial summary dict with ``"nct_id"`` and
                ``"study_first_posted_date"`` keys, as produced by
                `fetch_matching_trials`.
            parsed_criteria: The ``(criterion_text, criterion_type)`` pairs
                for this trial, as produced by `parse_eligibility_criteria`.

        Returns:
            One dict per criterion, each with keys ``"nct_id"``,
            ``"study_first_posted_date"``, ``"criterion_text"``, and
            ``"criterion_type"``, ready to be written by
            `write_eligibility_csv`.
        """
        return [
            {
                "nct_id": trial["nct_id"],
                "study_first_posted_date": trial["study_first_posted_date"],
                "criterion_text": criterion_text,
                "criterion_type": criterion_type,
            }
            for criterion_text, criterion_type in parsed_criteria
        ]

    def fetch_trials_with_eligibility(self, condition: str) -> tuple[list[dict], list[dict]]:
        """Fetch matching trials and their parsed eligibility criteria together.

        Calls `fetch_matching_trials` to find candidate trials, then calls
        `fetch_eligibility_criteria` and `parse_eligibility_criteria` for each
        one, dropping any trial for which no criteria could be retrieved or
        parsed.

        Args:
            condition: Condition to search for, e.g. "breast cancer".

        Returns:
            A ``(trials, records)`` tuple:
                trials: All trials returned by `fetch_matching_trials`
                    (including ones later dropped for lacking criteria).
                records: Flattened criterion rows (see
                    `_build_criterion_records`) across all trials that had
                    retrievable criteria.
        """
        trials = self.fetch_matching_trials(condition)

        records: list[dict] = []
        trials_with_criteria = 0
        for trial in trials:
            nct_id = trial["nct_id"]
            raw_criteria = self.fetch_eligibility_criteria(nct_id)
            parsed = self.parse_eligibility_criteria(raw_criteria, nct_id) if raw_criteria else []

            if not parsed:
                logger.info("%s: no eligibility criteria from CTRP API, dropping trial", nct_id)
                continue

            trials_with_criteria += 1
            records.extend(self.build_criterion_records(trial, parsed))

        logger.info(
            "fetch_trials_with_eligibility: %d trials found, %d with eligibility criteria "
            "retrieved/parsed (%d criterion rows)",
            len(trials),
            trials_with_criteria,
            len(records),
        )
        return trials, records

    # ------------------------------------------------------------------
    # CSV output
    # ------------------------------------------------------------------

    def write_eligibility_csv(self, records: list[dict], output_path: Path) -> None:
        """Write parsed eligibility criteria records to a CSV file.

        Creates any missing parent directories for `output_path`. Writes one
        row per criterion with a header row.

        Args:
            records: Criterion row dicts as produced by
                `_build_criterion_records`, each with keys ``"nct_id"``,
                ``"study_first_posted_date"``, ``"criterion_text"``, and
                ``"criterion_type"``.
            output_path: Destination CSV file path.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "nct_id",
                    "study_first_posted_date",
                    "criterion_text",
                    "criterion_type",
                ],
            )
            writer.writeheader()
            writer.writerows(records)

    @staticmethod
    def condition_to_slug(condition: str) -> str:
        """Convert a free-text condition into a filesystem-safe slug.

        Args:
            condition: Condition string as passed to `fetch_matching_trials`,
                e.g. "breast cancer".

        Returns:
            A lowercase slug with runs of non-alphanumeric characters
            collapsed to a single underscore and leading/trailing underscores
            stripped, e.g. "breast_cancer".
        """
        return re.sub(r"[^a-z0-9]+", "_", condition.strip().lower()).strip("_")

    # ------------------------------------------------------------------
    # End-to-end pipeline
    # ------------------------------------------------------------------

    def run(self, condition: str, output_path: Path | None = None) -> Path:
        """Run the full fetch-parse-write pipeline for one condition.

        Args:
            condition: Condition to search for, e.g. "breast cancer".
            output_path: Where to write the resulting CSV. If omitted,
                defaults to
                ``OUTPUT_DIR / f"{condition_to_slug(condition)}_trials_eligibility.csv"``.

        Returns:
            The path the CSV was written to.
        """
        if output_path is None:
            output_path = OUTPUT_DIR / f"{self.condition_to_slug(condition)}_trials_eligibility.csv"

        _, records = self.fetch_trials_with_eligibility(condition)
        self.write_eligibility_csv(records, output_path)

        logger.info("run: wrote %d criterion rows to %s", len(records), output_path)
        return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for this module.

    Returns:
        An ``argparse.ArgumentParser`` accepting a single positional
        ``condition`` argument.
    """
    parser = argparse.ArgumentParser(
        description="Fetch NCI-affiliated clinical trials and their eligibility criteria."
    )
    parser.add_argument(
        "condition",
        help='Condition to search for, e.g. "breast cancer" (passed as ClinicalTrials.gov query.cond)',
    )
    return parser


def main() -> None:
    """CLI entry point: fetch and write eligibility criteria for one condition."""
    logging.basicConfig(level=logging.INFO)

    args = build_arg_parser().parse_args()

    fetcher = EligibilityCriteriaFetcher()
    if not fetcher.api_key:
        raise RuntimeError(
            "CTRP_API_KEY is not set. Add it to a local .env file before running this script."
        )

    fetcher.run(args.condition)


if __name__ == "__main__":
    main()
