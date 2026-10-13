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
OUTPUT_DIR = REPO_ROOT / "data" / "external"

CLINICALTRIALS_STUDIES_URL = "https://clinicaltrials.gov/api/v2/studies"
CTRP_TRIALS_URL = "https://clinicaltrialsapi.cancer.gov/api/v2/trials"

SEARCH_FIELDS = "NCTId,StudyFirstPostDate,LeadSponsorName,CollaboratorName"
SEARCH_PAGE_SIZE = 1000
NCI_NAME_FRAGMENTS = ("national cancer institute", "nci")

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2
REQUEST_PACING_SECONDS = 0.15

# Header lines like "Inclusion Criteria:" or "Exclusion Criteria for fMRI component only:"
# that mark the start of a new section within an unstructured criteria block. Lines that
# themselves start with a bullet/numbered marker are excluded so bullet text mentioning
# "inclusion"/"exclusion" isn't mistaken for a header.
HEADER_PATTERN = re.compile(
    r"^(?!\s*(?:[-•]|\d+\.)\s)\s*.*\b(inclusion|exclusion)\b.*:\s*$",
    re.IGNORECASE,
)
NUMBERED_BULLET_PATTERN = re.compile(r"^\d+\.\s*")
UNORDERED_BULLET_PATTERN = re.compile(r"^[-•]\s*")

logger = logging.getLogger(__name__)

class EligibilityCriteriaFetcher:

    def __init__(self):
        self.CT_GOV_BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
        self.CTRP_BASE_URL = "https://clinicaltrialsapi.cancer.gov/api/v2/trials"
        self.CTRP_API_KEY = os.environ.get()
        self.SEARCH_PAGE_SIZE = 1000
        self.SEARCH_FIELDS = "NCTId,StudyFirstPostDate,LeadSponsorName,CollaboratorName"
        print("Fetcher")

    def construct_ct_gov_query(self, search_term: str, search_params: dict) -> str:
        # If no search filters were provided, then use the default filters for trial status [Active, Not Recruiting or Completed), trial first posted date (after 01/1/2014), Interventional trial, and sponsored by the NCI]
        if not search_params:
            search_params = {
                "query.cond": search_term,
                "query.spons": "National Cancer Institute",
                "filter.overallStatus": "ACTIVE_NOT_RECRUITING,COMPLETED",
                "filter.advanced": "AREA[StudyType]INTERVENTIONAL",
                "query.term": "AREA[StudyFirstPostDate]RANGE[2014-01-01,MAX]",
                "fields": SEARCH_FIELDS,
                "pageSize": SEARCH_PAGE_SIZE,
                "countTotal": "true",
            }
        request_query = f"{self.CT_BASE_URL}{search_term}{search_filters}"
        return request_query

    def search_for_trials(self, search_term: str):
        self.construct_ct_gov_query(search_term = search_term)

def fetch_matching_trials(session: requests.Session, condition: str) -> list[dict]:
    """Page through ClinicalTrials.gov for interventional trials matching the given
    condition, sponsored/affiliated with the NCI, posted after 2014-01-01, with status
    Active-not-recruiting or Completed."""
    params = {
        "query.cond": condition,
        "query.spons": "National Cancer Institute",
        "filter.overallStatus": "ACTIVE_NOT_RECRUITING,COMPLETED",
        "filter.advanced": "AREA[StudyType]INTERVENTIONAL",
        "query.term": "AREA[StudyFirstPostDate]RANGE[2014-01-01,MAX]",
        "fields": SEARCH_FIELDS,
        "pageSize": SEARCH_PAGE_SIZE,
        "countTotal": "true",
    }

    trials_by_nct_id: dict[str, dict] = {}
    reported_total: int | None = None
    page_token: str | None = None

    while True:
        page_params = dict(params)
        if page_token:
            page_params["pageToken"] = page_token

        response = session.get(CLINICALTRIALS_STUDIES_URL, params=page_params)
        response.raise_for_status()
        payload = response.json()

        if reported_total is None:
            reported_total = payload.get("totalCount")

        for study in payload.get("studies", []):
            protocol = study["protocolSection"]
            nct_id = protocol["identificationModule"]["nctId"]
            posted_date = protocol["statusModule"]["studyFirstPostDateStruct"]["date"]
            sponsor_module = protocol.get("sponsorCollaboratorsModule", {})

            sponsor_names = [sponsor_module.get("leadSponsor", {}).get("name", "")]
            sponsor_names.extend(
                collaborator.get("name", "")
                for collaborator in sponsor_module.get("collaborators", [])
            )
            if not any(
                fragment in name.lower()
                for name in sponsor_names
                for fragment in NCI_NAME_FRAGMENTS
            ):
                logger.info(
                    "%s: dropping false positive, no NCI sponsor/collaborator match in %s",
                    nct_id,
                    sponsor_names,
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


def fetch_eligibility_criteria(
    session: requests.Session, api_key: str, nct_id: str
) -> list[dict] | None:
    """Look up a single trial's unstructured eligibility criteria from the NCI CTS API."""
    headers = {"X-API-KEY": api_key}
    params = {"nct_id": nct_id}

    for attempt in range(1, MAX_RETRIES + 1):
        response = session.get(CTRP_TRIALS_URL, headers=headers, params=params)

        if response.status_code == 429 or response.status_code >= 500:
            logger.warning(
                "%s: CTRP API returned %d (attempt %d/%d), retrying",
                nct_id,
                response.status_code,
                attempt,
                MAX_RETRIES,
            )
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
            continue

        response.raise_for_status()
        payload = response.json()
        time.sleep(REQUEST_PACING_SECONDS)

        data = payload.get("data", [])
        if not data:
            logger.warning("%s: no eligibility data returned by CTRP API", nct_id)
            return None
        return data[0].get("eligibility", {}).get("unstructured", [])

    logger.warning("%s: giving up on CTRP API after %d attempts", nct_id, MAX_RETRIES)
    return None


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split a block of text on embedded "Inclusion Criteria:"/"Exclusion Criteria:"
    headers, returning a list of (criterion_type, section_text) pairs in document order.

    Lines before the first header (if any) are discarded as preamble, not criteria.
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


def _split_atomic_criteria(text: str, use_numbered_bullets: bool) -> list[str]:
    """Split a section's text into atomic criteria using a single bullet style.

    Lines that don't start with a bullet are treated as a continuation of the
    previous criterion (clinicaltrials.gov wraps long criteria across lines).
    If no bullets of the chosen style are present, the whole text is one criterion.
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


def parse_eligibility_criteria(raw: list[dict], nct_id: str) -> list[tuple[str, str]]:
    """Split CTRP's unstructured criteria list into (criterion_text, criterion_type) pairs.

    A single-element list means CTRP returned the criteria as one undifferentiated
    block of text (inclusion/exclusion headers embedded in the text itself, e.g. NCT02111850)
    rather than pre-split per-criterion entries. In that case the entry's
    inclusion_indicator is ignored (it's effectively meaningless here) and instead each
    criterion is typed by the "Inclusion Criteria:"/"Exclusion Criteria:" header it falls
    under, then split into atomic criteria on that block's numbered ("1.", "2.") or
    unordered ("-", "•") bullets, preferring numbered bullets when both styles appear.
    """
    if not raw:
        logger.warning("%s: empty eligibility criteria list", nct_id)
        return []

    if len(raw) == 1:
        text = raw[0]["description"]
        use_numbered_bullets = any(
            NUMBERED_BULLET_PATTERN.match(line.strip())
            for line in text.splitlines()
            if line.strip()
        )
        sections = _split_into_sections(text)
        if not sections:
            logger.warning(
                "%s: unstructured eligibility text has no inclusion/exclusion headers",
                nct_id,
            )
            return [
                (item, "both")
                for item in _split_atomic_criteria(text, use_numbered_bullets)
            ]
        return [
            (item, criterion_type)
            for criterion_type, section_text in sections
            for item in _split_atomic_criteria(section_text, use_numbered_bullets)
        ]

    ordered = sorted(raw, key=lambda criterion: criterion["display_order"])
    return [
        (c["description"], "inclusion" if c["inclusion_indicator"] else "exclusion")
        for c in ordered
    ]


def write_eligibility_csv(records: list[dict], output_path: Path) -> None:
    """Write one row per criterion, with its NCT ID, posted date, text, and type."""
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch NCI-affiliated clinical trials and their eligibility criteria."
    )
    parser.add_argument(
        "condition",
        help='Condition to search for, e.g. "breast cancer" (passed as ClinicalTrials.gov query.cond)',
    )
    return parser


def condition_to_slug(condition: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", condition.strip().lower()).strip("_")


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    args = build_arg_parser().parse_args()

    load_dotenv(REPO_ROOT / ".env")
    api_key = os.environ.get("CTRP_API_KEY")
    if not api_key:
        raise RuntimeError(
            "CTRP_API_KEY is not set. Add it to a local .env file before running this script."
        )

    session = requests.Session()

    trials = fetch_matching_trials(session, args.condition)

    records = []
    trials_with_criteria = 0
    for trial in trials:
        nct_id = trial["nct_id"]
        raw_criteria = fetch_eligibility_criteria(session, api_key, nct_id)
        parsed = parse_eligibility_criteria(raw_criteria, nct_id) if raw_criteria else []

        if not parsed:
            logger.info(
                "%s: no eligibility criteria from CTRP API, dropping trial", nct_id
            )
            continue

        trials_with_criteria += 1
        for criterion_text, criterion_type in parsed:
            records.append(
                {
                    "nct_id": nct_id,
                    "study_first_posted_date": trial["study_first_posted_date"],
                    "criterion_text": criterion_text,
                    "criterion_type": criterion_type,
                }
            )

    output_path = OUTPUT_DIR / f"{condition_to_slug(args.condition)}_trials_eligibility.csv"
    write_eligibility_csv(records, output_path)

    logger.info(
        "main: %d trials found, %d with eligibility criteria retrieved/parsed (%d criterion rows), written to %s",
        len(trials),
        trials_with_criteria,
        len(records),
        output_path,
    )


if __name__ == "__main__":
    main()
