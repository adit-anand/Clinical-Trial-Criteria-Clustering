"""Search ClinicalTrials.gov for NCI-affiliated trials matching a condition.

Wraps the ClinicalTrials.gov ``/api/v2/studies`` endpoint behind
`ClinicalTrialsGovClient`, which pages through search results, filters out
false-positive sponsor matches, and de-duplicates by NCT ID. This is
search-only — looking up a matched trial's eligibility criteria is a
separate concern handled by `ctrp_client.py`.
"""

import logging

import requests

logger = logging.getLogger(__name__)


class ClinicalTrialsGovClient:
    """Searches ClinicalTrials.gov for NCI-affiliated trials matching a condition.

    An instance holds the HTTP session used across calls, so a single
    client can be reused to search many conditions without re-opening
    connections each time.
    """

    STUDIES_URL = "https://clinicaltrials.gov/api/v2/studies"

    SEARCH_FIELDS = "NCTId,StudyFirstPostDate,LeadSponsorName,CollaboratorName"
    SEARCH_PAGE_SIZE = 1000
    NCI_NAME_FRAGMENTS = ("national cancer institute", "nci")

    def __init__(self, session: requests.Session | None = None):
        """Initialize the client's HTTP session.

        Args:
            session: A ``requests.Session`` to reuse for all HTTP calls (e.g.
                to share connection pooling with other code). If omitted, a
                new session is created.
        """
        self.session = session or requests.Session()

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
            argument of a GET request to `STUDIES_URL`. Does not include a
            page token; callers add one when paginating.
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
        false-positive sponsor matches via `is_nci_affiliated`, and
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

            response = self.session.get(self.STUDIES_URL, params=page_params)
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
