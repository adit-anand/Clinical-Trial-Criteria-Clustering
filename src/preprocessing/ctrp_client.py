"""Fetch a trial's unstructured eligibility criteria from the NCI CTRP API.

Wraps the NCI Clinical Trials Reporting Program (CTRP) API's
``/api/v2/trials`` endpoint behind `CTRPClient`, which retries on rate
limiting/server errors and paces successful requests to avoid tripping the
API's rate limit. Parsing the returned text into atomic criteria is a
separate concern handled by `criteria_parser.py`.
"""

import logging
import time

import requests

logger = logging.getLogger(__name__)


class CTRPClient:
    """Looks up trial eligibility criteria from the NCI CTRP API.

    An instance holds the HTTP session and API key used across calls, so a
    single client can be reused to look up many trials without
    re-authenticating or re-opening connections each time.
    """

    TRIALS_URL = "https://clinicaltrialsapi.cancer.gov/api/v2/trials"

    MAX_RETRIES = 3
    RETRY_BACKOFF_SECONDS = 2
    REQUEST_PACING_SECONDS = 0.15

    def __init__(self, api_key: str | None, session: requests.Session | None = None):
        """Initialize the client's HTTP session and CTRP credentials.

        Args:
            api_key: CTRP API key to send as the ``X-API-KEY`` header. May be
                `None` if the caller doesn't intend to call
                `fetch_eligibility_criteria` yet; it will raise if no key is
                available when it's actually called.
            session: A ``requests.Session`` to reuse for all HTTP calls (e.g.
                to share connection pooling with other code). If omitted, a
                new session is created.
        """
        self.api_key = api_key
        self.session = session or requests.Session()

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
            RuntimeError: If no CTRP API key was configured on this client.
            requests.HTTPError: If the API returns a non-retryable error
                status (anything other than 429/5xx).
        """
        if not self.api_key:
            raise RuntimeError(
                "No CTRP API key configured. Pass api_key= to "
                "CTRPClient() or set CTRP_API_KEY."
            )

        headers = {"X-API-KEY": self.api_key}
        params = {"nct_id": nct_id}

        for attempt in range(1, self.MAX_RETRIES + 1):
            response = self.session.get(
                self.TRIALS_URL, headers=headers, params=params
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
