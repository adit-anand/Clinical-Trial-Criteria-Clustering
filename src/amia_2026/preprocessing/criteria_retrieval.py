"""Fetch NCI-affiliated clinical trials and their eligibility criteria.

Orchestrates the fetch-parse-write pipeline for one condition by composing:

* `clinicaltrials_client.ClinicalTrialsGovClient` — searches
  ClinicalTrials.gov for interventional trials sponsored/affiliated with the
  National Cancer Institute (NCI).
* `ctrp_client.CTRPClient` — looks up a matched trial's eligibility criteria
  from the NCI Clinical Trials Reporting Program (CTRP) API.
* `criteria_parser` — normalizes whatever shape CTRP returns (pre-split
  per-criterion entries, or one unstructured block of text) into atomic
  inclusion/exclusion criteria.

`EligibilityCriteriaFetcher` is the single public entry point other pipeline
modules depend on.
"""

import argparse
import csv
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from amia_2026.logging_utils import configure_logging
from amia_2026.paths import REPO_ROOT
from amia_2026.preprocessing.clinicaltrials_client import ClinicalTrialsGovClient
from amia_2026.preprocessing.criteria_parser import parse_eligibility_criteria
from amia_2026.preprocessing.ctrp_client import CTRPClient

logger = logging.getLogger(__name__)


class EligibilityCriteriaFetcher:
    """Fetches NCI trial eligibility criteria for a condition, end to end.

    Composes a `ClinicalTrialsGovClient` and a `CTRPClient`, sharing one HTTP
    session between them, and normalizes whatever CTRP returns into atomic
    criterion rows ready to write to CSV.

    Typical usage:
        fetcher = EligibilityCriteriaFetcher()
        output_path = fetcher.run("breast cancer")
    """

    def __init__(
        self,
        api_key: str | None = None,
        session=None,
        load_env: bool = True,
    ):
        """Initialize the fetcher's clients and CTRP credentials.

        Args:
            api_key: CTRP API key to send as the ``X-API-KEY`` header. If not
                provided, falls back to the ``CTRP_API_KEY`` environment
                variable. May be left unset if the caller only intends to use
                `fetch_matching_trials`, which does not require a key;
                `fetch_eligibility_criteria` will raise if no key is
                available when it's actually called.
            session: A ``requests.Session`` to reuse for all HTTP calls
                across both the ClinicalTrials.gov and CTRP clients. If
                omitted, a new session is created.
            load_env: If True, load variables from a ``.env`` file at the
                repo root (via ``python-dotenv``) before reading
                ``CTRP_API_KEY`` from the environment. Has no effect if
                `api_key` is given explicitly.
        """
        if load_env and api_key is None:
            load_dotenv(REPO_ROOT / ".env")
        self.api_key = api_key or os.environ.get("CTRP_API_KEY")

        self.ctgov_client = ClinicalTrialsGovClient(session=session)
        self.ctrp_client = CTRPClient(api_key=self.api_key, session=self.ctgov_client.session)

    def fetch_trials_with_eligibility(self, condition: str) -> tuple[list[dict], list[dict]]:
        """Fetch matching trials and their parsed eligibility criteria together.

        Calls the ClinicalTrials.gov client to find candidate trials, then
        calls the CTRP client and `criteria_parser.parse_eligibility_criteria`
        for each one, dropping any trial for which no criteria could be
        retrieved or parsed.

        Args:
            condition: Condition to search for, e.g. "breast cancer".

        Returns:
            A ``(trials, records)`` tuple:
                trials: All trials found by the ClinicalTrials.gov client
                    (including ones later dropped for lacking criteria).
                records: Flattened criterion rows (see
                    `build_criterion_records`) across all trials that had
                    retrievable criteria.
        """
        trials = self.ctgov_client.fetch_matching_trials(condition)

        records: list[dict] = []
        trials_with_criteria = 0
        for trial in trials:
            nct_id = trial["nct_id"]
            raw_criteria = self.ctrp_client.fetch_eligibility_criteria(nct_id)
            parsed = parse_eligibility_criteria(raw_criteria, nct_id) if raw_criteria else []

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

    def build_criterion_records(
        self, trial: dict, parsed_criteria: list[tuple[str, str]]
    ) -> list[dict]:
        """Combine trial metadata with its parsed criteria into output rows.

        Args:
            trial: A trial summary dict with ``"nct_id"`` and
                ``"study_first_posted_date"`` keys, as produced by
                `ClinicalTrialsGovClient.fetch_matching_trials`.
            parsed_criteria: The ``(criterion_text, criterion_type)`` pairs
                for this trial, as produced by
                `criteria_parser.parse_eligibility_criteria`.

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

    def write_eligibility_csv(self, records: list[dict], output_path: Path) -> None:
        """Write parsed eligibility criteria records to a CSV file.

        Creates any missing parent directories for `output_path`. Writes one
        row per criterion with a header row.

        Args:
            records: Criterion row dicts as produced by
                `build_criterion_records`, each with keys ``"nct_id"``,
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

    def run(self, condition: str, output_path: Path) -> Path:
        """Run the full fetch-parse-write pipeline for one condition.

        Args:
            condition: Condition to search for, e.g. "breast cancer".
            output_path: Where to write the resulting CSV.

        Returns:
            The path the CSV was written to.
        """
        _, records = self.fetch_trials_with_eligibility(condition)
        self.write_eligibility_csv(records, output_path)

        logger.info("run: wrote %d criterion rows to %s", len(records), output_path)
        return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for this module.

    Returns:
        An ``argparse.ArgumentParser`` accepting a positional ``condition``
        argument and a required ``--output-path`` flag.
    """
    parser = argparse.ArgumentParser(
        description="Fetch NCI-affiliated clinical trials and their eligibility criteria."
    )
    parser.add_argument(
        "condition",
        help='Condition to search for, e.g. "breast cancer" (passed as ClinicalTrials.gov query.cond)',
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Output CSV path to write the fetched eligibility criteria to",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="If given, also append logs to this file (in addition to the console)",
    )
    return parser


def main() -> None:
    """CLI entry point: fetch and write eligibility criteria for one condition."""
    args = build_arg_parser().parse_args()
    configure_logging(args.log_file)

    fetcher = EligibilityCriteriaFetcher()
    if not fetcher.api_key:
        raise RuntimeError(
            "CTRP_API_KEY is not set. Add it to a local .env file before running this script."
        )

    fetcher.run(args.condition, args.output_path)


if __name__ == "__main__":
    main()
