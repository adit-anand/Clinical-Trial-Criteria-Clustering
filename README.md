# amia_2026

Pipeline for building embeddings of clinical trial eligibility criteria:

1. **`src/preprocessing/criteria_retrieval.py`** — fetches NCI-affiliated
   clinical trials for a condition from ClinicalTrials.gov, retrieves their
   eligibility criteria from the NCI CTRP API, parses them into atomic
   inclusion/exclusion criteria, and writes them to a CSV.
2. **`src/embeddings/criteria_embedding.py`** — reads that CSV, embeds each
   criterion's text with a configurable backend, and writes the result to a
   Parquet file.

## Setup

Requires Python >=3.14. Install dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Create a `.env` file at the repo root with your NCI CTRP API key (required by
`criteria_retrieval.py`):

```
CTRP_API_KEY=your-key-here
```

If you plan to embed with the `vllm` backend against a non-default server,
also set (optional, only needed to override the defaults):

```
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_API_KEY=your-key-here
```

If you plan to use a gated HuggingFace model with the `huggingface` backend,
also set:

```
HF_TOKEN=your-token-here
```

## 1. Fetching eligibility criteria

```bash
python -m src.preprocessing.criteria_retrieval "breast cancer"
```

This searches ClinicalTrials.gov for interventional trials sponsored/affiliated
with the National Cancer Institute matching the given condition, fetches each
trial's eligibility criteria from the CTRP API, parses them into atomic
inclusion/exclusion criteria, and writes:

```
data/eligibility_criteria/<condition_slug>_trials_eligibility.csv
```

with columns `nct_id`, `study_first_posted_date`, `criterion_text`,
`criterion_type` (`inclusion`, `exclusion`, or `both`).

### Arguments

| Argument    | Required | Description                                                       |
| ----------- | -------- | ------------------------------------------------------------------- |
| `condition` | yes      | Condition to search for, e.g. `"breast cancer"`. Passed through to ClinicalTrials.gov's `query.cond` parameter. |

## 2. Embedding criteria

```bash
python -m src.embeddings.criteria_embedding "breast cancer" --backend huggingface
```

This reads the CSV produced by the step above, embeds each row's
`criterion_text`, and writes:

```
data/embeddings/<condition_slug>_trials_embeddings.parquet
```

with one row per input criterion (including failed ones, so downstream steps
can load the file directly), containing the original columns plus `embedding`,
`embedding_success`, `embedding_dim`, `embedding_backend`, `embedding_model`,
and `embedded_at`.

Re-running this step for the same condition overwrites the existing output
file; there is no append/skip-if-exists/merge behavior.

### Arguments

| Argument         | Required | Description                                                                 |
| ----------------- | -------- | ---------------------------------------------------------------------------- |
| `condition`        | yes      | Condition to embed, e.g. `"breast cancer"`. Must match the condition passed to `criteria_retrieval.py` (used to derive default input/output paths). |
| `--backend`        | yes      | Embedding backend to use: `vllm` (calls an already-running vLLM server's OpenAI-compatible `/v1/embeddings` endpoint) or `huggingface` (runs a `sentence-transformers` model in-process). |
| `--model`           | no       | Override the backend's default model name. |
| `--input-path`      | no       | Override the default input CSV path (`data/eligibility_criteria/<condition_slug>_trials_eligibility.csv`). |
| `--output-path`     | no       | Override the default output Parquet path (`data/embeddings/<condition_slug>_trials_embeddings.parquet`). |
| `--batch-size`      | no       | Number of criteria to embed per backend call (default: `32`). |

### Backend notes

- **`vllm`** expects an externally-managed vLLM server already running an
  OpenAI-compatible embeddings endpoint (default `http://localhost:8000/v1`,
  default model `gpt-oss-20b`). Point it elsewhere with `VLLM_BASE_URL` /
  `VLLM_API_KEY` in `.env`, or `--model`.
- **`huggingface`** loads a `sentence-transformers` model in-process (default
  `Qwen/Qwen3-Embedding-8B`), which can be slow/memory-heavy on first run as
  the model downloads.

### Example: full pipeline for one condition

```bash
python -m src.preprocessing.criteria_retrieval "breast cancer"
python -m src.embeddings.criteria_embedding "breast cancer" --backend huggingface
```
