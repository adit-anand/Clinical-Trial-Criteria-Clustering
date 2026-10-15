# amia_2026

Pipeline for building embeddings of clinical trial eligibility criteria:

1. **`src/preprocessing/criteria_retrieval.py`** — fetches NCI-affiliated
   clinical trials for a condition from ClinicalTrials.gov, retrieves their
   eligibility criteria from the NCI CTRP API, parses them into atomic
   inclusion/exclusion criteria, and writes them to a CSV. Each concern lives
   in its own module (`clinicaltrials_client.py`, `ctrp_client.py`,
   `criteria_parser.py`); `criteria_retrieval.py` composes them and exposes
   the CLI.
2. **`src/embeddings/criteria_embedding.py`** — reads that CSV, embeds each
   criterion's text with a configurable backend, and writes the result to a
   Parquet file.
3. **`src/clustering/kmeans_clustering.py`** — reads that Parquet file,
   clusters the embedding vectors via k-means, and writes cluster
   assignments plus clustering performance metrics.
4. **`src/clustering/hdbscan_clustering.py`** — reads the same Parquet file
   and clusters the embedding vectors via HDBSCAN (density-based clustering,
   no fixed cluster count), writing cluster assignments plus performance
   metrics in a parallel format. Read/validate/write logic shared between
   the two clustering approaches lives in `src/clustering/clustering_io.py`.
5. **`src/clustering/clustering_report.py`** — reads the metrics JSON already
   written by whichever of the clustering steps above have been run for a
   condition, and writes one consolidated Markdown report comparing their
   performance side by side.
6. **`src/clustering/hdbscan_sweep.py`** — tries multiple combinations of
   HDBSCAN's hyperparameters for a condition's existing embeddings, and
   writes the best-performing combination's cluster assignments and metrics
   as that condition's HDBSCAN output.

## Project structure

`src/preprocessing/`, `src/embeddings/`, and `src/clustering/` are real
Python packages (each has an `__init__.py`), so every pipeline module
imports its siblings as ordinary absolute package imports (e.g. `from
src.preprocessing.criteria_retrieval import EligibilityCriteriaFetcher`,
`from src.clustering import clustering_io`). The one constant every module
needs — the repo root — is centralized in `src/paths.py` rather than
re-derived in each file. Every entry-point script can be run either as a
module (`uv run python -m src.<package>.<module> "..."`, used throughout
this README) or directly (`uv run src/<package>/<module>.py "..."`) — a
small bootstrap at the top of each one puts the repo root on `sys.path`
when it detects direct execution, so both styles behave identically.

```
src/
├── paths.py                              # shared REPO_ROOT constant
├── preprocessing/
│   ├── clinicaltrials_client.py          # ClinicalTrials.gov search
│   ├── ctrp_client.py                    # CTRP API fetch + retry
│   ├── criteria_parser.py                # unstructured-text -> atomic criteria
│   └── criteria_retrieval.py             # orchestration + CLI (step 1)
├── embeddings/
│   ├── embedding_model.py                # EmbeddingModel interface + backends
│   └── criteria_embedding.py             # orchestration + CLI (step 2)
└── clustering/
    ├── clustering_io.py                  # shared read/write logic
    ├── kmeans_clustering.py              # step 3
    ├── hdbscan_clustering.py             # step 4
    ├── clustering_report.py              # step 5
    └── hdbscan_sweep.py                  # step 6
```

Within `src/clustering/`, `clustering_io.py` holds everything genuinely
common to both clustering algorithms (reading/validating embeddings,
resolving default I/O paths, writing output); `kmeans_clustering.py` and
`hdbscan_clustering.py` each hold only their own algorithm's fitting and
metrics logic, so adding a third clustering algorithm means adding one new
sibling module, not touching the shared I/O layer or the other algorithms.
`clustering_report.py` and `hdbscan_sweep.py` build on top of that same
layer without modifying it.

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
uv run python -m src.preprocessing.criteria_retrieval "breast cancer"
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
uv run python -m src.embeddings.criteria_embedding "breast cancer" --backend huggingface
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

## 3. Clustering criteria

```bash
uv run python -m src.clustering.kmeans_clustering "breast cancer"
```

This reads the Parquet file produced by the step above, drops rows whose
embedding failed, and clusters the remaining embedding vectors via k-means.
By default, the number of clusters (`k`) is **selected automatically**:
k-means is fit once per candidate `k` in a search range (`--k-min` to
`--k-max`, default `2` to `min(100, floor(sqrt(n_samples / 2)), n_samples - 1)`),
and the `k` with the highest silhouette score is kept. Pass `--n-clusters` to
skip the search and force a specific `k` instead. Either way, it writes:

```
data/clustering/kmeans/<condition_slug>_trials_clusters.parquet
data/clustering/kmeans/<condition_slug>_trials_cluster_metrics.json
```

The Parquet file has one row per successfully-embedded criterion, containing
`nct_id`, `study_first_posted_date`, `criterion_text`, `criterion_type`,
`cluster_id`, `embedding_backend`, `embedding_model`, and `clustered_at`. The
JSON file reports run provenance (sample/cluster counts, k-means parameters)
plus clustering performance metrics: inertia, silhouette score, the
Davies-Bouldin index, and the Calinski-Harabasz index. The silhouette,
Davies-Bouldin, and Calinski-Harabasz scores require between 2 and
`n_samples - 1` distinct clusters to be computable; if a run falls outside
that range (e.g. due to empty-cluster collapse), those three metrics are
reported as `null` with an explanatory note and `metrics_valid` is `false`.
A `k_selection` object records whether `k` was `"auto"`-selected or `"fixed"`,
the search bounds used, and (in auto mode) the full per-`k` sweep results, so
the choice of `k` is auditable rather than a black box.

Re-running this step for the same condition overwrites both existing output
files; there is no append/skip-if-exists/merge behavior.

### Arguments

| Argument                     | Required | Description                                                                 |
| ----------------------------- | -------- | ---------------------------------------------------------------------------- |
| `condition`                    | yes      | Condition to cluster, e.g. `"breast cancer"`. Must match the condition passed to `criteria_embedding.py` (used to derive default input/output paths). |
| `--n-clusters`                 | no       | Fix the number of k-means clusters and skip automatic selection (default: auto-select via a silhouette-score sweep). |
| `--k-min`                      | no       | Minimum `k` to search when auto-selecting; ignored if `--n-clusters` is set (default: `2`). |
| `--k-max`                      | no       | Maximum `k` to search when auto-selecting; ignored if `--n-clusters` is set (default: `min(100, floor(sqrt(n_samples / 2)), n_samples - 1)`). |
| `--random-state`               | no       | Random seed for k-means initialization, for reproducibility (default: `42`). |
| `--n-init`                     | no       | Number of k-means initializations to run per `k` (default: `10`). |
| `--max-iter`                   | no       | Maximum k-means iterations per initialization (default: `300`). |
| `--input-path`                 | no       | Override the default input embeddings Parquet path (`data/embeddings/<condition_slug>_trials_embeddings.parquet`). |
| `--assignments-output-path`    | no       | Override the default output cluster-assignments Parquet path (`data/clustering/kmeans/<condition_slug>_trials_clusters.parquet`). |
| `--metrics-output-path`        | no       | Override the default output metrics JSON path (`data/clustering/kmeans/<condition_slug>_trials_cluster_metrics.json`). |

## 4. HDBSCAN clustering criteria

```bash
uv run python -m src.clustering.hdbscan_clustering "breast cancer"
```

This reads the same embeddings Parquet file as the k-means step, drops rows
whose embedding failed, and clusters the remaining embedding vectors via
HDBSCAN (density-based clustering). Unlike k-means, there is no cluster count
to choose — HDBSCAN discovers clusters of varying density directly from the
data, and explicitly labels points that don't belong to any cluster as noise
(`cluster_id == -1`). It writes:

```
data/clustering/hdbscan/<condition_slug>_trials_hdbscan_clusters.parquet
data/clustering/hdbscan/<condition_slug>_trials_hdbscan_cluster_metrics.json
```

The Parquet file has one row per successfully-embedded criterion, containing
`nct_id`, `study_first_posted_date`, `criterion_text`, `criterion_type`,
`cluster_id` (`-1` for noise), `cluster_probability` (HDBSCAN's cluster
membership strength, `0.0` for noise points), `embedding_backend`,
`embedding_model`, and `clustered_at`. The JSON file reports run provenance
(sample/cluster counts, HDBSCAN parameters, `noise_count`/`noise_fraction`)
plus clustering performance metrics — silhouette score, the Davies-Bouldin
index, and the Calinski-Harabasz index — computed only over non-noise points.
As with the k-means step, those three metrics require between 2 and
`n_non_noise - 1` distinct clusters among non-noise points to be computable;
if a run falls outside that range (e.g. an all-noise result), they are
reported as `null` with an explanatory note and `metrics_valid` is `false`.

Re-running this step for the same condition overwrites both existing output
files; there is no append/skip-if-exists/merge behavior.

### Arguments

| Argument                        | Required | Description                                                                 |
| --------------------------------- | -------- | ---------------------------------------------------------------------------- |
| `condition`                        | yes      | Condition to cluster, e.g. `"breast cancer"`. Must match the condition passed to `criteria_embedding.py` (used to derive default input/output paths). |
| `--min-cluster-size`               | no       | Minimum size of a group of points to be considered a cluster (default: `5`). |
| `--min-samples`                    | no       | Number of samples in a neighborhood for a point to be a core point (default: `min_cluster_size`). |
| `--metric`                         | no       | Distance metric to use (default: `euclidean`). |
| `--cluster-selection-epsilon`      | no       | Distance threshold below which clusters are merged (default: `0.0`). |
| `--cluster-selection-method`       | no       | `eom` (excess of mass) or `leaf` (default: `eom`). |
| `--alpha`                          | no       | Distance scaling parameter used in robust single linkage (default: `1.0`). |
| `--input-path`                     | no       | Override the default input embeddings Parquet path (`data/embeddings/<condition_slug>_trials_embeddings.parquet`). |
| `--assignments-output-path`        | no       | Override the default output cluster-assignments Parquet path (`data/clustering/hdbscan/<condition_slug>_trials_hdbscan_clusters.parquet`). |
| `--metrics-output-path`            | no       | Override the default output metrics JSON path (`data/clustering/hdbscan/<condition_slug>_trials_hdbscan_cluster_metrics.json`). |

## 5. Clustering performance report

```bash
uv run python -m src.clustering.clustering_report "breast cancer"
```

This reads whichever of the k-means/HDBSCAN metrics JSON files already exist
for the condition — it never re-runs clustering — and writes a single
consolidated Markdown report comparing their performance:

```
data/clustering/reports/<condition_slug>_clustering_report_<timestamp>.md
```

The report includes a summary table of each algorithm's availability, a
shared-metrics comparison table (silhouette score, Davies-Bouldin index,
Calinski-Harabasz index — the three metrics computed identically by both
algorithms), and a detailed section per algorithm covering its own
algorithm-specific fields (e.g. inertia and `k` selection for k-means; noise
count/fraction for HDBSCAN) plus its `metrics_valid`/`notes` status verbatim.
The report also calls out which algorithm has the best (highest) silhouette
score among those with valid metrics, since silhouette score is the only
metric comparable across every algorithm's schema.

If an algorithm hasn't been run yet for the condition, or its metrics file is
missing/unreadable, that algorithm's section clearly states so (naming the
exact path that was checked) instead of being silently omitted or aborting
the whole report. Each run writes a new timestamped file; prior reports for
the same condition are never overwritten or appended to.

### Arguments

| Argument       | Required | Description                                                                 |
| --------------- | -------- | ---------------------------------------------------------------------------- |
| `condition`      | yes      | Condition to report on, e.g. `"breast cancer"`. Must match the condition passed to `kmeans_clustering.py`/`hdbscan_clustering.py` (used to derive the metrics files to read). |
| `--output-dir`   | no       | Override the default report output directory (`data/clustering/reports`). |
| `--print`        | no       | Also print the generated report to stdout, in addition to writing the file. |

## 6. HDBSCAN hyperparameter sweep

```bash
uv run python -m src.clustering.hdbscan_sweep "breast cancer" --min-cluster-size 5,10,15 --alpha 0.8,1.0,1.2
```

This reads the same embeddings Parquet file as the ordinary HDBSCAN step and
fits HDBSCAN once per combination in the Cartesian product of the supplied
`--min-cluster-size`/`--min-samples`/`--cluster-selection-epsilon`/`--alpha`
value lists (`--metric`/`--cluster-selection-method` are held fixed across
the whole sweep, not swept). Each combination's clustering performance
metrics are logged as the sweep runs; no separate sweep-results file is
written. Once every combination has been tried, the single best one — by
relative validity index (DBCV, higher is better), with noise fraction (lower
is better) as a tiebreaker — has its cluster assignments and metrics written
to the condition's normal HDBSCAN output location, exactly as if
`hdbscan_clustering.py` had been run directly with those parameters:

```
data/clustering/hdbscan/<condition_slug>_trials_hdbscan_clusters.parquet
data/clustering/hdbscan/<condition_slug>_trials_hdbscan_cluster_metrics.json
```

The written metrics JSON has the same schema as an ordinary HDBSCAN run,
plus one extra field, `relative_validity_index`, recording the DBCV score
that decided the winner. A combination invalid for the dataset (e.g.
`--min-samples` exceeding the sample count) is logged as skipped rather than
aborting the rest of the sweep; a combination that fits but produces a
degenerate result (e.g. all noise) is logged but never eligible to win. If
no combination produces a usable relative validity index, nothing is
written. Re-running the sweep for the same condition overwrites the prior
HDBSCAN output, exactly as re-running `hdbscan_clustering.py` directly would.
The total number of combinations (the product of all four swept lists'
lengths) is capped at 50; exceeding it raises an error before any fitting is
attempted.

### Arguments

| Argument                        | Required | Description                                                                 |
| --------------------------------- | -------- | ---------------------------------------------------------------------------- |
| `condition`                        | yes      | Condition to sweep, e.g. `"breast cancer"`. Must match the condition passed to `criteria_embedding.py` (used to derive default input/output paths). |
| `--min-cluster-size`               | no       | Comma-separated list of `min_cluster_size` values to try (default: `5`). |
| `--min-samples`                    | no       | Comma-separated list of `min_samples` values to try (default: `5`). |
| `--cluster-selection-epsilon`      | no       | Comma-separated list of `cluster_selection_epsilon` values to try (default: `0.0`). |
| `--alpha`                          | no       | Comma-separated list of `alpha` values to try (default: `1.0`). |
| `--metric`                         | no       | Distance metric to use, held fixed across the sweep (default: `euclidean`). |
| `--cluster-selection-method`       | no       | `eom` or `leaf`, held fixed across the sweep (default: `eom`). |
| `--input-path`                     | no       | Override the default input embeddings Parquet path (`data/embeddings/<condition_slug>_trials_embeddings.parquet`). |
| `--assignments-output-path`        | no       | Override the default output cluster-assignments Parquet path (`data/clustering/hdbscan/<condition_slug>_trials_hdbscan_clusters.parquet`). |
| `--metrics-output-path`            | no       | Override the default output metrics JSON path (`data/clustering/hdbscan/<condition_slug>_trials_hdbscan_cluster_metrics.json`). |

### Example: full pipeline for one condition

```bash
uv run python -m src.preprocessing.criteria_retrieval "breast cancer"
uv run python -m src.embeddings.criteria_embedding "breast cancer" --backend huggingface
uv run python -m src.clustering.kmeans_clustering "breast cancer"
uv run python -m src.clustering.hdbscan_clustering "breast cancer"
uv run python -m src.clustering.clustering_report "breast cancer"
uv run python -m src.clustering.hdbscan_sweep "breast cancer" --min-cluster-size 5,10,15
```
