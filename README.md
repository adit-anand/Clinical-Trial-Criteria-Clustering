# amia_2026

Pipeline for building embeddings of clinical trial eligibility criteria:

Every step below takes its input and output file paths as explicit CLI
flags — there is no "condition" argument used to derive default paths
anywhere except step 1, where it's the actual ClinicalTrials.gov search
term. Chaining steps together means passing the previous step's output path
as the next step's input path.

1. **`src/preprocessing/criteria_retrieval.py`** — fetches NCI-affiliated
   clinical trials for a condition from ClinicalTrials.gov, retrieves their
   eligibility criteria from the NCI CTRP API, parses them into atomic
   inclusion/exclusion criteria, and writes them to the given CSV path. Each
   concern lives in its own module (`clinicaltrials_client.py`,
   `ctrp_client.py`, `criteria_parser.py`); `criteria_retrieval.py` composes
   them and exposes the CLI.
2. **`src/embeddings/criteria_embedding.py`** — reads that CSV, embeds each
   criterion's text with a configurable backend, and writes the result to
   the given Parquet path.
3. **`src/clustering/kmeans_clustering.py`** — reads that Parquet file,
   clusters the embedding vectors via k-means, sweeping the Cartesian
   product of its `k_min`/`k_max`/`random_state` value lists (a single
   value per flag is a sweep of size one) and writing the best-performing
   combination's cluster assignments plus performance metrics.
4. **`src/clustering/clustering_report.py`** — reads whichever of the
   clustering steps' metrics JSON paths are passed to it, and writes one
   consolidated Markdown report comparing their performance side by side.
5. **`src/clustering/umap_hdbscan_clustering.py`** — reduces embeddings with
   UMAP before clustering the reduced vectors with HDBSCAN, both tuned
   jointly by one Optuna search (search space/objective weights/trial
   budget come from a `--config-path` YAML file, not CLI flags; each trial
   also runs a configured seeds/repeats-per-seed grid internally to pick
   the most reproducible seed), writing the reduced embeddings plus cluster
   assignments and metrics. Read/validate/write logic shared between the
   clustering approaches lives in `src/clustering/clustering_io.py`; the
   HDBSCAN fitting/metrics code and Optuna search engine shared by both
   reducer-based steps live in `src/clustering/hdbscan_model.py` and
   `src/clustering/optuna_search.py`.
6. **`src/clustering/pacmap_hdbscan_clustering.py`** — the same idea as step
   5, reducing with PaCMAP instead of UMAP. The seed/repeat reproducibility
   mechanics both steps share live in `src/clustering/reduction_sweep.py`.
7. **`src/clustering/clustering_scatterplot.py`** — reads whichever
   clustering method's output paths are passed to it and writes an
   interactive HTML scatterplot of its criteria in a 2-dimensional UMAP
   space (colored by cluster), reusing that method's own optimal-run seed
   for the projection.

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
config/
├── pacmap_hdbscan_optuna.yaml             # Optuna search config for step 6
└── umap_hdbscan_optuna.yaml               # Optuna search config for step 5
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
    ├── hdbscan_model.py                  # shared HDBSCAN fitting/metrics (steps 5-6)
    ├── optuna_search.py                  # shared Optuna search engine (steps 5-6)
    ├── clustering_report.py              # step 4
    ├── reduction_sweep.py                # shared seed/repeat reproducibility engine (steps 5-6)
    ├── umap_hdbscan_clustering.py        # step 5
    ├── pacmap_hdbscan_clustering.py      # step 6
    └── clustering_scatterplot.py         # step 7
```

Within `src/clustering/`, `clustering_io.py` holds everything genuinely
common to every clustering algorithm (reading/validating embeddings,
writing output); each algorithm module (`kmeans_clustering.py`,
`umap_hdbscan_clustering.py`, `pacmap_hdbscan_clustering.py`) holds only its
own fitting and metrics logic, so adding another clustering algorithm means
adding one new sibling module, not touching the shared I/O layer or the
other algorithms. `clustering_report.py` builds on top of that same layer
without modifying it. `umap_hdbscan_clustering.py` and
`pacmap_hdbscan_clustering.py` share `hdbscan_model.py` (the HDBSCAN
fitting/metrics class both reduce into, since standalone HDBSCAN clustering
is not a pipeline in this repo — only these two reducer-based steps use
it) and `optuna_search.py` — the Optuna config loader, hyperparameter
suggestion, composite objective scoring, and study-lifecycle/trial-history
mechanics common to both, so neither re-implements it. They additionally
share `reduction_sweep.py`, which holds the reducer-agnostic seed/repeat
reproducibility mechanics each of their Optuna trials runs internally
(picking the most reproducible seed, then a representative repeat) so
neither module re-implements it.
`clustering_scatterplot.py` reads whichever of the three algorithms' output
already exists (via its own `VISUALIZATION_SPECS` registry, one entry per
algorithm, following the same "no branching elsewhere" pattern
`clustering_report.py`'s `ALGORITHM_SPECS` already established) and never
re-runs or modifies any of them.

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
uv run python -m src.preprocessing.criteria_retrieval "breast cancer" \
  --output-path data/eligibility_criteria/breast_cancer_trials_eligibility.csv
```

This searches ClinicalTrials.gov for interventional trials sponsored/affiliated
with the National Cancer Institute matching the given condition, fetches each
trial's eligibility criteria from the CTRP API, parses them into atomic
inclusion/exclusion criteria, and writes them to `--output-path` with columns
`nct_id`, `study_first_posted_date`, `criterion_text`, `criterion_type`
(`inclusion`, `exclusion`, or `both`).

### Arguments

| Argument    | Required | Description                                                       |
| ----------- | -------- | ------------------------------------------------------------------- |
| `condition` | yes      | Condition to search for, e.g. `"breast cancer"`. Passed through to ClinicalTrials.gov's `query.cond` parameter. |
| `--output-path` | yes  | Output CSV path to write the fetched eligibility criteria to. |

## 2. Embedding criteria

```bash
uv run python -m src.embeddings.criteria_embedding \
  --input-path data/eligibility_criteria/breast_cancer_trials_eligibility.csv \
  --output-path data/embeddings/breast_cancer_trials_embeddings.parquet \
  --backend huggingface
```

This reads the CSV written by the step above from `--input-path`, embeds
each row's `criterion_text`, and writes `--output-path` with one row per
input criterion (including failed ones, so downstream steps can load the
file directly), containing the original columns plus `embedding`,
`embedding_success`, `embedding_dim`, `embedding_backend`, `embedding_model`,
and `embedded_at`.

Re-running this step against the same output path overwrites the existing
file; there is no append/skip-if-exists/merge behavior.

### Arguments

| Argument         | Required | Description                                                                 |
| ----------------- | -------- | ---------------------------------------------------------------------------- |
| `--backend`        | yes      | Embedding backend to use: `vllm` (calls an already-running vLLM server's OpenAI-compatible `/v1/embeddings` endpoint) or `huggingface` (runs a `sentence-transformers` model in-process). |
| `--model`           | no       | Override the backend's default model name. |
| `--input-path`      | yes      | Input criteria CSV path (as written by `criteria_retrieval.py`). |
| `--output-path`     | yes      | Output Parquet path to write embeddings to. |
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
uv run python -m src.clustering.kmeans_clustering \
  --input-path data/embeddings/breast_cancer_trials_embeddings.parquet \
  --assignments-output-path data/clustering/kmeans/breast_cancer_trials_clusters.parquet \
  --metrics-output-path data/clustering/kmeans/breast_cancer_trials_cluster_metrics.json \
  --k-min 10,15 --k-max 50,100 --random-state 1,2,3
```

This reads the Parquet file given via `--input-path`, drops rows whose
embedding failed, and clusters the remaining embedding vectors via k-means,
sweeping the Cartesian product of the supplied `--k-min`/`--k-max`/
`--random-state` value lists (a single value per flag — the default — is
simply a sweep of size one). For each `(k_min, k_max, random_state)`
combination, the number of clusters (`k`) is **auto-selected**: k-means is
fit once per candidate `k` in `[k_min, k_max]` with that `random_state`, and
the `k` with the highest silhouette score is kept as that combination's
result. The single overall winner across every combination is then whichever
one's chosen `k` has the highest silhouette score of all. Pass `--n-clusters`
to skip `k`-selection entirely and fix `k` — in that mode, `--k-min`/
`--k-max` are ignored and only `--random-state` is swept, over that fixed
`k`. Either way, it writes the winning combination's cluster assignments and
metrics to `--assignments-output-path`/`--metrics-output-path`.

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
the winning combination's search bounds, and its full per-`k` sweep results;
a `combination_search` object separately records every `(k_min, k_max,
random_state)` (or `random_state`-only, in fixed mode) combination tried —
each with its own chosen `k` and silhouette score — so the choice across the
whole sweep is auditable rather than a black box. A combination whose
`[k_min, k_max]` range is invalid for the dataset (e.g. `k_max < k_min`) is
skipped with a logged reason rather than aborting the rest of the sweep; if
no combination produces a usable silhouette score, nothing is written. The
total number of combinations is capped at 100; exceeding it raises an error
before any fitting is attempted.

Re-running this step against the same output paths overwrites both existing
output files; there is no append/skip-if-exists/merge behavior.

`--metric` controls the distance metric, held fixed across the sweep.
sklearn's `KMeans` has no pluggable metric of its own — Lloyd's algorithm's
centroid-averaging step is only mathematically well-defined under squared
Euclidean distance (`--metric euclidean`, the default). `--metric cosine` is
still offered because it has an exact equivalence: embedding vectors are
L2-normalized to unit length *before* fitting ordinary k-means ("spherical
k-means") — squared Euclidean distance between unit vectors is a strictly
monotonic function of their cosine distance, so this gives identical cluster
assignments to a true cosine-distance k-means, not an approximation. No
other metric (e.g. manhattan) has an analogous equivalence, so none is
offered. `kmeans_params.metric` in the metrics JSON records which was used.

### Arguments

| Argument                     | Required | Description                                                                 |
| ----------------------------- | -------- | ---------------------------------------------------------------------------- |
| `--n-clusters`                 | no       | Fix the number of k-means clusters and skip automatic selection (`--k-min`/`--k-max` are ignored if set; default: auto-select via a silhouette-score sweep). |
| `--k-min`                      | no       | Comma-separated list of `k_min` values to sweep when auto-selecting; ignored if `--n-clusters` is set (default: `2`). |
| `--k-max`                      | no       | Comma-separated list of `k_max` values to sweep when auto-selecting; ignored if `--n-clusters` is set (default: a single auto-computed value, `min(100, floor(sqrt(n_samples / 2)), n_samples - 1)`). |
| `--random-state`               | no       | Comma-separated list of random seeds to sweep for k-means initialization (default: `42`). |
| `--n-init`                     | no       | Number of k-means initializations to run per `k`, held fixed across the sweep (default: `10`). |
| `--max-iter`                   | no       | Maximum k-means iterations per initialization, held fixed across the sweep (default: `300`). |
| `--metric`                     | no       | Distance metric, held fixed across the sweep: `euclidean` or `cosine` (see above; default: `euclidean`). |
| `--input-path`                 | yes      | Input embeddings Parquet path (as written by `criteria_embedding.py`). |
| `--assignments-output-path`    | yes      | Output cluster-assignments Parquet path. |
| `--metrics-output-path`        | yes      | Output metrics JSON path. |

## 4. Clustering performance report

```bash
uv run python -m src.clustering.clustering_report \
  --output-path data/clustering/reports/breast_cancer_clustering_report.md \
  --kmeans-metrics-path data/clustering/kmeans/breast_cancer_trials_cluster_metrics.json \
  --umap-hdbscan-metrics-path data/clustering/umap_hdbscan/breast_cancer_trials_umap_hdbscan_cluster_metrics.json
```

This reads whichever of the k-means/UMAP+HDBSCAN/PaCMAP+HDBSCAN metrics
JSON paths are passed to it (one optional flag per algorithm: any omitted
algorithm is reported as "not yet run") — it never re-runs clustering —
and writes a single consolidated Markdown report comparing their
performance to `--output-path`.

The report includes a summary table of each algorithm's availability, a
shared-metrics comparison table (silhouette score, Davies-Bouldin index,
Calinski-Harabasz index — the three metrics computed identically by every
algorithm), and a detailed section per algorithm covering its own
algorithm-specific fields (e.g. inertia and `k` selection for k-means; noise
count/fraction and winning config/seed for the two reducer-based steps)
plus its `metrics_valid`/`notes` status verbatim. The report also calls out
which algorithm has the best (highest) silhouette score among those with
valid metrics, since silhouette score is the only metric comparable across
every algorithm's schema.

If an algorithm's metrics-path flag was omitted, or the given file is
missing/unreadable, that algorithm's section clearly states so (naming the
flag to pass, or the exact path that was checked) instead of being silently
omitted or aborting the whole report. Each run overwrites `--output-path`
wholesale.

### Arguments

| Argument       | Required | Description                                                                 |
| --------------- | -------- | ---------------------------------------------------------------------------- |
| `--output-path` | yes      | Output Markdown report path. |
| `--kmeans-metrics-path`         | no | K-Means metrics JSON path (omit if not yet run). |
| `--umap-hdbscan-metrics-path`   | no | UMAP + HDBSCAN metrics JSON path (omit if not yet run). |
| `--pacmap-hdbscan-metrics-path` | no | PaCMAP + HDBSCAN metrics JSON path (omit if not yet run). |
| `--print`        | no       | Also print the generated report to stdout, in addition to writing the file. |

## 5. UMAP + HDBSCAN clustering

```bash
uv run python -m src.clustering.umap_hdbscan_clustering \
  --input-path data/embeddings/breast_cancer_trials_embeddings.parquet \
  --reduced-embeddings-output-path data/clustering/umap_hdbscan/breast_cancer_trials_umap_reduced_embeddings.parquet \
  --assignments-output-path data/clustering/umap_hdbscan/breast_cancer_trials_umap_hdbscan_clusters.parquet \
  --metrics-output-path data/clustering/umap_hdbscan/breast_cancer_trials_umap_hdbscan_cluster_metrics.json
```

This reads the same embeddings Parquet file format as the other clustering
steps (given via `--input-path`), drops rows whose embedding failed, and
runs an Optuna study whose every trial reduces the remaining embedding
vectors' dimensionality with UMAP before clustering the reduced vectors
with HDBSCAN. Each trial suggests a *combined* hyperparameter set — UMAP's
own (`n_components`/`n_neighbors`/`min_dist`) *and* HDBSCAN's own
(`min_cluster_size`/`min_samples`/`cluster_selection_epsilon`/
`cluster_selection_method`/`alpha`) — from the search space defined in
`--config-path`'s YAML file (default: `config/umap_hdbscan_optuna.yaml`;
`metric` stays a fixed scalar there, defaulting to `cosine` rather than
`euclidean` — it's shared by UMAP's own fit, HDBSCAN's own fit, DBCV, and
the dimensionality-reduction fidelity metrics below).

Within one trial, two things happen nested:

1. **Seed** — every value in the YAML file's `reproducibility.seeds` list,
   UMAP's `random_state` for that trial's hyperparameters.
2. **Repeat** — `reproducibility.repeats_per_seed` independent fits of the
   same (trial, seed) pair, used to measure how *reproducible* that seed is.

The seed whose repeats are most mutually consistent (lowest standard
deviation of noise fraction across its repeats) is chosen — not the seed
with the best score — and one of its repeats (the one closest to that
seed's own mean noise fraction) becomes that trial's representative result.
That representative result's composite objective score — DBCV plus
trustworthiness/continuity/steadiness/cohesiveness (computed via
[`zadu`](https://github.com/hj-n/zadu) under cosine distance; stress is
deliberately excluded), folded with noise-fraction/dominant-cluster-share/
cluster-count/micro-cluster-fragmentation penalties (weights configurable
in the YAML file) — is the single ranking criterion: it drives Optuna's own
trial selection *and* determines which trial's result gets written. The
study is resumable (SQLite-backed, under the YAML file's `study_dir`):
re-running the same command against the same input continues the same
study rather than starting over, and the configured `n_trials` (default
`50`, capped at `100`) is a *total* budget across resumes, not re-added
every invocation. A full per-trial history CSV is written under the YAML
file's `reports_dir` (`data/clustering/reports/` by default) on every run.

It writes the reduced embeddings, cluster assignments, and metrics to
`--reduced-embeddings-output-path`/`--assignments-output-path`/
`--metrics-output-path`.

The reduced-embeddings Parquet file has one row per successfully-embedded
criterion, containing `nct_id`, `study_first_posted_date`, `criterion_text`,
`criterion_type`, `reduced_embedding`, `embedding_backend`,
`embedding_model`, and `reduced_at` — independently joinable back to the
source criteria via `nct_id` without needing the cluster assignments. The
cluster-assignments Parquet file contains `nct_id`,
`study_first_posted_date`, `criterion_text`, `criterion_type`, `cluster_id`
(`-1` for noise), `cluster_probability` (`0.0` for noise points),
`embedding_backend`, `embedding_model`, and `clustered_at`. The metrics
JSON reports run provenance (sample/cluster counts, the winning trial's
HDBSCAN parameters, `noise_count`/`noise_fraction`) plus clustering
performance metrics — silhouette score, the Davies-Bouldin index, the
Calinski-Harabasz index (computed only over non-noise points),
`relative_validity_index` (DBCV), `winning_umap_config`, `winning_seed`,
`winning_seed_noise_fraction_std`, `dr_metrics`
(trustworthiness/continuity/steadiness/cohesiveness for the winning trial),
and `optuna_study_summary`/`objective_weights` (trial counts by status, the
winning trial number/score, the study/trial-history file paths, and the
weights actually used).

A trial is only considered fully valid if its resulting cluster count
falls between `HDBSCANClusterer.MIN_N_CLUSTERS` (`10`) and `MAX_N_CLUSTERS`
(`100`); no single cluster holds more than `MAX_CLUSTER_SHARE` (`20%`) of
all points; and `noise_fraction` does not exceed `MAX_NOISE_FRACTION`
(`20%`). A trial whose winning seed's representative repeat fails any of
these still has its metrics computed and reported where possible, but is
treated as degenerate and scores lower via the same continuous penalty
terms, with its own explanatory note per violated constraint. A trial
invalid for the dataset (e.g. `n_neighbors` exceeding the sample count) is
logged and scored with a guaranteed-worst sentinel rather than aborting the
study. If every trial that fit is degenerate, the least-bad trial by the
same composite score is still written, with `metrics_valid=false`
recorded — nothing is written only if every trial in the study was invalid
(never fit at all). Re-running this step against the same output paths
overwrites all three existing output files (and the trial-history CSV)
wholesale; only the Optuna study's own SQLite storage accumulates across
runs.

The YAML config's `n_trials` is capped at 100; `reproducibility.seeds` is
capped at 10 entries, `repeats_per_seed` at 10, and their product with
`n_trials` is capped at 25000 total fits — exceeding any of these raises an
error before any trial runs.

### Arguments

| Argument                           | Required | Description                                                                 |
| ----------------------------------- | -------- | ---------------------------------------------------------------------------- |
| `--input-path`                       | yes      | Input embeddings Parquet path (as written by `criteria_embedding.py`). |
| `--reduced-embeddings-output-path`   | yes      | Output reduced-embeddings Parquet path. |
| `--assignments-output-path`          | yes      | Output cluster-assignments Parquet path. |
| `--metrics-output-path`              | yes      | Output metrics JSON path. |
| `--config-path`                      | no       | Optuna search config YAML (combined UMAP + HDBSCAN search space, objective weights, trial budget, reproducibility seeds/repeats); default: `config/umap_hdbscan_optuna.yaml`. |

## 6. PaCMAP + HDBSCAN clustering

```bash
uv run python -m src.clustering.pacmap_hdbscan_clustering \
  --input-path data/embeddings/breast_cancer_trials_embeddings.parquet \
  --reduced-embeddings-output-path data/clustering/pacmap_hdbscan/breast_cancer_trials_pacmap_reduced_embeddings.parquet \
  --assignments-output-path data/clustering/pacmap_hdbscan/breast_cancer_trials_pacmap_hdbscan_clusters.parquet \
  --metrics-output-path data/clustering/pacmap_hdbscan/breast_cancer_trials_pacmap_hdbscan_cluster_metrics.json
```

The same idea as step 5, reducing with [PaCMAP](https://github.com/YingfanWang/PaCMAP)
instead of UMAP. PaCMAP has no `min_dist` equivalent — its
neighborhood/distance behavior is instead shaped by `mn_ratio` and
`fp_ratio` in the search space (default config: `config/pacmap_hdbscan_optuna.yaml`),
which weight mid-near vs. further point pairs during optimization.

A large `n_neighbors` (e.g. 30+) tends to over-smooth PaCMAP's projection
into one dominant blob — HDBSCAN then reports one giant cluster plus a
couple of stragglers, which the `MIN_N_CLUSTERS`/`MAX_CLUSTER_SHARE`/
`MAX_NOISE_FRACTION` guards (see step 5) correctly treat as degenerate via
the same continuous penalty terms, pushing Optuna's search away from that
region over successive trials. Everything else — the seed/repeat
reproducibility mechanics, composite objective (DBCV plus
trustworthiness/continuity/steadiness/cohesiveness via `zadu` under cosine
distance, folded with the same degenerate-result penalties), study
resumability, trial-history export, and overwrite-on-rerun behavior — is
identical to step 5. It writes the reduced embeddings, cluster assignments,
and metrics to `--reduced-embeddings-output-path`/`--assignments-output-path`/
`--metrics-output-path`, with the reduced-embeddings schema,
cluster-assignments schema, and metrics JSON shape (as
`winning_pacmap_config` instead of `winning_umap_config`) otherwise
identical to step 5's.

### Arguments

| Argument                           | Required | Description                                                                 |
| ----------------------------------- | -------- | ---------------------------------------------------------------------------- |
| `--input-path`                       | yes      | Input embeddings Parquet path (as written by `criteria_embedding.py`). |
| `--reduced-embeddings-output-path`   | yes      | Output reduced-embeddings Parquet path. |
| `--assignments-output-path`          | yes      | Output cluster-assignments Parquet path. |
| `--metrics-output-path`              | yes      | Output metrics JSON path. |
| `--config-path`                      | no       | Optuna search config YAML (combined PaCMAP + HDBSCAN search space, objective weights, trial budget, reproducibility seeds/repeats); default: `config/pacmap_hdbscan_optuna.yaml`. |

## 7. Scatterplot visualization

```bash
uv run python -m src.clustering.clustering_scatterplot \
  --method umap_hdbscan \
  --metrics-path data/clustering/umap_hdbscan/breast_cancer_trials_umap_hdbscan_cluster_metrics.json \
  --assignments-path data/clustering/umap_hdbscan/breast_cancer_trials_umap_hdbscan_clusters.parquet \
  --reduced-embeddings-path data/clustering/umap_hdbscan/breast_cancer_trials_umap_reduced_embeddings.parquet \
  --output-path data/clustering/umap_hdbscan/breast_cancer_trials_umap_hdbscan_scatterplot.html
```

This reads whichever clustering method's output paths are passed to it
(`--method`, one of `kmeans`/`umap_hdbscan`/`pacmap_hdbscan`, plus
`--metrics-path`/`--assignments-path`) — it never re-runs clustering — and
writes an interactive HTML scatterplot of those criteria in a 2-dimensional
UMAP space to `--output-path`, with points colored by cluster (a distinct
color for the two HDBSCAN-based methods' noise points), a legend mapping
color to cluster, and a hover tooltip showing each point's criterion text.
Pass `--embeddings-path` to supply the original embeddings Parquet, needed
whenever a fresh 2D projection has to be fit rather than reusing a
persisted one.

The 2D projection always reuses that method's own "optimal run" seed rather
than introducing new randomness: k-means's `kmeans_params.random_state`;
`umap_hdbscan`/`pacmap_hdbscan`'s `winning_seed`. For `umap_hdbscan`
specifically, if that run's winning configuration was already 2-dimensional
and you pass `--reduced-embeddings-path` pointing at its persisted
reduced-embeddings Parquet, those coordinates are reused directly instead
of fitting a new UMAP projection; otherwise (and always for
`pacmap_hdbscan`, whose own persisted projection is PaCMAP, not UMAP) a
fresh 2-dimensional UMAP projection is computed from `--embeddings-path`.

Re-running this step against the same output path overwrites the existing
HTML file wholesale; there is no append/skip-if-exists/merge behavior.

### Arguments

| Argument         | Required | Description                                                                 |
| ----------------- | -------- | ---------------------------------------------------------------------------- |
| `--method`          | yes      | Which clustering method's output to visualize: `kmeans`, `umap_hdbscan`, or `pacmap_hdbscan`. |
| `--metrics-path`    | yes      | That method's metrics JSON path. |
| `--assignments-path`| yes      | That method's cluster-assignments Parquet path (the file whose rows become the plotted points). |
| `--output-path`     | yes      | Output HTML path. |
| `--seed`            | no       | Override the 2D UMAP projection's seed (default: reuse that method's own optimal-run seed). |
| `--embeddings-path` | no       | Embeddings Parquet path to read when a fresh 2D projection is needed (every method except `umap_hdbscan` reusing an already-2D persisted projection via `--reduced-embeddings-path`). |
| `--reduced-embeddings-path` | no | For `--method umap_hdbscan` only: that run's persisted reduced-embeddings Parquet path, reused as-is instead of fitting a fresh projection if its winning config was already 2D. |

### Example: full pipeline for one condition

```bash
uv run python -m src.preprocessing.criteria_retrieval "breast cancer" \
  --output-path data/eligibility_criteria/breast_cancer_trials_eligibility.csv

uv run python -m src.embeddings.criteria_embedding \
  --input-path data/eligibility_criteria/breast_cancer_trials_eligibility.csv \
  --output-path data/embeddings/breast_cancer_trials_embeddings.parquet \
  --backend huggingface

uv run python -m src.clustering.kmeans_clustering \
  --input-path data/embeddings/breast_cancer_trials_embeddings.parquet \
  --assignments-output-path data/clustering/kmeans/breast_cancer_trials_clusters.parquet \
  --metrics-output-path data/clustering/kmeans/breast_cancer_trials_cluster_metrics.json \
  --k-min 10,15 --k-max 50,100 --random-state 1,2,3

uv run python -m src.clustering.umap_hdbscan_clustering \
  --input-path data/embeddings/breast_cancer_trials_embeddings.parquet \
  --reduced-embeddings-output-path data/clustering/umap_hdbscan/breast_cancer_trials_umap_reduced_embeddings.parquet \
  --assignments-output-path data/clustering/umap_hdbscan/breast_cancer_trials_umap_hdbscan_clusters.parquet \
  --metrics-output-path data/clustering/umap_hdbscan/breast_cancer_trials_umap_hdbscan_cluster_metrics.json

uv run python -m src.clustering.pacmap_hdbscan_clustering \
  --input-path data/embeddings/breast_cancer_trials_embeddings.parquet \
  --reduced-embeddings-output-path data/clustering/pacmap_hdbscan/breast_cancer_trials_pacmap_reduced_embeddings.parquet \
  --assignments-output-path data/clustering/pacmap_hdbscan/breast_cancer_trials_pacmap_hdbscan_clusters.parquet \
  --metrics-output-path data/clustering/pacmap_hdbscan/breast_cancer_trials_pacmap_hdbscan_cluster_metrics.json

uv run python -m src.clustering.clustering_report \
  --output-path data/clustering/reports/breast_cancer_clustering_report.md \
  --kmeans-metrics-path data/clustering/kmeans/breast_cancer_trials_cluster_metrics.json \
  --umap-hdbscan-metrics-path data/clustering/umap_hdbscan/breast_cancer_trials_umap_hdbscan_cluster_metrics.json \
  --pacmap-hdbscan-metrics-path data/clustering/pacmap_hdbscan/breast_cancer_trials_pacmap_hdbscan_cluster_metrics.json

uv run python -m src.clustering.clustering_scatterplot \
  --method umap_hdbscan \
  --metrics-path data/clustering/umap_hdbscan/breast_cancer_trials_umap_hdbscan_cluster_metrics.json \
  --assignments-path data/clustering/umap_hdbscan/breast_cancer_trials_umap_hdbscan_clusters.parquet \
  --reduced-embeddings-path data/clustering/umap_hdbscan/breast_cancer_trials_umap_reduced_embeddings.parquet \
  --output-path data/clustering/umap_hdbscan/breast_cancer_trials_umap_hdbscan_scatterplot.html
```
