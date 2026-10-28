# Clinical Trial Criteria Clustering

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
3. **`src/clustering/clustering_report.py`** — reads whichever of the
   clustering steps' metrics JSON paths are passed to it, and writes one
   consolidated Markdown report comparing their performance side by side.
4. **`src/clustering/umap_hdbscan_clustering.py`** — reduces embeddings with
   UMAP, tuned jointly with HDBSCAN by one Optuna search (search
   space/objective weights/trial budget come from a `--config-path` YAML
   file, not CLI flags; each trial also runs a configured
   seeds/repeats-per-seed grid internally to pick the most reproducible
   seed). Once a trial's representative repeat is chosen, BERTopic re-fits
   that same HDBSCAN configuration once more (skipping its own internal
   UMAP step) to additionally compute per-cluster topic keywords plus a
   zero-shot-classification label against a caller-supplied candidate label
   set — BERTopic's own cluster assignments, not the earlier HDBSCAN fit,
   are what the Optuna objective scores and what gets written. Writes the
   reduced embeddings plus cluster assignments and metrics.
   Read/validate/write logic shared between the clustering approaches lives
   in `src/clustering/clustering_io.py`; the HDBSCAN fitting/metrics code
   and BERTopic topic-representation layer shared by both reducer-based
   steps live in `src/clustering/algorithms/hdbscan_model.py` and
   `src/clustering/algorithms/bertopic_model.py` respectively, the Optuna
   search engine lives in `src/clustering/optimization/optuna_search.py`,
   and all of it is orchestrated by
   `src/clustering/reduction_hdbscan_pipeline.py`.
5. **`src/clustering/pacmap_hdbscan_clustering.py`** — the same idea as step
   4, reducing with PaCMAP instead of UMAP. The seed/repeat reproducibility
   mechanics both steps share live in
   `src/clustering/optimization/reduction_sweep.py`.
6. **`src/clustering/clustering_scatterplot.py`** — reads whichever
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
├── pacmap_hdbscan_optuna.yaml             # Optuna search config for step 5
└── umap_hdbscan_optuna.yaml               # Optuna search config for step 4
src/
├── paths.py                              # shared REPO_ROOT constant
├── logging_utils.py                      # shared console+optional-file logging setup
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
    ├── reduction_hdbscan_pipeline.py      # shared reduce+BERTopic/HDBSCAN+Optuna orchestration (steps 4-5)
    ├── clustering_report.py              # step 3
    ├── umap_hdbscan_clustering.py        # step 4
    ├── pacmap_hdbscan_clustering.py      # step 5
    ├── clustering_scatterplot.py         # step 6
    ├── optimization/                     # reducer-and-algorithm-agnostic Optuna search engine
    │   ├── optuna_search.py              #   config/search-space parsing, composite scoring, study lifecycle
    │   └── reduction_sweep.py            #   seed/repeat reproducibility engine
    └── algorithms/                       # "how do I fit *this* algorithm, and is its output any good" modules
        ├── hdbscan_model.py              #   HDBSCAN fitting + eligibility-gate metrics
        └── bertopic_model.py             #   BERTopic topic-representation layer
```

`src/clustering/` splits three ways, each with a distinct job:

- **Orchestration, at the package root**: `clustering_io.py` holds
  everything genuinely common to every clustering algorithm (reading/
  validating embeddings, writing output). `umap_hdbscan_clustering.py` and
  `pacmap_hdbscan_clustering.py` are thin shells: each supplies only its own
  reducer-specific hyperparameter validation/fit function plus a
  `ReducerPlugin` describing the handful of ways it differs from the other
  (output schema, hyperparameter names, a couple of metrics-JSON field
  names), and delegates the entire read-filter-search-write orchestration to
  `reduction_hdbscan_pipeline.run` — so adding another reducer means adding
  one new sibling module with its own plugin, not duplicating the
  orchestration again. `clustering_report.py` builds on top of the same
  `clustering_io.py` layer without modifying it, and `clustering_scatterplot.py`
  reads whichever of the two algorithms' output already exists (via its own
  `VISUALIZATION_SPECS` registry, one entry per algorithm, following the
  same "no branching elsewhere" pattern `clustering_report.py`'s
  `ALGORITHM_SPECS` already established) — neither re-runs or modifies
  anything. `reduction_hdbscan_pipeline.py` itself is the integration layer
  that composes the two subpackages below; it isn't itself "optimization" or
  "an algorithm," so it stays at the root rather than living inside either.
- **`optimization/`**: the Optuna machinery that has no idea which reducer
  or which clustering algorithm it's scoring. `optuna_search.py` is the
  config loader, hyperparameter suggestion (including dependent `int_ratio`
  parameters, e.g. keeping `min_samples` bounded relative to
  `min_cluster_size`), composite objective scoring, and study-lifecycle/
  trial-history mechanics; `reduction_sweep.py` holds the reducer-agnostic
  seed/repeat reproducibility mechanics each Optuna trial runs internally
  (picking the most reproducible seed, then a representative repeat). Both
  take generic callables/an already-configured `HDBSCANClusterer` as
  arguments — neither module imports a specific reducer.
- **`algorithms/`**: the modules that know how to fit one specific
  algorithm and judge its own output. `hdbscan_model.py` is the shared
  HDBSCAN fitting/metrics class (standalone HDBSCAN clustering is not a
  pipeline in this repo — only the two reducer-based steps use it).
  `bertopic_model.py` is the BERTopic topic-representation layer: once
  `reduction_hdbscan_pipeline.run` picks a trial's representative repeat, it
  fits `bertopic_model.fit_bertopic_topics` exactly once per trial, reusing
  that trial's own HDBSCAN hyperparameters (skipping BERTopic's own internal
  UMAP step) — BERTopic's own cluster assignments, not the sweep's earlier
  plain-HDBSCAN fit, are what the Optuna objective scores and what the
  winning trial's output reports.

None of this changes how any step is invoked — every CLI command in this
README still runs exactly as `uv run python -m src.clustering.<module>
"..."`; only the internal module layout moved.

### Extending the pipeline

Three extension points already exist in this codebase; adding to any of
them doesn't require touching the others.

**A new reducer** (alongside UMAP/PaCMAP): add one new sibling module at
`src/clustering/` top level, matching the shape of
`umap_hdbscan_clustering.py`/`pacmap_hdbscan_clustering.py` — its own
hyperparameter-validation/fit function plus a `ReducerPlugin` instance — and
point it at the existing `reduction_hdbscan_pipeline.run`. No changes needed
to the orchestration, the Optuna engine, or either algorithm module.

**A new objective-score term** (alongside DBCV, the dimensionality-reduction
fidelity terms, and the BERTopic topic-quality terms): add a new frozen
dataclass next to `DRMetrics`/`BERTopicQualityMetrics` in
`optimization/optuna_search.py`, a `compute_*` function that returns it (or
`None` on failure — every metric here is fail-soft, never aborting a
trial), new `ObjectiveWeights` fields following the existing naming
convention (`<metric>_weight` for rewards, `<type>_penalty` for penalties,
`missing_<context>_fallback_score` for the whole-computation-failed case),
fold it into `score_reduction_candidate`, and add the matching
`objective_weights:` keys to both `config/pacmap_hdbscan_optuna.yaml` and
`config/umap_hdbscan_optuna.yaml`.

**A new clustering algorithm** (alongside HDBSCAN): add a new module under
`src/clustering/algorithms/` shaped like `hdbscan_model.py` — a fitter
class/function plus a `compute_metrics`-style eligibility gate that reports
`metrics_valid` (cluster count, dominant-cluster share, noise fraction, or
whatever bounds make sense for that algorithm) — so
`reduction_hdbscan_pipeline.py`'s generic orchestration, and
`optimization/reduction_sweep.py`'s generic reproducibility sweep, can drive
it the same way they drive HDBSCAN today.

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
| `--log-file` | no      | If given, also append logs to this file, in addition to the console. |

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
| `--log-file`        | no       | If given, also append logs to this file, in addition to the console. |

### Backend notes

- **`vllm`** expects an externally-managed vLLM server already running an
  OpenAI-compatible embeddings endpoint (default `http://localhost:8000/v1`,
  default model `gpt-oss-20b`). Point it elsewhere with `VLLM_BASE_URL` /
  `VLLM_API_KEY` in `.env`, or `--model`.
- **`huggingface`** loads a `sentence-transformers` model in-process (default
  `Qwen/Qwen3-Embedding-8B`), which can be slow/memory-heavy on first run as
  the model downloads.

## 3. Clustering performance report

```bash
uv run python -m src.clustering.clustering_report \
  --output-path data/clustering/reports/breast_cancer_clustering_report.md \
  --umap-hdbscan-metrics-path data/clustering/umap_hdbscan/breast_cancer_trials_umap_hdbscan_cluster_metrics.json
```

This reads whichever of the UMAP+HDBSCAN/PaCMAP+HDBSCAN metrics
JSON paths are passed to it (one optional flag per algorithm: any omitted
algorithm is reported as "not yet run") — it never re-runs clustering —
and writes a single consolidated Markdown report comparing their
performance to `--output-path`.

The report includes a summary table of each algorithm's availability, a
shared-metrics comparison table (silhouette score, Davies-Bouldin index,
Calinski-Harabasz index — the three metrics computed identically by every
algorithm), and a detailed section per algorithm covering its own
algorithm-specific fields (e.g. noise count/fraction and winning
config/seed for the two reducer-based steps) plus its `metrics_valid`/
`notes` status verbatim. The report also calls out which algorithm has the
best (highest) silhouette score among those with valid metrics, since
silhouette score is the only metric comparable across every algorithm's
schema.

If an algorithm's metrics-path flag was omitted, or the given file is
missing/unreadable, that algorithm's section clearly states so (naming the
flag to pass, or the exact path that was checked) instead of being silently
omitted or aborting the whole report. Each run overwrites `--output-path`
wholesale.

### Arguments

| Argument       | Required | Description                                                                 |
| --------------- | -------- | ---------------------------------------------------------------------------- |
| `--output-path` | yes      | Output Markdown report path. |
| `--umap-hdbscan-metrics-path`   | no | UMAP + HDBSCAN metrics JSON path (omit if not yet run). |
| `--pacmap-hdbscan-metrics-path` | no | PaCMAP + HDBSCAN metrics JSON path (omit if not yet run). |
| `--print`        | no       | Also print the generated report to stdout, in addition to writing the file. |
| `--log-file`     | no       | If given, also append logs to this file, in addition to the console. |

## 4. UMAP + HDBSCAN + BERTopic clustering

```bash
uv run python -m src.clustering.umap_hdbscan_clustering \
  --input-path data/embeddings/breast_cancer_trials_embeddings.parquet \
  --criterion-type inclusion \
  --reduced-embeddings-output-path data/clustering/umap_hdbscan/breast_cancer_inclusion_trials_umap_reduced_embeddings.parquet \
  --assignments-output-path data/clustering/umap_hdbscan/breast_cancer_inclusion_trials_umap_hdbscan_clusters.parquet \
  --metrics-output-path data/clustering/umap_hdbscan/breast_cancer_inclusion_trials_umap_hdbscan_cluster_metrics.json \
  --candidate-labels "oncology eligibility,treatment history,disease measurement"

uv run python -m src.clustering.umap_hdbscan_clustering \
  --input-path data/embeddings/breast_cancer_trials_embeddings.parquet \
  --criterion-type exclusion \
  --reduced-embeddings-output-path data/clustering/umap_hdbscan/breast_cancer_exclusion_trials_umap_reduced_embeddings.parquet \
  --assignments-output-path data/clustering/umap_hdbscan/breast_cancer_exclusion_trials_umap_hdbscan_clusters.parquet \
  --metrics-output-path data/clustering/umap_hdbscan/breast_cancer_exclusion_trials_umap_hdbscan_cluster_metrics.json \
  --candidate-labels "oncology eligibility,treatment history,disease measurement"
```

`--criterion-type` is required (`inclusion` or `exclusion`,
with `"both"`-typed rows included on both sides) and each side must be run
separately with its own output filenames. The study name, its resumable
Optuna storage journal, and its trial-history CSV filename are all derived
from the input filename *and* `--criterion-type` together, so the two sides
always get independent, non-colliding resumable studies even when read from
the same input Parquet.

This reads the same embeddings Parquet file format as the other clustering
steps (given via `--input-path`), filters it down to the selected
`--criterion-type` side, drops rows whose embedding failed, and runs an
Optuna study whose every trial reduces the remaining embedding
vectors' dimensionality with UMAP before clustering the reduced vectors
with HDBSCAN. Each trial suggests a *combined* hyperparameter set — UMAP's
own (`n_components`/`n_neighbors`/`min_dist`) *and* HDBSCAN's own
(`min_cluster_size`/`min_samples`/`cluster_selection_epsilon`/
`cluster_selection_method`/`alpha`) — from the search space defined in
`--config-path`'s YAML file (default: `config/umap_hdbscan_optuna.yaml`;
`metric` stays a fixed scalar there, defaulting to `cosine` rather than
`euclidean` — it's shared by UMAP's own fit, HDBSCAN's own fit, DBCV, and
the dimensionality-reduction fidelity metrics below).

Once each trial's most reproducible seed and representative repeat are
chosen (see below), BERTopic re-fits that exact HDBSCAN configuration once
more over the same already-reduced embeddings — skipping BERTopic's own
internal UMAP step entirely — to additionally compute, per cluster, its
representative keywords and a label assigned by scoring those keywords
against `--candidate-labels` via a zero-shot-classification pipeline
(`--zeroshot-model`; a topic falls back to its raw keywords if no candidate
label scores confidently enough). BERTopic's own resulting cluster
assignments for that trial — not the earlier plain-HDBSCAN fit — are what
every trial's quality metrics and Optuna objective score are computed from,
and what the winning trial's output reports.

Within one trial, two things happen nested:

1. **Seed** — every value in the YAML file's `reproducibility.seeds` list,
   UMAP's `random_state` for that trial's hyperparameters.
2. **Repeat** — `reproducibility.repeats_per_seed` independent fits of the
   same (trial, seed) pair, used to measure how *reproducible* that seed is.

The seed whose repeats are most mutually consistent (lowest standard
deviation of noise fraction across its repeats) is chosen — not the seed
with the best score — and one of its repeats (the one closest to that
seed's own mean noise fraction) becomes that trial's representative result.
That representative result's reduced embeddings, re-clustered by BERTopic
(see above), feed its composite objective score — DBCV plus
trustworthiness/continuity/steadiness/cohesiveness (computed via
[`zadu`](https://github.com/hj-n/zadu) under cosine distance; stress is
deliberately excluded), folded with noise-fraction/dominant-cluster-share/
cluster-count/micro-cluster-fragmentation penalties (weights configurable
in the YAML file) — is the single ranking criterion: it drives Optuna's own
trial selection *and* determines which trial's result gets written. The
study is resumable (file-backed, under the YAML file's `study_dir`):
re-running the same command against the same input continues the same
study rather than starting over, and the configured `n_trials` (default
`50`, capped at `100`) is a *total* budget across resumes, not re-added
every invocation. A full per-trial history CSV is written under the YAML
file's `results_dir` (`data/clustering/reports/` by default) on every run.

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
(`-1` for BERTopic's own outlier/noise topic), `cluster_probability` (`0.0`
for noise points), `embedding_backend`, `embedding_model`, `clustered_at`,
`topic_keywords` (BERTopic's own per-topic keywords), and `topic_label`
(the zero-shot-assigned candidate label, or a keyword-derived fallback) —
every row sharing a `cluster_id` carries that cluster's own topic
representation, including noise rows. The metrics JSON reports run
provenance (sample/cluster counts, `noise_count`/`noise_fraction`) plus
clustering performance metrics computed from BERTopic's own cluster
assignments — silhouette score, the Davies-Bouldin index, the
Calinski-Harabasz index (computed only over non-noise points),
`relative_validity_index` (DBCV), `winning_umap_config`, `winning_seed`,
`winning_seed_noise_fraction_std`, `dr_metrics`
(trustworthiness/continuity/steadiness/cohesiveness for the winning trial's
UMAP output), `topics` (one entry per cluster/topic id, each with its
keywords, assigned label, and member count), `candidate_labels`/
`zeroshot_model` (what the zero-shot layer was configured with), and
`optuna_study_summary`/`objective_weights` (trial counts by status, the
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
wholesale; only the Optuna study's own file-based storage accumulates across
runs.

The YAML config's `n_trials` is capped at 100; `reproducibility.seeds` is
capped at 10 entries, `repeats_per_seed` at 10, and their product with
`n_trials` is capped at 25000 total fits — exceeding any of these raises an
error before any trial runs.

### Arguments

| Argument                           | Required | Description                                                                 |
| ----------------------------------- | -------- | ---------------------------------------------------------------------------- |
| `--input-path`                       | yes      | Input embeddings Parquet path (as written by `criteria_embedding.py`). |
| `--criterion-type`                   | yes      | Which side to cluster: `inclusion` or `exclusion`. Rows with `criterion_type="both"` are included on both sides. |
| `--reduced-embeddings-output-path`   | yes      | Output reduced-embeddings Parquet path. |
| `--assignments-output-path`          | yes      | Output cluster-assignments Parquet path. |
| `--metrics-output-path`              | yes      | Output metrics JSON path. |
| `--config-path`                      | no       | Optuna search config YAML (combined UMAP + HDBSCAN search space, objective weights, trial budget, reproducibility seeds/repeats); default: `config/umap_hdbscan_optuna.yaml`. |
| `--candidate-labels`                 | yes      | Comma-separated candidate topic labels for BERTopic's zero-shot classification layer, e.g. `"oncology eligibility,treatment history,disease measurement"`. |
| `--zeroshot-model`                   | no       | HuggingFace zero-shot-classification model id for that same layer (default: `facebook/bart-large-mnli`). |
| `--log-file`                         | no       | If given, also append logs to this file, in addition to the console. |

## 5. PaCMAP + HDBSCAN + BERTopic clustering

```bash
uv run python -m src.clustering.pacmap_hdbscan_clustering \
  --input-path data/embeddings/breast_cancer_trials_embeddings.parquet \
  --criterion-type inclusion \
  --reduced-embeddings-output-path data/clustering/pacmap_hdbscan/breast_cancer_inclusion_trials_pacmap_reduced_embeddings.parquet \
  --assignments-output-path data/clustering/pacmap_hdbscan/breast_cancer_inclusion_trials_pacmap_hdbscan_clusters.parquet \
  --metrics-output-path data/clustering/pacmap_hdbscan/breast_cancer_inclusion_trials_pacmap_hdbscan_cluster_metrics.json \
  --candidate-labels "oncology eligibility,treatment history,disease measurement"

uv run python -m src.clustering.pacmap_hdbscan_clustering \
  --input-path data/embeddings/breast_cancer_trials_embeddings.parquet \
  --criterion-type exclusion \
  --reduced-embeddings-output-path data/clustering/pacmap_hdbscan/breast_cancer_exclusion_trials_pacmap_reduced_embeddings.parquet \
  --assignments-output-path data/clustering/pacmap_hdbscan/breast_cancer_exclusion_trials_pacmap_hdbscan_clusters.parquet \
  --metrics-output-path data/clustering/pacmap_hdbscan/breast_cancer_exclusion_trials_pacmap_hdbscan_cluster_metrics.json \
  --candidate-labels "oncology eligibility,treatment history,disease measurement"
```

Same `--criterion-type` requirement and study-name-qualification behavior as
step 4 (see above).

The same idea as step 4, reducing with [PaCMAP](https://github.com/YingfanWang/PaCMAP)
instead of UMAP. PaCMAP has no `min_dist` equivalent — its
neighborhood/distance behavior is instead shaped by `mn_ratio` and
`fp_ratio` in the search space (default config: `config/pacmap_hdbscan_optuna.yaml`),
which weight mid-near vs. further point pairs during optimization.

A large `n_neighbors` (e.g. 30+) tends to over-smooth PaCMAP's projection
into one dominant blob — HDBSCAN then reports one giant cluster plus a
couple of stragglers, which the `MIN_N_CLUSTERS`/`MAX_CLUSTER_SHARE`/
`MAX_NOISE_FRACTION` guards (see step 4) correctly treat as degenerate via
the same continuous penalty terms, pushing Optuna's search away from that
region over successive trials. Everything else — the seed/repeat
reproducibility mechanics, the BERTopic topic-representation layer, the
composite objective (DBCV plus trustworthiness/continuity/steadiness/
cohesiveness via `zadu` under cosine distance, folded with the same
degenerate-result penalties), study resumability, trial-history export, and
overwrite-on-rerun behavior — is identical to step 4. It writes the reduced
embeddings, cluster assignments,
and metrics to `--reduced-embeddings-output-path`/`--assignments-output-path`/
`--metrics-output-path`, with the reduced-embeddings schema,
cluster-assignments schema, and metrics JSON shape (as
`winning_pacmap_config` instead of `winning_umap_config`) otherwise
identical to step 4's.

### Arguments

| Argument                           | Required | Description                                                                 |
| ----------------------------------- | -------- | ---------------------------------------------------------------------------- |
| `--input-path`                       | yes      | Input embeddings Parquet path (as written by `criteria_embedding.py`). |
| `--criterion-type`                   | yes      | Which side to cluster: `inclusion` or `exclusion`. Rows with `criterion_type="both"` are included on both sides. |
| `--reduced-embeddings-output-path`   | yes      | Output reduced-embeddings Parquet path. |
| `--assignments-output-path`          | yes      | Output cluster-assignments Parquet path. |
| `--metrics-output-path`              | yes      | Output metrics JSON path. |
| `--config-path`                      | no       | Optuna search config YAML (combined PaCMAP + HDBSCAN search space, objective weights, trial budget, reproducibility seeds/repeats); default: `config/pacmap_hdbscan_optuna.yaml`. |
| `--candidate-labels`                 | yes      | Comma-separated candidate topic labels for BERTopic's zero-shot classification layer, e.g. `"oncology eligibility,treatment history,disease measurement"`. |
| `--zeroshot-model`                   | no       | HuggingFace zero-shot-classification model id for that same layer (default: `facebook/bart-large-mnli`). |
| `--log-file`                         | no       | If given, also append logs to this file, in addition to the console. |

## 6. Scatterplot visualization

```bash
uv run python -m src.clustering.clustering_scatterplot \
  --method umap_hdbscan \
  --metrics-path data/clustering/umap_hdbscan/breast_cancer_trials_umap_hdbscan_cluster_metrics.json \
  --assignments-path data/clustering/umap_hdbscan/breast_cancer_trials_umap_hdbscan_clusters.parquet \
  --reduced-embeddings-path data/clustering/umap_hdbscan/breast_cancer_trials_umap_reduced_embeddings.parquet \
  --output-path data/clustering/umap_hdbscan/breast_cancer_trials_umap_hdbscan_scatterplot.html
```

This reads whichever clustering method's output paths are passed to it
(`--method`, one of `umap_hdbscan`/`pacmap_hdbscan`, plus
`--metrics-path`/`--assignments-path`) — it never re-runs clustering — and
writes an interactive HTML scatterplot of those criteria in a 2-dimensional
UMAP space to `--output-path`, with points colored by cluster (a distinct
color for the two HDBSCAN-based methods' noise points), a legend mapping
color to cluster, and a hover tooltip showing each point's criterion text.
Pass `--embeddings-path` to supply the original embeddings Parquet, needed
whenever a fresh 2D projection has to be fit rather than reusing a
persisted one.

The 2D projection always reuses that method's own "optimal run" seed rather
than introducing new randomness: `umap_hdbscan`/`pacmap_hdbscan`'s
`winning_seed`. For `umap_hdbscan` specifically, if that run's winning configuration was already 2-dimensional
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
| `--method`          | yes      | Which clustering method's output to visualize: `umap_hdbscan` or `pacmap_hdbscan`. |
| `--metrics-path`    | yes      | That method's metrics JSON path. |
| `--assignments-path`| yes      | That method's cluster-assignments Parquet path (the file whose rows become the plotted points). |
| `--output-path`     | yes      | Output HTML path. |
| `--seed`            | no       | Override the 2D UMAP projection's seed (default: reuse that method's own optimal-run seed). |
| `--embeddings-path` | no       | Embeddings Parquet path to read when a fresh 2D projection is needed (every method except `umap_hdbscan` reusing an already-2D persisted projection via `--reduced-embeddings-path`). |
| `--reduced-embeddings-path` | no | For `--method umap_hdbscan` only: that run's persisted reduced-embeddings Parquet path, reused as-is instead of fitting a fresh projection if its winning config was already 2D. |
| `--log-file`        | no       | If given, also append logs to this file, in addition to the console. |

### Example: full pipeline for one condition

Retrieval and embedding run once per condition, covering all criteria types.
Every clustering/report/scatterplot step, however, must be run once per
`--criterion-type` (`inclusion`, then `exclusion`) — each side gets its own
output filenames (`_inclusion_`/`_exclusion_` inserted into every path
below) and its own fully independent results and report.

```bash
uv run python -m src.preprocessing.criteria_retrieval "breast cancer" \
  --output-path data/eligibility_criteria/breast_cancer_trials_eligibility.csv

uv run python -m src.embeddings.criteria_embedding \
  --input-path data/eligibility_criteria/breast_cancer_trials_eligibility.csv \
  --output-path data/embeddings/breast_cancer_trials_embeddings.parquet \
  --backend huggingface

# --- inclusion side ---

uv run python -m src.clustering.umap_hdbscan_clustering \
  --input-path data/embeddings/breast_cancer_trials_embeddings.parquet \
  --criterion-type inclusion \
  --reduced-embeddings-output-path data/clustering/umap_hdbscan/breast_cancer_inclusion_trials_umap_reduced_embeddings.parquet \
  --assignments-output-path data/clustering/umap_hdbscan/breast_cancer_inclusion_trials_umap_hdbscan_clusters.parquet \
  --metrics-output-path data/clustering/umap_hdbscan/breast_cancer_inclusion_trials_umap_hdbscan_cluster_metrics.json \
  --candidate-labels "oncology eligibility,treatment history,disease measurement"

uv run python -m src.clustering.pacmap_hdbscan_clustering \
  --input-path data/embeddings/breast_cancer_trials_embeddings.parquet \
  --criterion-type inclusion \
  --reduced-embeddings-output-path data/clustering/pacmap_hdbscan/breast_cancer_inclusion_trials_pacmap_reduced_embeddings.parquet \
  --assignments-output-path data/clustering/pacmap_hdbscan/breast_cancer_inclusion_trials_pacmap_hdbscan_clusters.parquet \
  --metrics-output-path data/clustering/pacmap_hdbscan/breast_cancer_inclusion_trials_pacmap_hdbscan_cluster_metrics.json \
  --candidate-labels "oncology eligibility,treatment history,disease measurement"

uv run python -m src.clustering.clustering_report \
  --output-path data/clustering/reports/breast_cancer_inclusion_clustering_report.md \
  --umap-hdbscan-metrics-path data/clustering/umap_hdbscan/breast_cancer_inclusion_trials_umap_hdbscan_cluster_metrics.json \
  --pacmap-hdbscan-metrics-path data/clustering/pacmap_hdbscan/breast_cancer_inclusion_trials_pacmap_hdbscan_cluster_metrics.json

uv run python -m src.clustering.clustering_scatterplot \
  --method umap_hdbscan \
  --metrics-path data/clustering/umap_hdbscan/breast_cancer_inclusion_trials_umap_hdbscan_cluster_metrics.json \
  --assignments-path data/clustering/umap_hdbscan/breast_cancer_inclusion_trials_umap_hdbscan_clusters.parquet \
  --reduced-embeddings-path data/clustering/umap_hdbscan/breast_cancer_inclusion_trials_umap_reduced_embeddings.parquet \
  --output-path data/clustering/umap_hdbscan/breast_cancer_inclusion_trials_umap_hdbscan_scatterplot.html

# --- exclusion side (repeat with --criterion-type exclusion and matching paths) ---

uv run python -m src.clustering.umap_hdbscan_clustering \
  --input-path data/embeddings/breast_cancer_trials_embeddings.parquet \
  --criterion-type exclusion \
  --reduced-embeddings-output-path data/clustering/umap_hdbscan/breast_cancer_exclusion_trials_umap_reduced_embeddings.parquet \
  --assignments-output-path data/clustering/umap_hdbscan/breast_cancer_exclusion_trials_umap_hdbscan_clusters.parquet \
  --metrics-output-path data/clustering/umap_hdbscan/breast_cancer_exclusion_trials_umap_hdbscan_cluster_metrics.json \
  --candidate-labels "oncology eligibility,treatment history,disease measurement"

uv run python -m src.clustering.pacmap_hdbscan_clustering \
  --input-path data/embeddings/breast_cancer_trials_embeddings.parquet \
  --criterion-type exclusion \
  --reduced-embeddings-output-path data/clustering/pacmap_hdbscan/breast_cancer_exclusion_trials_pacmap_reduced_embeddings.parquet \
  --assignments-output-path data/clustering/pacmap_hdbscan/breast_cancer_exclusion_trials_pacmap_hdbscan_clusters.parquet \
  --metrics-output-path data/clustering/pacmap_hdbscan/breast_cancer_exclusion_trials_pacmap_hdbscan_cluster_metrics.json \
  --candidate-labels "oncology eligibility,treatment history,disease measurement"

uv run python -m src.clustering.clustering_report \
  --output-path data/clustering/reports/breast_cancer_exclusion_clustering_report.md \
  --umap-hdbscan-metrics-path data/clustering/umap_hdbscan/breast_cancer_exclusion_trials_umap_hdbscan_cluster_metrics.json \
  --pacmap-hdbscan-metrics-path data/clustering/pacmap_hdbscan/breast_cancer_exclusion_trials_pacmap_hdbscan_cluster_metrics.json

uv run python -m src.clustering.clustering_scatterplot \
  --method umap_hdbscan \
  --metrics-path data/clustering/umap_hdbscan/breast_cancer_exclusion_trials_umap_hdbscan_cluster_metrics.json \
  --assignments-path data/clustering/umap_hdbscan/breast_cancer_exclusion_trials_umap_hdbscan_clusters.parquet \
  --reduced-embeddings-path data/clustering/umap_hdbscan/breast_cancer_exclusion_trials_umap_reduced_embeddings.parquet \
  --output-path data/clustering/umap_hdbscan/breast_cancer_exclusion_trials_umap_hdbscan_scatterplot.html
```
