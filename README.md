# Clinical Trial Criteria Clustering

Pipeline for building embeddings of clinical trial eligibility criteria:

Every step below takes its input and output file paths as explicit CLI
flags — there is no "condition" argument used to derive default paths
anywhere except step 1, where it's the actual ClinicalTrials.gov search
term. Chaining steps together means passing the previous step's output path
as the next step's input path. Clustering (steps 4-5) runs once over every
eligibility criterion regardless of `criterion_type` (`inclusion`,
`exclusion`, or `both`) — there is no per-side flag or output subdirectory.

1. **`amia_2026/preprocessing/criteria_retrieval.py`** — fetches NCI-affiliated
   clinical trials for a condition from ClinicalTrials.gov, retrieves their
   eligibility criteria from the NCI CTRP API, parses them into atomic
   inclusion/exclusion criteria, and writes them to the given CSV path. Each
   concern lives in its own module (`clinicaltrials_client.py`,
   `ctrp_client.py`, `criteria_parser.py`); `criteria_retrieval.py` composes
   them and exposes the CLI.
2. **`amia_2026/embeddings/criteria_embedding.py`** — reads that CSV, embeds each
   criterion's text with a configurable backend, and writes the result to
   the given Parquet path.
3. **`amia_2026/clustering/clustering_report.py`** — reads whichever of the
   clustering steps' metrics JSON paths are passed to it, and writes one
   consolidated Markdown report comparing their performance side by side.
4. **`amia_2026/clustering/umap_hdbscan_clustering.py`** — reduces embeddings with
   UMAP, tuned jointly with HDBSCAN by one Optuna search (search
   space/objective weights/trial budget come from a `--config-path` YAML
   file, not CLI flags; each trial also runs a configured
   seeds/repeats-per-seed grid internally to pick the most reproducible
   seed). Once a trial's representative repeat is chosen, BERTopic re-fits
   that same HDBSCAN configuration once more (skipping its own internal
   UMAP step) to additionally compute per-cluster topic keywords plus a
   zero-shot-classification label against a caller-supplied candidate label
   set — BERTopic's own cluster assignments, not the earlier HDBSCAN fit,
   are what the Optuna objective scores and what gets written. Writes only
   a metrics JSON — no cluster-assignments Parquet (`clustering_scatterplot.py`,
   step 6, re-fits the winning trial on demand instead when it needs per-row
   assignments). Read/validate/write logic shared between the clustering approaches lives
   in `amia_2026/clustering/clustering_io.py`; the HDBSCAN fitting/metrics code
   and BERTopic topic-representation layer shared by both reducer-based
   steps live in `amia_2026/clustering/algorithms/hdbscan_model.py` and
   `amia_2026/clustering/algorithms/bertopic_model.py` respectively, the Optuna
   search engine lives in `amia_2026/clustering/optimization/optuna_search.py`,
   and all of it is orchestrated by
   `amia_2026/clustering/reduction_hdbscan_pipeline.py`.
5. **`amia_2026/clustering/pacmap_hdbscan_clustering.py`** — the same idea as step
   4, reducing with PaCMAP instead of UMAP. The seed/repeat reproducibility
   mechanics both steps share live in
   `amia_2026/clustering/optimization/reduction_sweep.py`.
6. **`amia_2026/clustering/clustering_scatterplot.py`** — re-fits whichever
   clustering method's winning trial the given metrics JSON points to
   (steps 4-5 write no cluster-assignments Parquet of their own) and writes
   an interactive HTML scatterplot of its criteria in a 2-dimensional UMAP
   space (colored by cluster), reusing that method's own optimal-run seed
   for the projection.
7. **`amia_2026/clustering/clustering_review.py`** — a combined step that runs
   step 3's report, then automatically runs step 6's scatterplot for
   whichever methodology has the higher composite objective score (ties
   favor PaCMAP+HDBSCAN), resolving every path the scatterplot step needs
   from the metrics JSON the report step already read — no
   `--method`/`--embeddings-path` re-entry. Steps 3
   and 6 are untouched and still work standalone; this is purely additive.
8. **`amia_2026/clustering/generate_cluster_labels.py`** — a standalone,
   read-only script that re-fits a reducer's winning (or an explicitly
   named) Optuna trial directly via its seed/repeat sweep, skipping
   BERTopic/zero-shot entirely, then LLM-labels the resulting plain-HDBSCAN
   clusters with the same two-stage prompts steps 4-5 use for their winning
   trial's relabeling pass. Writes one Parquet of criterion text/embedding/
   cluster assignment/cluster label to a given output directory. Does not
   require steps 4-5 to have completed successfully for that trial — it
   finds the trial via the already-existing resumable Optuna study and
   fits it from scratch.

## Project structure

`amia_2026` is a real installable package rooted at `src/amia_2026/`
(declared in `pyproject.toml`, built by `uv_build`); `preprocessing/`,
`embeddings/`, and `clustering/` are its subpackages (each has an
`__init__.py`), so every pipeline module imports its siblings as ordinary
absolute package imports (e.g. `from amia_2026.preprocessing.criteria_retrieval
import EligibilityCriteriaFetcher`, `from amia_2026.clustering import
clustering_io`) — no `sys.path` manipulation anywhere. The one constant
every module needs — the repo root — is centralized in `amia_2026/paths.py`
rather than re-derived in each file. Every entry-point script is run as a
module: `uv run python -m amia_2026.<package>.<module> "..."` (used
throughout this README); `uv sync`/`uv run` makes the package importable
regardless of the current working directory.

```
config/
├── pacmap_hdbscan_optuna.yaml             # Optuna search config for step 5
└── umap_hdbscan_optuna.yaml               # Optuna search config for step 4
src/
└── amia_2026/
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
        ├── clustering_review.py              # step 7 (combines steps 3 and 6)
        ├── generate_cluster_labels.py         # step 8 (standalone winning-trial LLM labeling)
        ├── optimization/                     # reducer-and-algorithm-agnostic Optuna search engine
        │   ├── optuna_search.py              #   config/search-space parsing, composite scoring, study lifecycle
        │   └── reduction_sweep.py            #   seed/repeat reproducibility engine
        └── algorithms/                       # "how do I fit *this* algorithm, and is its output any good" modules
            ├── hdbscan_model.py              #   HDBSCAN fitting + eligibility-gate metrics
            └── bertopic_model.py             #   BERTopic topic-representation layer
```

`amia_2026/clustering/` splits three ways, each with a distinct job:

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
README still runs exactly as `uv run python -m amia_2026.clustering.<module>
"..."`; only the internal module layout moved.

### Extending the pipeline

Three extension points already exist in this codebase; adding to any of
them doesn't require touching the others.

**A new reducer** (alongside UMAP/PaCMAP): add one new sibling module at
`amia_2026/clustering/` top level, matching the shape of
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
`amia_2026/clustering/algorithms/` shaped like `hdbscan_model.py` — a fitter
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

Step 6's scatterplot LLM relabeling (see below) and step 8's standalone
cluster labeling both require an Anthropic API key:

```
ANTHROPIC_API_KEY=your-key-here
```

## 1. Fetching eligibility criteria

```bash
uv run python -m amia_2026.preprocessing.criteria_retrieval "breast cancer" \
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
uv run python -m amia_2026.embeddings.criteria_embedding \
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
uv run python -m amia_2026.clustering.clustering_report \
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
uv run python -m amia_2026.clustering.umap_hdbscan_clustering \
  --input-path data/embeddings/breast_cancer_trials_embeddings.parquet \
  --metrics-output-path data/clustering/umap_hdbscan/breast_cancer_trials_umap_hdbscan_cluster_metrics.json \
  --candidate-labels "oncology eligibility,treatment history,disease measurement"
```

This clusters every eligibility criterion in `--input-path` together,
regardless of its `criterion_type` (`inclusion`, `exclusion`, or `both`) —
there is no per-side flag or run. The study name, its resumable Optuna
storage journal, and its trial-history CSV filename are all derived from
the input filename alone.

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

Once each trial's most reproducible seed and representative repeat are
chosen (see below), BERTopic re-fits that exact HDBSCAN configuration once
more over the same already-reduced embeddings — skipping BERTopic's own
internal UMAP step entirely — to additionally compute, per cluster, its
representative keywords and a label assigned by scoring those keywords
against `--candidate-labels` via a zero-shot-classification pipeline
(`--zeroshot-model`; a topic falls back to its raw keywords if no candidate
label scores confidently enough). BERTopic's own resulting cluster
assignments for that trial — not the earlier plain-HDBSCAN fit — are what
every trial's quality metrics and Optuna objective score are computed from.
This zero-shot-classification step runs identically for every trial,
including re-evaluations of the eventual winner, and is what every quality
metric (`relative_validity_index`/`bertopic_quality_metrics`, including
`zeroshot_label_confidence_rate`) and the Optuna objective are scored from —
nothing below changes any of that.

Once the winning trial is already selected and scored, its topics are
written out exactly as `ZeroShotClassification` produced them — this step
never calls out to an LLM itself. A second, two-stage Anthropic-LLM
relabeling pass (**primary summarization** per cluster, then **secondary
summarization** across all clusters for semantic distinctness) does exist,
but it now runs inside step 6's scatterplot rendering instead, over this
step's already-written output, purely for that plot's own legend — see
step 6 below for its mechanics. This step's own output (below) always
reports the plain zero-shot labels.

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

It writes a metrics JSON to `--metrics-output-path` — no cluster-assignments
Parquet is written by this step. Per-row cluster assignments for the
winning trial (criterion text/cluster id/cluster probability) are instead
re-fit on demand, directly from this run's own resumable Optuna study, by
whatever later needs them (`clustering_scatterplot.py`, see step 6 below;
`generate_cluster_labels.py`/`verify_trial_reproducibility.py`, steps 8 and
the diagnostic below, already worked this way).

The metrics JSON reports run provenance (sample/cluster counts,
`noise_count`/`noise_fraction`) plus clustering performance metrics
computed from BERTopic's own cluster assignments — silhouette score, the
Davies-Bouldin index, the Calinski-Harabasz index (computed only over
non-noise points), `relative_validity_index` (DBCV), `winning_umap_config`,
`winning_seed`, `winning_seed_noise_fraction_std`, `dr_metrics`
(trustworthiness/continuity/steadiness/cohesiveness for the winning trial's
UMAP output), `topics` (one entry per cluster/topic id, each with its
keywords, zero-shot label, and member count), `candidate_labels`/
`zeroshot_model` (what the per-trial zero-shot layer was configured with —
`candidate_labels` is also what step 6 later reuses as its own LLM
relabeling pass's guidance list), and `optuna_study_summary`/
`objective_weights` (trial counts by status, the winning trial number/score,
the study/trial-history file paths, and the weights actually used).

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
| `--metrics-output-path`              | yes      | Output metrics JSON path (no cluster-assignments Parquet is written by this step). |
| `--config-path`                      | no       | Optuna search config YAML (combined UMAP + HDBSCAN search space, objective weights, trial budget, reproducibility seeds/repeats); default: `config/umap_hdbscan_optuna.yaml`. |
| `--candidate-labels`                 | yes      | Comma-separated candidate topic labels. Drives every trial's BERTopic zero-shot classification layer; also written to the metrics JSON, where step 6's scatterplot later reuses it as soft guidance for its own LLM relabeling pass, e.g. `"oncology eligibility,treatment history,disease measurement"`. |
| `--zeroshot-model`                   | no       | HuggingFace zero-shot-classification model id for the per-trial layer (default: `facebook/bart-large-mnli`). |
| `--log-file`                         | no       | If given, also append logs to this file, in addition to the console. |

## 5. PaCMAP + HDBSCAN + BERTopic clustering

```bash
uv run python -m amia_2026.clustering.pacmap_hdbscan_clustering \
  --input-path data/embeddings/breast_cancer_trials_embeddings.parquet \
  --metrics-output-path data/clustering/pacmap_hdbscan/breast_cancer_trials_pacmap_hdbscan_cluster_metrics.json \
  --candidate-labels "oncology eligibility,treatment history,disease measurement"
```

Same study-name-qualification behavior as step 4 (see above): it clusters
every criterion in `--input-path` together, with no per-side flag or run.

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
overwrite-on-rerun behavior — is identical to step 4. It writes a metrics
JSON to `--metrics-output-path` (no cluster-assignments Parquet, same as
step 4), with the metrics JSON shape (as `winning_pacmap_config` instead of
`winning_umap_config`) otherwise identical to step 4's.

### Arguments

| Argument                           | Required | Description                                                                 |
| ----------------------------------- | -------- | ---------------------------------------------------------------------------- |
| `--input-path`                       | yes      | Input embeddings Parquet path (as written by `criteria_embedding.py`). |
| `--metrics-output-path`              | yes      | Output metrics JSON path (no cluster-assignments Parquet is written by this step). |
| `--config-path`                      | no       | Optuna search config YAML (combined PaCMAP + HDBSCAN search space, objective weights, trial budget, reproducibility seeds/repeats); default: `config/pacmap_hdbscan_optuna.yaml`. |
| `--candidate-labels`                 | yes      | Comma-separated candidate topic labels. Drives every trial's BERTopic zero-shot classification layer; also written to the metrics JSON, where step 6's scatterplot later reuses it as soft guidance for its own LLM relabeling pass, e.g. `"oncology eligibility,treatment history,disease measurement"`. |
| `--zeroshot-model`                   | no       | HuggingFace zero-shot-classification model id for the per-trial layer (default: `facebook/bart-large-mnli`). |
| `--log-file`                         | no       | If given, also append logs to this file, in addition to the console. |

## 6. Scatterplot visualization

```bash
uv run python -m amia_2026.clustering.clustering_scatterplot \
  --method umap_hdbscan \
  --metrics-path data/clustering/umap_hdbscan/breast_cancer_trials_umap_hdbscan_cluster_metrics.json \
  --embeddings-path data/embeddings/breast_cancer_trials_embeddings.parquet \
  --output-path data/clustering/umap_hdbscan/breast_cancer_trials_umap_hdbscan_scatterplot.html
```

Steps 4-5 write no cluster-assignments Parquet of their own (see steps 4-5
above), so this step re-fits the winning trial itself: it locates the same
resumable Optuna study steps 4-5 would (derived from
`--embeddings-path`'s filename and `--method`, under `--config-path`'s own
`study_dir`; default config paths match steps 4-5's), loads the exact trial
`--metrics-path`'s own `optuna_study_summary.best_trial_number` names as the
winner, and re-fits its hyperparameters via the same seed/repeat
reproducibility sweep every trial ran internally — the same re-fit-on-demand
approach step 8 and the diagnostic below already use. If that re-fit
doesn't come back with the same status the trial was originally recorded
with, this step fails loudly rather than rendering a plot with no valid
cluster assignments.

It then writes an interactive HTML scatterplot of those criteria in a
2-dimensional UMAP space to `--output-path`, with points colored by cluster
(a distinct color for the two HDBSCAN-based methods' noise points), a
legend mapping color to cluster, and a hover tooltip showing each point's
criterion text. `--embeddings-path` (the original embeddings Parquet) is
read exactly once, both for the trial re-fit above and to fit this step's
own fresh 2D projection — neither method persists a reusable projection of
its own.

Before building the legend, this step also runs the same two-stage
Anthropic-LLM relabeling pass that used to live inside steps 4-5 themselves
— **primary summarization** sends each cluster's own top-50 eligibility
criteria (by `cluster_probability`, descending) to `--llm-model`, asking
for a single overarching theme of 5 words or fewer; **secondary
summarization** then sends every cluster's primary theme together in one
call, asking for final labels that are semantically distinct from each
other (also 5 words or fewer). It runs over the `topics`/`candidate_labels`
already recorded in `--metrics-path` and the freshly re-fit cluster
assignments above — `--metrics-path` is never written back to, so the
underlying metrics JSON keeps its own plain zero-shot labels. A fresh LLM
call (and a fresh trial re-fit) is made every time this step runs (nothing
is cached), and an LLM-relabeling failure (e.g. missing
`ANTHROPIC_API_KEY`) is logged and falls back to the zero-shot label for
that cluster rather than aborting the render.
`--primary-summary-prompt-path`/`--secondary-summary-prompt-path` point at
the system prompt used for each stage (defaults under `config/llm_prompts/`).

The 2D projection always reuses that method's own "optimal run" seed rather
than introducing new randomness: `umap_hdbscan`/`pacmap_hdbscan`'s
`winning_seed`.

Re-running this step against the same output path overwrites the existing
HTML file wholesale; there is no append/skip-if-exists/merge behavior.

### Arguments

| Argument         | Required | Description                                                                 |
| ----------------- | -------- | ---------------------------------------------------------------------------- |
| `--method`          | yes      | Which clustering method's output to visualize: `umap_hdbscan` or `pacmap_hdbscan`. |
| `--metrics-path`    | yes      | That method's metrics JSON path. |
| `--embeddings-path` | yes      | The same embeddings Parquet originally passed to that method's own `--input-path` — used both to re-fit the winning trial and to fit the fresh 2D projection. |
| `--output-path`     | yes      | Output HTML path. |
| `--config-path`     | no       | Optuna search config YAML used to re-fit the winning trial; default: that method's own reducer's default config (same as steps 4-5). |
| `--seed`            | no       | Override the 2D UMAP projection's seed (default: reuse that method's own optimal-run seed). |
| `--llm-model`       | no       | Anthropic model id for this step's own two-stage LLM cluster relabeling pass, run for display only (default: `claude-haiku-5-5`); requires `ANTHROPIC_API_KEY` in `.env`. |
| `--primary-summary-prompt-path`   | no | Text file with the primary-summarization system prompt (default: `config/llm_prompts/primary_summary_prompt.txt`). |
| `--secondary-summary-prompt-path` | no | Text file with the secondary-summarization system prompt (default: `config/llm_prompts/secondary_summary_prompt.txt`). |
| `--log-file`        | no       | If given, also append logs to this file, in addition to the console. |

## 7. Combined report + scatterplot

```bash
uv run python -m amia_2026.clustering.clustering_review \
  --umap-hdbscan-metrics-path data/clustering/umap_hdbscan/breast_cancer_trials_umap_hdbscan_cluster_metrics.json \
  --pacmap-hdbscan-metrics-path data/clustering/pacmap_hdbscan/breast_cancer_trials_pacmap_hdbscan_cluster_metrics.json \
  --report-path data/clustering/reports/breast_cancer_clustering_report.md \
  --scatterplot-path data/clustering/reports/breast_cancer_clustering_scatterplot.html
```

This chains steps 3 and 6 automatically: it generates the same report as
step 3, then renders step 6's scatterplot for whichever methodology has the
higher composite objective score (`optuna_study_summary.best_score`, already
surfaced in the report's "Best by composite score" Summary line and Shared
Metrics table), with ties resolved in favor of PaCMAP+HDBSCAN. The
`--embeddings-path` the scatterplot step needs (for both its own winning-trial
re-fit and its fresh 2D projection — see step 6 above) is resolved
automatically from the winning methodology's own metrics JSON's `input_path`
field — there is no `--method` or path flag to pass for the scatterplot half
of this command. Pass only whichever `--<algorithm>-metrics-path` flag(s)
you have; the report is always written even if no methodology qualifies for
a scatterplot (e.g. both omitted, both invalid, or both from an older
schema without a composite score), and even if the winning methodology's
own trial re-fit fails — a clear reason is logged and the command does not
fail. Steps 3 and 6 themselves are untouched and remain usable standalone,
including for cases this command doesn't cover (such as visualizing the
lower-scoring methodology on demand).
`--llm-model`/`--primary-summary-prompt-path`/`--secondary-summary-prompt-path`
are forwarded verbatim to step 6's own (display-only) LLM relabeling pass.

### Arguments

| Argument         | Required | Description                                                                 |
| ----------------- | -------- | ---------------------------------------------------------------------------- |
| `--umap-hdbscan-metrics-path`   | no (at least one of the two `--*-metrics-path` flags should be given) | UMAP + HDBSCAN metrics JSON path (omit if this algorithm hasn't been run). |
| `--pacmap-hdbscan-metrics-path` | no       | PaCMAP + HDBSCAN metrics JSON path (omit if this algorithm hasn't been run). |
| `--report-path`      | yes      | Output Markdown report path. |
| `--scatterplot-path` | yes      | Output HTML scatterplot path (written only if a methodology qualifies for selection). |
| `--print`            | no       | Also print the generated report to stdout. |
| `--llm-model`        | no       | Anthropic model id, passed through to step 6's own (display-only) LLM relabeling pass (default: `claude-haiku-5-5`); requires `ANTHROPIC_API_KEY` in `.env`. |
| `--primary-summary-prompt-path`   | no | Passed through to step 6 (default: `config/llm_prompts/primary_summary_prompt.txt`). |
| `--secondary-summary-prompt-path` | no | Passed through to step 6 (default: `config/llm_prompts/secondary_summary_prompt.txt`). |
| `--log-file`         | no       | If given, also append logs to this file, in addition to the console. |

## 8. Standalone cluster labeling

```bash
uv run python -m amia_2026.clustering.generate_cluster_labels \
  --input-path data/embeddings/breast_cancer_trials_embeddings.parquet \
  --reducer pacmap \
  --output-dir data/clustering/pacmap_hdbscan \
  --candidate-labels "oncology eligibility,treatment history,disease measurement"
```

This is a standalone, read-only alternative to step 6's own (display-only)
LLM relabeling pass: it re-fits one reducer's already-tuned Optuna study's
winning hyperparameter configuration directly, skipping BERTopic/zero-shot
entirely, then LLM-labels the resulting plain-HDBSCAN clusters, and
persists those labels to its own output Parquet (unlike step 6's pass,
which is display-only and never written to disk). Useful on its own (no
need to re-run the full read-search-write pipeline just to get labels for a
trial), and in particular for a trial whose own run through step 4/5 never
wrote output (e.g. a re-evaluation failure) but is still on record in the
study.

It locates the same resumable Optuna study step 4/5 would (derived from
`--input-path`'s filename and `--reducer`, under `--config-path`'s own
`study_dir`; default config paths match steps 4/5's), and by default uses
the study's current best trial whose own `status` is `"ok"`
(`optuna_search.best_ok_trial_or_none` — never a degenerate trial, even if
it out-scored every eligible one). `--trial-number` overrides this to use
a specific trial instead. Either way, that trial's hyperparameters are
re-derived from the study's own recorded `trial.params` via
`optuna_search.resolve_trial_hyperparams` (not used as-is — an
`"int_ratio"`-kind hyperparameter like `min_samples` is recorded as its raw
sampled ratio, not the resolved integer the original fit used), then fit
via the same seed/repeat reproducibility sweep every trial runs internally
(`reduction_sweep.evaluate_config`, using `--config-path`'s own
`reproducibility.seeds`/`repeats_per_seed`) to pick the most reproducible
seed and a representative repeat. If that re-fit doesn't come back `"ok"`,
the script fails loudly rather than writing a degenerate/partial result.

The representative repeat's own plain-HDBSCAN clusters (not a BERTopic
refit) are then labeled by the same two-stage Anthropic-LLM pass steps 4-5
use for their winning trial's relabeling (`--llm-model`,
`--primary-summary-prompt-path`/`--secondary-summary-prompt-path`, default
prompts under `config/llm_prompts/`; `--candidate-labels` is reused as soft
guidance for both calls): primary summarization labels each cluster from
its own top criteria, then secondary summarization revises every label at
once for cross-cluster distinctness. Noise (`cluster_id == -1`) is never
sent to the LLM and is always labeled `"Noise"`.

Writes one Parquet file, named after the study (`<study_name>_cluster_labels.parquet`),
into `--output-dir` (created if missing) — one row per successfully-embedded
criterion, containing only `criterion_text`, that row's original
(pre-reduction) `embedding` vector, `cluster_id`, and `cluster_label`. This
is deliberately a narrower schema than steps 4-5's own cluster-assignments
Parquet (no `nct_id`/`cluster_probability`/etc.) — this script's own job is
just criterion text, embedding, cluster assignment, and label. Re-running
against the same `--output-dir`/study overwrites the existing file
wholesale; the Optuna study's own journal file is never written to.

### Arguments

| Argument                           | Required | Description                                                                 |
| ----------------------------------- | -------- | ---------------------------------------------------------------------------- |
| `--input-path`                       | yes      | Input embeddings Parquet path (as written by `criteria_embedding.py`); also identifies which Optuna study to read. |
| `--reducer`                          | yes      | Which reducer's study/search space to use: `pacmap` or `umap`. |
| `--output-dir`                       | yes      | Directory to write the output cluster-labels Parquet into (created if missing). |
| `--config-path`                      | no       | Optuna search config YAML; default: that reducer's own default config (same as steps 4-5). |
| `--trial-number`                     | no       | Use a specific Optuna trial instead of the study's current best `"ok"` trial. |
| `--candidate-labels`                 | yes      | Comma-separated candidate topic labels, reused as soft guidance for both LLM calls, e.g. `"oncology eligibility,treatment history,disease measurement"`. |
| `--llm-model`                        | no       | Anthropic model id for cluster labeling (default: `claude-haiku-5-5`); requires `ANTHROPIC_API_KEY` in `.env`. |
| `--primary-summary-prompt-path`      | no       | Text file with the primary-summarization system prompt (default: `config/llm_prompts/primary_summary_prompt.txt`). |
| `--secondary-summary-prompt-path`    | no       | Text file with the secondary-summarization system prompt (default: `config/llm_prompts/secondary_summary_prompt.txt`). |
| `--log-file`                         | no       | If given, also append logs to this file, in addition to the console. |

## Diagnostic: verifying trial reproducibility

```bash
uv run python -m amia_2026.clustering.verify_trial_reproducibility \
  --input-path data/embeddings/breast_cancer_trials_embeddings.parquet \
  --reducer pacmap \
  --trial-number 87 \
  --repeats 20 \
  --output-path data/clustering/pacmap_hdbscan/breast_cancer_trials_trial_87_reproducibility.parquet
```

`amia_2026/clustering/verify_trial_reproducibility.py` is a standalone
diagnostic, not a pipeline step: it never writes to the study's journal
file and never writes a metrics JSON. Steps 4-5 assume `evaluate_trial` is
deterministic given the same hyperparameters, and rely on that assumption
to re-fit the study's best trial once at the end of a run. When a trial
that was persisted as `"ok"` re-evaluates to a different status, use this
script to replay that same trial's seed/repeat sweep on demand, with
`--repeats` repeats per seed instead of whatever the config originally
used, to see exactly which repeats succeed/fail/disagree and why.

It locates the same resumable Optuna study step 4/5/8 would (derived from
`--input-path`'s filename and `--reducer`, under `--config-path`'s own
`study_dir`), loads the trial identified by `--trial-number` from that
study, and resolves its effective hyperparameters via
`optuna_search.resolve_trial_hyperparams` rather than using `trial.params`
directly — for an `"int_ratio"`-kind hyperparameter (e.g. `min_samples`
relative to `min_cluster_size`), `trial.params` holds the raw sampled
ratio, not the resolved integer the original fit actually used. It then
re-runs that trial's seed/repeat sweep (`reduction_sweep.evaluate_config`)
and logs, per seed, each repeat's status and noise fraction plus that
seed's noise-fraction standard deviation, and finally the trial's overall
status and how many repeats were usable across all seeds.

If the trial's winning seed has a usable (`"ok"` or `"degenerate"`)
representative repeat, this script also writes `--output-path`: one
Parquet row per successfully-embedded criterion, containing `trial_number`,
`criterion_text`, that row's original (pre-reduction) `embedding`, and
`cluster_id` from that representative repeat's own plain-HDBSCAN fit (see
`TRIAL_REPRODUCIBILITY_PARQUET_SCHEMA`). If every seed errored (no
representative repeat exists at all), nothing is written to `--output-path`
and a warning is logged instead.

`--config-path` is the same Optuna search config YAML steps 4/5 use
(default: that reducer's own default config, `config/umap_hdbscan_optuna.yaml`
or `config/pacmap_hdbscan_optuna.yaml`) — it is not a place to hand-write
hyperparameter values. It supplies the search-space definition needed to
resolve the trial's `int_ratio` parameters correctly, the fixed parameters
(`metric`/`alpha`/`cluster_selection_epsilon`) shared by every trial, and
the `reproducibility.seeds`/`repeats_per_seed` this script falls back to
when `--seeds`/`--repeats` are omitted. The actual hyperparameters being
validated always come from the recorded trial in the study, identified by
`--trial-number` — this script has no mode for validating an arbitrary,
not-yet-recorded hyperparameter combination.

### Arguments

| Argument         | Required | Description                                                                 |
| ----------------- | -------- | ---------------------------------------------------------------------------- |
| `--input-path`      | yes      | Embeddings Parquet, the same one originally passed to the reducer pipeline's `--input-path`; also identifies which Optuna study to read. |
| `--reducer`         | yes      | Which reducer's study/search space to replay against: `pacmap` or `umap`. |
| `--trial-number`    | yes      | Optuna trial number to replay (see the trial-history CSV or runtime log, e.g. `"Trial 87 finished"`). |
| `--output-path`     | yes      | Output Parquet path for this trial's own re-fit cluster assignments (`trial_number`/`criterion_text`/`embedding`/`cluster_id`); written only if the trial's winning seed has a usable representative repeat. |
| `--config-path`     | no       | Optuna search config YAML; default: that reducer's own default config (same as steps 4/5/8). |
| `--repeats`         | no       | Number of repeats per seed (default: `10`). |
| `--seeds`           | no       | Comma-separated seeds to use instead of the config's own `reproducibility.seeds`, e.g. `"1,2,3"`. |
| `--debug`           | no       | Log at DEBUG level, surfacing the full traceback for any repeat that errors. |
| `--log-file`        | no       | If given, also append logs to this file, in addition to the console. |

### Example: full pipeline for one condition

Retrieval and embedding run once per condition, covering all criteria types.
Every clustering/report/scatterplot step also runs once, over every
eligibility criterion regardless of `criterion_type` — there is no
inclusion/exclusion split anywhere in the pipeline.

```bash
uv run python -m amia_2026.preprocessing.criteria_retrieval "breast cancer" \
  --output-path data/eligibility_criteria/breast_cancer_trials_eligibility.csv

uv run python -m amia_2026.embeddings.criteria_embedding \
  --input-path data/eligibility_criteria/breast_cancer_trials_eligibility.csv \
  --output-path data/embeddings/breast_cancer_trials_embeddings.parquet \
  --backend huggingface

uv run python -m amia_2026.clustering.umap_hdbscan_clustering \
  --input-path data/embeddings/breast_cancer_trials_embeddings.parquet \
  --metrics-output-path data/clustering/umap_hdbscan/breast_cancer_trials_umap_hdbscan_cluster_metrics.json \
  --candidate-labels "oncology eligibility,treatment history,disease measurement"

uv run python -m amia_2026.clustering.pacmap_hdbscan_clustering \
  --input-path data/embeddings/breast_cancer_trials_embeddings.parquet \
  --metrics-output-path data/clustering/pacmap_hdbscan/breast_cancer_trials_pacmap_hdbscan_cluster_metrics.json \
  --candidate-labels "oncology eligibility,treatment history,disease measurement"

uv run python -m amia_2026.clustering.clustering_report \
  --output-path data/clustering/reports/breast_cancer_clustering_report.md \
  --umap-hdbscan-metrics-path data/clustering/umap_hdbscan/breast_cancer_trials_umap_hdbscan_cluster_metrics.json \
  --pacmap-hdbscan-metrics-path data/clustering/pacmap_hdbscan/breast_cancer_trials_pacmap_hdbscan_cluster_metrics.json

uv run python -m amia_2026.clustering.clustering_scatterplot \
  --method umap_hdbscan \
  --metrics-path data/clustering/umap_hdbscan/breast_cancer_trials_umap_hdbscan_cluster_metrics.json \
  --embeddings-path data/embeddings/breast_cancer_trials_embeddings.parquet \
  --output-path data/clustering/umap_hdbscan/breast_cancer_trials_umap_hdbscan_scatterplot.html
```
