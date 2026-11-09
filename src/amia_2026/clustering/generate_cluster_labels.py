"""Standalone script: LLM-label one reducer's winning clustering configuration.

Unlike `reduction_hdbscan_pipeline.run` (which fits BERTopic's own
zero-shot-classification topic layer per trial, then LLM-relabels only the
winning trial's topics on top of that), this script skips BERTopic/zero-shot
entirely: it re-fits the winning (or a given) Optuna trial's reducer+HDBSCAN
configuration directly via `reduction_sweep.evaluate_config`, then runs
`llm_labeling`'s two-stage Anthropic-LLM pass (prompts under
`config/llm_prompts/`) straight over the resulting plain-HDBSCAN clusters.
Useful for getting LLM cluster labels for a trial that never made it through
the full pipeline's own write step (e.g. its BERTopic refit failed, or -- see
`reduction_hdbscan_pipeline.run`'s now-fixed re-evaluation step -- its
hyperparameters were temporarily mis-resolved).

Winning-trial resolution mirrors `reduction_hdbscan_pipeline.run`: the
highest-scoring trial whose own `status` user-attribute is `"ok"`
(`optuna_search.best_ok_trial_or_none`), with its hyperparameters re-derived
via `optuna_search.resolve_trial_hyperparams` (not `trial.params` directly --
see that function's docstring for why). `--trial-number` overrides this to
label a specific trial instead (e.g. for manual verification).

Writes a single Parquet file (named after the study) into `--output-dir`,
with one row per successfully-embedded criterion: `criterion_text`, that
row's original (pre-reduction) `embedding`, its `cluster_id` (`-1` for
HDBSCAN noise), and `cluster_label` (the LLM-assigned label for its
cluster; noise rows get the fixed placeholder below, since noise is never
sent to the LLM -- same convention as `llm_labeling.relabel_topics_with_llm`).

Read-only against the Optuna study (never writes to its journal file); the
only output is the cluster-labels Parquet described above.
"""

import argparse
import logging
from pathlib import Path

import optuna
import pyarrow as pa

from amia_2026.logging_utils import configure_logging
from amia_2026.paths import slug_from_filename
from amia_2026.clustering import clustering_io
from amia_2026.clustering import pacmap_hdbscan_clustering
from amia_2026.clustering import umap_hdbscan_clustering
from amia_2026.clustering.algorithms import llm_labeling
from amia_2026.clustering.algorithms.bertopic_model import TopicRepresentation, parse_candidate_labels
from amia_2026.clustering.algorithms.hdbscan_model import HDBSCANClusterer
from amia_2026.clustering.optimization import optuna_search
from amia_2026.clustering.optimization import reduction_sweep
from amia_2026.clustering.reduction_hdbscan_pipeline import ReducerPlugin

logger = logging.getLogger(__name__)

# Maps each `--reducer` choice to its own `ReducerPlugin` and default
# `--config-path`, mirroring `pacmap_hdbscan_clustering.main`/
# `umap_hdbscan_clustering.main`'s own plugin wiring.
REDUCER_PLUGINS: dict[str, tuple[ReducerPlugin, Path]] = {
    "pacmap": (pacmap_hdbscan_clustering.PLUGIN, pacmap_hdbscan_clustering.DEFAULT_CONFIG_PATH),
    "umap": (umap_hdbscan_clustering.PLUGIN, umap_hdbscan_clustering.DEFAULT_CONFIG_PATH),
}

# Placeholder label for HDBSCAN's own noise/outlier cluster (`cluster_id ==
# -1`) -- never sent to the LLM (mirrors `llm_labeling.relabel_topics_with_llm`'s
# own noise-is-never-relabeled convention), so it needs a fixed, non-empty
# label of its own rather than an empty string.
NOISE_CLUSTER_LABEL = "Noise"

CLUSTER_LABELS_PARQUET_SCHEMA = pa.schema(
    [
        ("criterion_text", pa.string()),
        ("embedding", pa.list_(pa.float32())),
        ("cluster_id", pa.int32()),
        ("cluster_label", pa.string()),
    ]
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit the winning (or a specified) Optuna trial's reducer+HDBSCAN configuration, "
        "LLM-label each resulting cluster via the two-stage prompts in config/llm_prompts, and write "
        "a Parquet of criterion text/embedding/cluster assignment/cluster label."
    )
    parser.add_argument(
        "--input-path", type=Path, required=True,
        help="Embeddings Parquet, the same one originally passed to the reducer pipeline's --input-path",
    )
    parser.add_argument(
        "--reducer", choices=sorted(REDUCER_PLUGINS), required=True,
        help="Which reducer's study/search space to use",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory to write the output cluster-labels Parquet into (created if missing)",
    )
    parser.add_argument(
        "--config-path", type=Path, default=None,
        help="Optuna search config YAML; defaults to this reducer's own default config",
    )
    parser.add_argument(
        "--trial-number", type=int, default=None,
        help="Optuna trial number to use instead of the study's current best 'ok' trial (see "
        "verify_trial_reproducibility.py for how to find one)",
    )
    parser.add_argument(
        "--candidate-labels", type=str, required=True,
        help="Comma-separated candidate topic labels, reused as soft guidance for both LLM calls "
        "(e.g. 'oncology eligibility,treatment history,disease measurement')",
    )
    parser.add_argument(
        "--llm-model", type=str, default=llm_labeling.DEFAULT_LLM_MODEL,
        help=f"Anthropic model id for cluster labeling (default: {llm_labeling.DEFAULT_LLM_MODEL}); "
        "requires ANTHROPIC_API_KEY in .env",
    )
    parser.add_argument(
        "--primary-summary-prompt-path", type=Path, default=llm_labeling.DEFAULT_PRIMARY_PROMPT_PATH,
        help=f"Text file with the primary-summarization system prompt (default: "
        f"{llm_labeling.DEFAULT_PRIMARY_PROMPT_PATH})",
    )
    parser.add_argument(
        "--secondary-summary-prompt-path", type=Path, default=llm_labeling.DEFAULT_SECONDARY_PROMPT_PATH,
        help=f"Text file with the secondary-summarization system prompt (default: "
        f"{llm_labeling.DEFAULT_SECONDARY_PROMPT_PATH})",
    )
    parser.add_argument("--log-file", type=Path, default=None, help="If given, also append logs to this file")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    configure_logging(args.log_file)
    candidate_labels = parse_candidate_labels(args.candidate_labels)

    plugin, default_config_path = REDUCER_PLUGINS[args.reducer]
    config_path = args.config_path if args.config_path is not None else default_config_path
    config = optuna_search.load_optuna_config(config_path, require_reproducibility=True)

    table = clustering_io.read_embeddings_parquet(args.input_path)
    filtered = clustering_io.filter_successful_rows(table)
    n_samples = filtered.num_rows
    embedding_matrix = clustering_io.extract_embedding_matrix(filtered)
    traceability = clustering_io.extract_traceability_columns(filtered)
    criterion_texts = traceability["criterion_text"]
    embeddings = traceability["embedding"]

    slug = slug_from_filename(args.input_path.name, "_trials_embeddings.parquet")
    study_name = f"{slug}_{plugin.name}_hdbscan"
    storage_path = config.study_dir / f"{study_name}_study.journal"
    if not storage_path.exists():
        raise FileNotFoundError(f"No Optuna study journal found at {storage_path}")
    storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(str(storage_path)))
    study = optuna.load_study(study_name=study_name, storage=storage)

    if args.trial_number is not None:
        trial = optuna_search.find_trial_by_number(study, args.trial_number)
    else:
        trial = optuna_search.best_ok_trial_or_none(study)
        if trial is None:
            raise ValueError(f"Study {study_name!r} has no trial with status='ok' to use as the winning configuration")

    hyperparams = optuna_search.resolve_trial_hyperparams(trial, config.search_space)
    logger.info("generate_cluster_labels: using trial #%d hyperparams=%s", trial.number, hyperparams)

    fixed_metric = config.fixed[plugin.fixed_metric_config_key]
    fit_reduced = plugin.make_fit_reduced(fixed_metric)

    def build_hdbscan_clusterer(params: dict) -> HDBSCANClusterer:
        return HDBSCANClusterer(
            min_cluster_size=params["min_cluster_size"],
            min_samples=params["min_samples"],
            metric=fixed_metric,
            cluster_selection_epsilon=config.fixed["cluster_selection_epsilon"],
            cluster_selection_method=params["cluster_selection_method"],
            alpha=config.fixed["alpha"],
        )

    candidate = reduction_sweep.evaluate_config(
        fit_reduced, hyperparams, config.reproducibility.seeds, config.reproducibility.repeats_per_seed,
        plugin.validate_hyperparameters, build_hdbscan_clusterer, embedding_matrix, n_samples,
    )
    if candidate.status != "ok":
        raise RuntimeError(
            f"Trial #{trial.number}'s configuration re-evaluated to status={candidate.status!r} instead "
            f"of 'ok' ({candidate.detail}); cannot generate cluster labels"
        )

    representative = reduction_sweep.representative_repeat(candidate)
    labels = representative.labels
    probabilities = representative.probabilities
    logger.info(
        "generate_cluster_labels: trial #%d fit %d cluster(s) (winning_seed=%d, noise=%.1f%%)",
        trial.number, representative.metrics_result["n_clusters_actual"], candidate.winning_seed.seed,
        representative.metrics_result["noise_fraction"] * 100,
    )

    cluster_sizes = clustering_io.compute_cluster_sizes(labels)
    placeholder_topics = [
        TopicRepresentation(
            topic_id=topic_id,
            keywords=[],
            label=NOISE_CLUSTER_LABEL if topic_id == -1 else "",
            count=cluster_sizes[str(topic_id)],
        )
        for topic_id in sorted({int(label) for label in labels})
    ]

    labeled_topics = llm_labeling.relabel_topics_with_llm(
        criterion_texts, labels, probabilities, placeholder_topics, candidate_labels, args.llm_model,
        args.primary_summary_prompt_path, args.secondary_summary_prompt_path,
    )
    label_by_cluster_id = {topic.topic_id: topic.label for topic in labeled_topics}
    logger.info("generate_cluster_labels: LLM-assigned labels=%s", label_by_cluster_id)

    output_rows = [
        {
            "criterion_text": criterion_texts[i],
            "embedding": embeddings[i],
            "cluster_id": int(labels[i]),
            "cluster_label": label_by_cluster_id[int(labels[i])],
        }
        for i in range(n_samples)
    ]

    output_path = args.output_dir / f"{study_name}_cluster_labels.parquet"
    clustering_io.write_parquet(output_rows, CLUSTER_LABELS_PARQUET_SCHEMA, output_path)
    logger.info(
        "generate_cluster_labels: wrote %d row(s) across %d cluster(s) to %s",
        n_samples, len(placeholder_topics), output_path,
    )


if __name__ == "__main__":
    main()
