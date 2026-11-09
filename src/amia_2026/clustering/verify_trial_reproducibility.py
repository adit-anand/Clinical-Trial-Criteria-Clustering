"""Standalone diagnostic: re-run one already-recorded Optuna trial's hyperparameters repeatedly.

`reduction_hdbscan_pipeline.run` assumes `evaluate_trial` is deterministic
given the same hyperparameters, and relies on that to re-fit the study's
best trial once at the end of a run, purely to recover fit artifacts it
deliberately doesn't keep resident for every trial (see that module's own
`run`/`evaluate_trial` docstrings). When a trial persisted as `"ok"`
re-evaluates to a different status, this script lets you replay that same
trial's seed/repeat sweep (`reduction_sweep.evaluate_config`) standalone,
with `--repeats` repeats per seed instead of whatever the config originally
used, to see exactly which repeats succeed/fail/disagree and why (the new
`logger.warning`/`logger.debug` calls in `reduction_sweep.evaluate_repeat`
now surface the swallowed exception for any repeat that errors; pass
`--debug` to also print the full traceback).

Hyperparameters are read via `optuna_search.resolve_trial_hyperparams`, not
`trial.params` directly: for an `"int_ratio"`-kind hyperparameter (e.g.
`min_samples` relative to `min_cluster_size`), `trial.params` holds the
*raw sampled ratio*, not the resolved integer the original fit actually
used -- see that function's own docstring. Passing `trial.params` straight
into `HDBSCANClusterer` fits a different configuration than the one Optuna
actually scored; `reduction_hdbscan_pipeline.run` had exactly this bug when
re-evaluating the study's best trial (now fixed via the same helper).

Never writes to the study's journal file, and never writes a metrics JSON.
Reads the embeddings Parquet and the already-existing study; optionally
writes one Parquet of this trial's own re-fit cluster assignments (see
`--output-path`/`TRIAL_REPRODUCIBILITY_PARQUET_SCHEMA`).
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
from amia_2026.clustering.optimization import optuna_search
from amia_2026.clustering.optimization import reduction_sweep
from amia_2026.clustering.algorithms.hdbscan_model import HDBSCANClusterer
from amia_2026.clustering.reduction_hdbscan_pipeline import ReducerPlugin

logger = logging.getLogger(__name__)

# Maps each `--reducer` choice to its own `ReducerPlugin` and default
# `--config-path`, mirroring `pacmap_hdbscan_clustering.main`/
# `umap_hdbscan_clustering.main`'s own plugin wiring.
REDUCER_PLUGINS: dict[str, tuple[ReducerPlugin, Path]] = {
    "pacmap": (pacmap_hdbscan_clustering.PLUGIN, pacmap_hdbscan_clustering.DEFAULT_CONFIG_PATH),
    "umap": (umap_hdbscan_clustering.PLUGIN, umap_hdbscan_clustering.DEFAULT_CONFIG_PATH),
}

# One row per successfully-embedded criterion: this trial's own re-fit
# plain-HDBSCAN cluster assignment (see `--output-path`), from the winning
# seed's representative repeat (the same repeat `reduction_sweep`'s own
# seed/repeat sweep would pick as "the" result for this trial).
TRIAL_REPRODUCIBILITY_PARQUET_SCHEMA = pa.schema(
    [
        ("trial_number", pa.int32()),
        ("criterion_text", pa.string()),
        ("embedding", pa.list_(pa.float32())),
        ("cluster_id", pa.int32()),
    ]
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Re-run one already-recorded Optuna trial's effective hyperparameters repeatedly, "
        "to check whether its 'ok' seed/repeat sweep result is actually reproducible on demand."
    )
    parser.add_argument(
        "--input-path", type=Path, required=True,
        help="Embeddings Parquet, the same one originally passed to the reducer pipeline's --input-path",
    )
    parser.add_argument(
        "--reducer", choices=sorted(REDUCER_PLUGINS), required=True,
        help="Which reducer's study/search space to replay against",
    )
    parser.add_argument(
        "--trial-number", type=int, required=True,
        help="Optuna trial number to replay (see the trial-history CSV or runtime log, e.g. 'Trial 87 finished')",
    )
    parser.add_argument(
        "--config-path", type=Path, default=None,
        help="Optuna search config YAML; defaults to this reducer's own default config",
    )
    parser.add_argument(
        "--output-path", type=Path, required=True,
        help="Output Parquet path for this trial's own re-fit cluster assignments (trial_number, "
        "criterion_text, embedding, cluster_id) -- written only if the trial's winning seed has a "
        "usable ('ok' or 'degenerate') representative repeat",
    )
    parser.add_argument("--repeats", type=int, default=10, help="Number of repeats per seed (default: 10)")
    parser.add_argument(
        "--seeds", type=str, default=None,
        help="Comma-separated seeds to use instead of the config's own reproducibility.seeds (e.g. '1,2,3')",
    )
    parser.add_argument("--log-file", type=Path, default=None, help="If given, also append logs to this file")
    parser.add_argument(
        "--debug", action="store_true",
        help="Log at DEBUG level, surfacing the full traceback for any repeat that errors",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    configure_logging(args.log_file)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    plugin, default_config_path = REDUCER_PLUGINS[args.reducer]
    config_path = args.config_path if args.config_path is not None else default_config_path
    config = optuna_search.load_optuna_config(config_path, require_reproducibility=True)

    table = clustering_io.read_embeddings_parquet(args.input_path)
    filtered = clustering_io.filter_successful_rows(table)
    n_samples = filtered.num_rows
    embedding_matrix = clustering_io.extract_embedding_matrix(filtered)
    traceability = clustering_io.extract_traceability_columns(filtered)

    slug = slug_from_filename(args.input_path.name, "_trials_embeddings.parquet")
    study_name = f"{slug}_{plugin.name}_hdbscan"
    storage_path = config.study_dir / f"{study_name}_study.journal"
    if not storage_path.exists():
        raise FileNotFoundError(f"No Optuna study journal found at {storage_path}")
    storage = optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(str(storage_path)))
    study = optuna.load_study(study_name=study_name, storage=storage)
    trial = optuna_search.find_trial_by_number(study, args.trial_number)

    hyperparams = optuna_search.resolve_trial_hyperparams(trial, config.search_space)
    logger.info(
        "verify_trial_reproducibility: trial #%d raw optuna params=%s -> effective hyperparams=%s "
        "(recorded status=%r, recorded score=%s)",
        trial.number, trial.params, hyperparams, trial.user_attrs.get("status"), trial.value,
    )

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

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else config.reproducibility.seeds

    candidate = reduction_sweep.evaluate_config(
        fit_reduced, hyperparams, seeds, args.repeats, plugin.validate_hyperparameters,
        build_hdbscan_clusterer, embedding_matrix, n_samples,
    )
    reduction_sweep.log_config_candidate(trial.number, candidate)

    for seed_result in candidate.seed_results:
        statuses = [repeat.status for repeat in seed_result.repeats]
        noise_fractions = [
            repeat.metrics_result["noise_fraction"] if repeat.metrics_result else None
            for repeat in seed_result.repeats
        ]
        logger.info(
            "verify_trial_reproducibility: seed=%d statuses=%s noise_fractions=%s noise_fraction_std=%s",
            seed_result.seed, statuses, noise_fractions, seed_result.noise_fraction_std,
        )

    n_usable = sum(
        1 for seed_result in candidate.seed_results for repeat in seed_result.repeats
        if repeat.status in ("ok", "degenerate")
    )
    n_total = len(seeds) * args.repeats
    logger.info(
        "verify_trial_reproducibility: trial #%d -> overall status=%s (%d/%d repeat(s) usable across %d seed(s))",
        trial.number, candidate.status, n_usable, n_total, len(seeds),
    )

    if candidate.winning_seed is None:
        logger.warning(
            "verify_trial_reproducibility: trial #%d has no usable seed to re-fit; writing nothing to %s",
            trial.number, args.output_path,
        )
        return

    representative = reduction_sweep.representative_repeat(candidate)
    criterion_texts = traceability["criterion_text"]
    embeddings = traceability["embedding"]
    output_rows = [
        {
            "trial_number": trial.number,
            "criterion_text": criterion_texts[i],
            "embedding": embeddings[i],
            "cluster_id": int(representative.labels[i]),
        }
        for i in range(n_samples)
    ]
    clustering_io.write_parquet(output_rows, TRIAL_REPRODUCIBILITY_PARQUET_SCHEMA, args.output_path)
    logger.info(
        "verify_trial_reproducibility: wrote %d row(s) for trial #%d (representative repeat status=%r) -> %s",
        n_samples, trial.number, representative.status, args.output_path,
    )


if __name__ == "__main__":
    main()
