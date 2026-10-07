import argparse
import cProfile
import json
import logging
import pstats
import random
import subprocess
import warnings

import numpy as np
import pandas as pd
from IPython.display import HTML, display

from fair_irl.config import (
    DEFAULT_CONFIG_PATH,
    build_experiment_plan,
    load_config,
    parse_assignment,
)
from fair_irl.datasets import *
from fair_irl.experiment_utils import *
from fair_irl.irl.fair_irl import *
from fair_irl.utils import *


def play_notification():
    subprocess.run(
        ["powershell.exe", "-Command", "[System.Media.SystemSounds]::Hand.Play()"]
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate FairIRL Bias Reduction and its baselines, as"
            " configured by an experiment config file, and report the results"
            " to W&B."
        )
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="The experiment config file. Default: %(default)s",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override one config value for every experiment, e.g. `--set"
            " SH_ITERS=10` or `--set 'SELECTED_DATASETS=[\"COMPAS\"]'`. VALUE"
            " is read as JSON, falling back to a plain string. Repeatable."
        ),
    )
    parser.add_argument(
        "--annotate",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Record an extra value in every W&B run's config, without"
            " affecting the experiments. Repeatable."
        ),
    )
    parser.add_argument(
        "--results-file",
        help=(
            "Also write every run's outcome and subdominance results to this"
            " JSON file. Used by W&B sweep trials (see fair_irl.sweep)."
        ),
    )
    return parser.parse_args(argv)


def _run_result_recorder(records):
    """
    A `RUN_RESULT_LISTENERS` entry that collects each run's outcome into
    `records`, keyed by run id so that a run first finalized and then marked as
    failed ends up failed.
    """

    def record(run, summary):
        records[run.id] = {
            "run_id": run.id,
            "ALGORITHM": run.config["ALGORITHM"],
            "EXPERIMENT_NAME": run.config["EXPERIMENT_NAME"],
            "DATASET": run.config["DATASET"],
            "DATASET_BIAS_TYPE": run.config["DATASET_BIAS_TYPE"],
            "WEIGHT_ADJUST": run.config["WEIGHT_ADJUST"],
            "TRIAL": run.config["TRIAL"],
            "converged": summary is not None,
            "subdominance": (
                {key: float(summary[key]) for key in SUBDOMINANCE_KEYS}
                if summary is not None
                else None
            ),
        }

    return record


def main(argv=None):
    args = parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    warnings.filterwarnings("ignore")

    display(HTML("<style>.container { width:2000px !important; }</style>"))
    pd.set_option("display.max_columns", None)
    # pd.set_option('display.max_colwidth', None)

    # Prevent long logging lines from wrapping
    # display(HTML("<style>div.output_area pre {white-space: pre;}</style>"))
    np.set_printoptions(linewidth=np.inf)

    # Every experiment setting lives in the config file; see the header of
    # configs/experiment.yaml for how it is laid out.
    logging.info(f"Loading experiment config: {args.config}")
    plan = build_experiment_plan(
        load_config(args.config),
        overrides=dict(parse_assignment(text) for text in args.overrides),
        annotations=dict(parse_assignment(text) for text in args.annotate),
    )

    np.random.seed(plan.random_seed)
    random.seed(plan.random_seed)

    run_records = {}
    if args.results_file:
        RUN_RESULT_LISTENERS.append(_run_result_recorder(run_records))

    # Run experiments. Results are reported to Weights & Biases (project
    # `WANDB_PROJECT`, default "fair-irl"); the server address and credentials
    # come from the usual wandb configuration, i.e. the WANDB_BASE_URL
    # environment variable or ~/.config/wandb/settings.
    #
    # Every run of every experiment below shares this one session id, which is
    # what lets the plotting notebook pick out this execution's results rather
    # than mixing them with those of earlier executions.
    session_id = new_session_id()
    logging.info(f"Logging results to W&B project: {WANDB_PROJECT}")
    logging.info(f"W&B session: {session_id}")

    # Create config for each experiment
    for base_exp_info in plan.experiments:
        if base_exp_info["DATASET"] not in plan.selected_datasets:
            logging.info(
                f"Skipping experiment {base_exp_info['EXPERIMENT_NAME']} since it is not in the selected datasets list."
            )
            continue
        experiments = []
        for expert_algo in plan.expert_algos:
            experiments.append(
                {
                    "EXPERT_ALGO": expert_algo,
                    "DATASET": base_exp_info["DATASET"],
                }
            )
        for exp_i, experiment in enumerate(experiments):
            logging.info(f"EXPERIMENT {exp_i+1}/{len(experiments)}")

            exp_info = dict(base_exp_info)

            for k in experiment:
                exp_info[k] = experiment[k]

            source_X, source_y, source_feature_types = generate_dataset(
                experiment["DATASET"],
                n_samples=exp_info["N_DATASET_SAMPLES"],
            )

            for f in source_feature_types["categoric"]:
                # .map(str) (rather than .astype(str)) matches Pandas < 3
                # behavior: it naively stringifies every value, including
                # missing ones (NaN -> "nan"). Pandas >= 3's .astype(str) is
                # NA-aware and leaves missing values as missing instead.
                # .astype(object) then keeps the legacy object dtype instead
                # of the new strict "str" dtype, which rejects assigning
                # non-string sentinel values used elsewhere in the pipeline
                # (e.g. state reduction).
                source_X[f] = source_X[f].map(str).astype(object)

            source_X_cols = input_columns(source_feature_types)

            if exp_info["USE_HIDDEN_FEATURES_SOURCE"]:
                source_X_cols += source_feature_types["hidden"]
            _source_X = source_X[source_X_cols]

            logging.info(
                f"For dataset: {experiment['DATASET']} and expert algo: {experiment['EXPERT_ALGO']}:"
            )

            run_bias_experiment(
                exp_info,
                source_X=_source_X,
                source_y=source_y,
                source_feature_types=source_feature_types,
                session_id=session_id,
            )

    if args.results_file:
        with open(args.results_file, "w") as f:
            json.dump(
                {"session_id": session_id, "runs": list(run_records.values())},
                f,
                indent=2,
            )

    logging.info(f"TRAINING FINISHED SUCESSFULLY! W&B session: {session_id}")

    # play_notification()


if __name__ == "__main__":
    main()
