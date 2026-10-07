"""
Hyperparameter tuning of FairIRL Bias Reduction and its baselines with W&B
sweeps.

Each technique is tuned by its own sweep, configured by one of the files in
configs/sweeps/: a search over that technique's hyperparameters that
minimizes its mean validation subdominance -- a grid search for FairIRL Bias
Reduction, whose search space is small enough to try in full, and a Bayesian
one for every baseline. Two commands:

    # Create the sweeps for one technique -- by default one per selected
    # dataset, so that each dataset gets its own hyperparameters:
    uv run python -m fair_irl.sweep create configs/sweeps/superhuman_fairness.yaml

    # Then start an agent for each sweep id it prints:
    uv run wandb agent <entity>/<project>/<sweep id>

`create --joint` instead creates one sweep that tunes a single set of
hyperparameters across all the datasets. `wandb sweep <file>` also works, and
does the same as `--joint`.

The agent runs `python -m fair_irl.sweep trial` for every set of
hyperparameters the search picks. That trial is the sweep's own W&B run: it
runs the experiment script with those hyperparameters as `--set` overrides, in
a child process, and logs the objective the search minimizes.

The child process is needed because the experiment script reports one W&B run
per technique, bias type and trial, while inside a sweep agent every
`wandb.init()` is forced onto the sweep's own run. The child is started without
the agent's sweep environment variables, so its runs are ordinary runs -- the
same ones the script reports outside a sweep -- each recording the sweep trial
that launched it as `SWEEP_ID`/`SWEEP_RUN_ID` in its config.
"""

import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import wandb
import yaml

from fair_irl.config import DEFAULT_CONFIG_PATH, build_experiment_plan, load_config
from fair_irl.experiment_utils import (
    ALGORITHM_FAIRIRL,
    SUBDOMINANCE_KEYS,
    WANDB_ENTITY,
    WANDB_PROJECT,
)

# The sweep's metric: what `trial` logs, and what the sweep configs minimize.
OBJECTIVE = "objective"

# The `run_experiment_trial()` summary key the objective is the mean of, by
# default. Validation, so that the test split stays untouched by tuning; sum
# aggregated absolute subdominance, which is what FairIRL Bias Reduction's own
# `opt_debias` weight search minimizes.
DEFAULT_OBJECTIVE_KEY = "sum_abs_subdominance_val"

# Set by the sweep agent for its trial, and read by `wandb.init()`. A child
# process that inherits them has its every run forced onto the trial's run.
SWEEP_ENV_VARS = ("WANDB_RUN_ID", "WANDB_SWEEP_ID", "WANDB_SWEEP_PARAM_PATH")


def _scored_runs(runs):
    """
    The runs the objective is computed over: every run, except FairIRL Bias
    Reduction's runs with the unadjusted weights. Those are only its starting
    point; its results are the runs of its weight adjustments.
    """
    return [
        run
        for run in runs
        if not (run["ALGORITHM"] == ALGORITHM_FAIRIRL and not run["WEIGHT_ADJUST"])
    ]


def _summarize(runs, objective_key):
    """
    Compute what a sweep trial logs from its runs' results.

    Returns
    -------
    metrics : dict<str, float> or None
        `None` when there is nothing to score, or when any scored run failed:
        a mean over only the runs that succeeded would favor hyperparameters
        that make a technique fail on its hardest cases.
    """
    scored = _scored_runs(runs)
    if not scored or not all(run["converged"] for run in scored):
        return None

    metrics = {
        OBJECTIVE: float(np.mean([run["subdominance"][objective_key] for run in scored]))
    }
    for key in SUBDOMINANCE_KEYS:
        metrics[f"mean/{key}"] = float(
            np.mean([run["subdominance"][key] for run in scored])
        )
    for name in sorted({run["EXPERIMENT_NAME"] for run in scored}):
        metrics[f"{OBJECTIVE}/{name}"] = float(
            np.mean(
                [
                    run["subdominance"][objective_key]
                    for run in scored
                    if run["EXPERIMENT_NAME"] == name
                ]
            )
        )
    return metrics


def run_trial(args):
    """Run one sweep trial. See the module docstring."""
    run = wandb.init(job_type="sweep_trial")
    # The values the search picked, plus the sweep config's fixed ones. Keys
    # starting with "_" are wandb's own bookkeeping.
    overrides = {k: v for k, v in dict(run.config).items() if not k.startswith("_")}

    with tempfile.TemporaryDirectory() as tmp:
        results_file = os.path.join(tmp, "results.json")
        command = [
            sys.executable,
            "-m",
            "fair_irl.Fair_IRL_Biased_Demonstrations",
            "--config",
            args.config,
            "--results-file",
            results_file,
            "--annotate",
            f"SWEEP_ID={run.sweep_id}",
            "--annotate",
            f"SWEEP_RUN_ID={run.id}",
        ]
        for key, value in overrides.items():
            command += ["--set", f"{key}={json.dumps(value)}"]

        env = {k: v for k, v in os.environ.items() if k not in SWEEP_ENV_VARS}
        returncode = subprocess.run(command, env=env).returncode

        if returncode != 0:
            run.summary["trial_error"] = f"experiment exited with code {returncode}"
            run.finish(exit_code=1)
            sys.exit(returncode)

        with open(results_file) as f:
            results = json.load(f)

    runs = results["runs"]
    metrics = _summarize(runs, args.objective_key)

    run.summary["child_session_id"] = results["session_id"]
    run.summary["objective_key"] = args.objective_key
    run.summary["n_runs"] = len(runs)
    run.summary["n_scored_runs"] = len(_scored_runs(runs))
    run.summary["n_failed_runs"] = sum(not r["converged"] for r in _scored_runs(runs))

    if metrics is None:
        # Leaving the objective unlogged keeps this trial out of the search.
        run.summary["trial_error"] = "no scored runs, or a scored run failed"
        run.finish(exit_code=1)
        sys.exit(1)

    run.log(metrics)
    run.finish()


def _sweep_datasets(args):
    if args.dataset:
        return args.dataset
    return build_experiment_plan(load_config(args.config)).selected_datasets


def create_sweeps(args):
    """Create the sweeps of one sweep config file. See the module docstring."""
    with open(args.sweep_config) as f:
        sweep_config = yaml.safe_load(f)

    datasets = _sweep_datasets(args)
    groups = [datasets] if args.joint else [[dataset] for dataset in datasets]

    project = args.project or WANDB_PROJECT
    entity = args.entity or WANDB_ENTITY

    agent_commands = []
    for group in groups:
        config = copy.deepcopy(sweep_config)
        config["name"] = f"{sweep_config.get('name', 'sweep')} | {', '.join(group)}"
        config["project"] = project
        config["parameters"]["SELECTED_DATASETS"] = {"value": group}
        # A different experiment config than the one the sweep config names.
        if args.config != str(DEFAULT_CONFIG_PATH):
            command = config["command"]
            command[command.index("--config") + 1] = args.config

        if args.dry_run:
            print(yaml.safe_dump(config, sort_keys=False))
            continue

        sweep_id = wandb.sweep(config, entity=entity, project=project)
        path = "/".join(p for p in (entity, project, sweep_id) if p)
        agent_commands.append(f"uv run wandb agent {path}")

    if agent_commands:
        print("\nStart an agent for each sweep (from the repository root):")
        for command in agent_commands:
            print(f"  {command}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser(
        "create",
        help="Create the W&B sweeps of one sweep config file.",
    )
    create.add_argument("sweep_config", help="A file in configs/sweeps/.")
    create.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=(
            "The experiment config file. Sets which datasets get a sweep"
            " when --dataset is not given. Default: %(default)s"
        ),
    )
    create.add_argument(
        "--dataset",
        action="append",
        help=(
            "A dataset to tune on. Repeatable. Default: the experiment"
            " config's SELECTED_DATASETS."
        ),
    )
    create.add_argument(
        "--joint",
        action="store_true",
        help=(
            "Create one sweep that tunes a single set of hyperparameters across"
            " all the datasets, instead of one sweep per dataset."
        ),
    )
    create.add_argument("--project", help=f"Default: {WANDB_PROJECT}")
    create.add_argument("--entity")
    create.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the sweep configs instead of creating the sweeps.",
    )
    create.set_defaults(func=create_sweeps)

    trial = subparsers.add_parser(
        "trial",
        help="Run one sweep trial. Started by the sweep agent, not by hand.",
    )
    trial.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    trial.add_argument(
        "--objective-key",
        default=DEFAULT_OBJECTIVE_KEY,
        choices=SUBDOMINANCE_KEYS,
        help=(
            "The subdominance measurement whose mean over the trial's runs is"
            " the objective. Default: %(default)s"
        ),
    )
    trial.set_defaults(func=run_trial)

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
