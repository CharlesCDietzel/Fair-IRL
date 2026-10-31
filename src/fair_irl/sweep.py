"""
Hyperparameter tuning of FairIRL Bias Reduction and its baselines with W&B
sweeps.

Each technique is tuned by its own sweep, configured by one of the files in
configs/sweeps/: a search over that technique's hyperparameters that
minimizes its mean validation subdominance, by a Bayesian search of 60 runs.
One command does all of it:

    uv run fair-irl-tune [--algorithm NAME] [--dataset NAME]

For every selected technique and dataset (or just the ones named) it creates a
sweep, so that each dataset gets its own hyperparameters, runs W&B agents until
every sweep has finished, and writes configs/experiment.yaml.new: a copy of
configs/experiment.yaml with each sweep's best values set in its dataset's
entry. The techniques are tuned one after another, each running as many of its
sweeps at once as the `parallel_sweeps` of its sweep config says, a number
measured to make that technique's sweeps finish soonest on their own. What it
has created is recorded in `.tuning_state/`, so that running the
same command again after an interruption continues the same sweeps.

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
import functools
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import wandb
import yaml

from fair_irl.config import (
    DEFAULT_CONFIG_PATH,
    build_experiment_plan,
    load_config,
    resolve_tuned_values,
    set_experiment_values,
)

# This module must not import `fair_irl.experiment_utils`: it pulls in PyTorch
# and every model library, close to 1 GB, which a sweep trial -- one per agent,
# alive for as long as its experiment runs -- does not need.

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

REPO_ROOT = DEFAULT_CONFIG_PATH.parents[1]
SWEEP_CONFIG_DIR = REPO_ROOT / "configs" / "sweeps"

# What `fair-irl-tune` writes its results to, next to the config file it copies.
TUNED_CONFIG_PATH = DEFAULT_CONFIG_PATH.with_name(DEFAULT_CONFIG_PATH.name + ".new")

# The W&B project of `fair-irl-tune`'s sweeps. It is not the project the final
# experiments report to because the plotting notebook loads the most recent
# session of its project, and every sweep trial's runs form a session.
TUNING_PROJECT = "fair-irl-tuning"

# Where `fair-irl-tune` keeps the sweeps it has yet to finish, and its agents' logs.
TUNE_STATE_DIR = REPO_ROOT / ".tuning_state"
TUNE_STATE_FILE = TUNE_STATE_DIR / "sweeps.json"
TUNE_LOG_DIR = TUNE_STATE_DIR / "logs"

# How many times in a row a sweep's agents may exit without it having any new
# finished runs before it is given up on.
MAX_AGENT_RESTARTS = 3

# How long `fair-irl-tune` waits between checks of its sweeps.
POLL_SECONDS = 30

# The seconds a stopped agent gets to end its run before it is killed.
AGENT_STOP_SECONDS = 30

# The seconds a run left behind by its agent gets to end before it is killed.
AGENT_TERM_SECONDS = 5

# The states of a W&B sweep that will not start any more runs, and of a W&B run
# that has ended.
ENDED_SWEEP_STATES = ("FINISHED", "CANCELED")
ENDED_RUN_STATES = ("finished", "failed", "crashed", "killed")


def _scored_runs(runs):
    """
    The runs the objective is computed over: every run, except FairIRL Bias
    Reduction's runs with the unadjusted weights. Those are only its starting
    point; its results are the runs of its weight adjustments. The experiment
    script marks which is which, as each run's `scored`.
    """
    return [run for run in runs if run["scored"]]


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

    # Every subdominance measurement the experiment reported, in its order.
    subdominance_keys = list(scored[0]["subdominance"])
    if objective_key not in subdominance_keys:
        raise ValueError(
            f"Unknown objective key {objective_key!r}; the experiment reported"
            f" {subdominance_keys}."
        )

    metrics = {
        OBJECTIVE: float(np.mean([run["subdominance"][objective_key] for run in scored]))
    }
    for key in subdominance_keys:
        metrics[f"mean/{key}"] = float(
            np.mean([run["subdominance"][key] for run in scored])
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
            "--no-pause",
            "--annotate",
            f"SWEEP_ID={run.sweep_id}",
            "--annotate",
            f"SWEEP_RUN_ID={run.id}",
        ]
        for key, value in overrides.items():
            command += ["--set", f"{key}={json.dumps(value)}"]

        env = {k: v for k, v in os.environ.items() if k not in SWEEP_ENV_VARS}
        experiment = subprocess.Popen(command, env=env)
        # An agent stops its run with SIGTERM, which must not leave the
        # experiment running on its own.
        signal.signal(signal.SIGTERM, lambda *_: experiment.terminate())
        returncode = experiment.wait()

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


def _dataset_sweep_config(sweep_config, dataset):
    """The config of the sweep of `sweep_config` on `dataset`."""
    config = copy.deepcopy(sweep_config)
    # How `fair-irl-tune` schedules the sweep, which is none of W&B's business.
    del config["parallel_sweeps"]
    config["name"] = f"{sweep_config['name']} | {dataset}"
    config["project"] = TUNING_PROJECT
    config["parameters"]["SELECTED_DATASETS"] = {"value": [dataset]}
    return config


@dataclass
class TuningSweep:
    """
    One sweep of `fair-irl-tune`, and how far it has got.

    Attributes
    ----------
    name : str
        The sweep's name, "<technique> | <dataset>".
    algorithm : str
        The technique it tunes.
    parallel_sweeps : int
        The `parallel_sweeps` of its sweep config.
    path : str
        The sweep's W&B path, "<entity>/<project>/<id>".
    exp_info : dict
        The `exp_info` of the experiment the sweep tunes.
    tuned_keys : list<str>
        The sweep parameters the search chooses.
    run_cap : float
        How many runs the sweep starts in total.
    n_runs : int
        How many runs it has started so far.
    n_ended : int
        How many of those have ended, however they ended.
    n_finished : int
        How many of those ended successfully.
    agents : list<subprocess.Popen>
        Its agents that are running.
    agent_started : bool
        Whether this `fair-irl-tune` has started an agent for it yet.
    n_restarts : int
        How many times in a row its agents have been restarted.
    checked : bool
        Whether `n_runs` and `n_ended` are up to date.
    finished : bool
        Whether the sweep has run to its end.
    gave_up : bool
        Whether its agents kept exiting before it could.
    """

    name: str
    algorithm: str
    parallel_sweeps: int
    path: str
    exp_info: dict
    tuned_keys: list
    run_cap: float
    n_runs: int = 0
    n_ended: int = 0
    n_finished: int = 0
    agents: list = field(default_factory=list)
    agent_started: bool = False
    n_restarts: int = 0
    checked: bool = False
    finished: bool = False
    gave_up: bool = False

    @property
    def settled(self):
        return self.finished or self.gave_up


def _log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def _load_sweep_configs():
    """The contents of every sweep config, by the technique it tunes."""
    sweep_configs = {}
    for path in sorted(SWEEP_CONFIG_DIR.glob("*.yaml")):
        with open(path) as f:
            sweep_config = yaml.safe_load(f)
        algorithm = sweep_config["parameters"]["ALGORITHMS"]["value"][0]
        sweep_configs[algorithm] = sweep_config
    return sweep_configs


def _tune_targets(args, plan, sweep_configs):
    """
    What `fair-irl-tune` was asked to tune.

    Returns
    -------
    targets : list<tuple<str, dict>>
        Every technique to tune and the `exp_info` of the experiment to tune it
        on. Techniques without a sweep config have nothing to tune.
    """
    exp_infos = {exp_info["DATASET"]: exp_info for exp_info in plan.experiments}

    if args.algorithm and args.algorithm not in sweep_configs:
        sys.exit(
            f"Nothing to tune for algorithm {args.algorithm!r}. Algorithms with"
            f" hyperparameters: {sorted(sweep_configs)}."
        )
    if args.dataset and args.dataset not in exp_infos:
        sys.exit(f"Unknown dataset {args.dataset!r}. Datasets: {sorted(exp_infos)}.")

    targets = []
    for dataset in [args.dataset] if args.dataset else plan.selected_datasets:
        exp_info = exp_infos[dataset]
        for algorithm in [args.algorithm] if args.algorithm else exp_info["ALGORITHMS"]:
            if algorithm in sweep_configs:
                targets.append((algorithm, exp_info))
    return targets


def _load_tune_state():
    if TUNE_STATE_FILE.exists():
        return json.loads(TUNE_STATE_FILE.read_text())
    return {}


def _save_tune_state(state):
    TUNE_STATE_DIR.mkdir(exist_ok=True)
    temporary = TUNE_STATE_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2))
    temporary.replace(TUNE_STATE_FILE)


@functools.cache
def _api():
    # Just the one: each `wandb.Api` stays open in wandb's service process, whose
    # memory would otherwise grow for as long as tuning runs.
    return wandb.Api(timeout=60)


def _fetch_sweep(path):
    sweep = _api().sweep(path)
    # `wandb.Api` keeps the sweeps it has fetched, and `flush()` does not clear
    # them, so this one may be out of date.
    sweep.load(force=True)
    return sweep


def _can_continue(saved, config):
    """Whether a sweep from an earlier `fair-irl-tune` matches `config`."""
    if saved is None or saved["config"] != config:
        return False
    try:
        _fetch_sweep(saved["path"])
    except wandb.errors.CommError:
        _log(f"{saved['path']} is gone from W&B; starting {config['name']} again.")
        return False
    return True


def _open_sweeps(targets, sweep_configs):
    """
    Create the sweeps of `targets`, or find the ones an interrupted
    `fair-irl-tune` left, as long as their sweep config hasn't changed since.
    """
    state = _load_tune_state()
    entity = os.environ.get("WANDB_ENTITY") or _api().default_entity

    sweeps = []
    for algorithm, exp_info in targets:
        sweep_config = sweep_configs[algorithm]
        config = _dataset_sweep_config(sweep_config, exp_info["DATASET"])
        saved = state.get(config["name"])
        if _can_continue(saved, config):
            path = saved["path"]
            _log(f"Continuing sweep {path}: {config['name']}")
        else:
            sweep_id = wandb.sweep(config, entity=entity, project=TUNING_PROJECT)
            path = f"{entity}/{TUNING_PROJECT}/{sweep_id}"
            state[config["name"]] = {"path": path, "config": config}
            _save_tune_state(state)
            _log(f"Created sweep {path}: {config['name']}")

        sweeps.append(
            TuningSweep(
                name=config["name"],
                algorithm=algorithm,
                parallel_sweeps=sweep_config["parallel_sweeps"],
                path=path,
                exp_info=exp_info,
                # A fixed `value` is set by the sweep, not chosen by its search.
                tuned_keys=[
                    key
                    for key, spec in sweep_config["parameters"].items()
                    if "value" not in spec
                ],
                run_cap=sweep_config.get("run_cap", math.inf),
            )
        )
    return sweeps


def _check_sweep(sweep):
    """Bring `sweep` up to date with W&B, and settle it if it has ended."""
    try:
        remote = _fetch_sweep(sweep.path)
        states = [run.state for run in remote.runs]
    except wandb.errors.Error as error:
        # The server being briefly away is no reason to abandon days of work.
        _log(f"Could not check {sweep.name}: {error}")
        return

    n_ended = sum(state in ENDED_RUN_STATES for state in states)
    n_finished = states.count("finished")
    if n_ended > sweep.n_ended:
        _log(f"{sweep.name}: {n_ended}/{sweep.run_cap} runs done")
    # Runs that fail are no sign that restarting the agents is doing any good.
    if n_finished > sweep.n_finished:
        sweep.n_restarts = 0
    sweep.n_runs = len(states)
    sweep.n_ended = n_ended
    sweep.n_finished = n_finished
    sweep.checked = True

    if sweep.agents:
        return
    if remote.state.upper() in ENDED_SWEEP_STATES or sweep.n_runs >= sweep.run_cap:
        # Runs that are still active are waited for, even with no agent left to
        # run them: after a crash, W&B takes minutes to notice they are dead.
        if n_ended == sweep.n_runs:
            sweep.finished = True
            _log(f"{sweep.name}: finished")
    elif sweep.n_restarts >= MAX_AGENT_RESTARTS:
        sweep.gave_up = True
        _log(f"{sweep.name}: giving up, its agents keep exiting. See {TUNE_LOG_DIR}.")


def _start_agent(sweep):
    TUNE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    slug = sweep.name.lower().replace(" | ", "__").replace(" ", "_")
    log_path = TUNE_LOG_DIR / f"{slug}.log"

    env = dict(os.environ)
    # The `python` the sweep configs run must be this environment's.
    env["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{env['PATH']}"
    # Using wandb here starts a service process and records it in WANDB_SERVICE.
    # An agent that inherits that depends on this process staying alive.
    env.pop("WANDB_SERVICE", None)

    with open(log_path, "a") as log_file:
        sweep.agents.append(
            subprocess.Popen(
                [sys.executable, "-m", "wandb", "agent", sweep.path],
                # The sweep configs name the experiment config relative to the
                # repository, and the datasets are loaded relative to it too.
                cwd=REPO_ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                # Stopping agents is left to `_stop_agents()`: a signal from the
                # terminal would reach the agents, but not the runs they started.
                start_new_session=True,
            )
        )
    _log(f"{sweep.name}: started an agent, logging to {log_path}")


def _start_agents(sweeps, parallel_sweeps):
    """
    Start agents until `parallel_sweeps` are running. Every sweep gets one
    before any gets a second, so that many sweeps run at once if that many are
    left. With fewer left, the spare agents join the ones that are, rather than
    leaving the slowest sweep to finish alone. A sweep doesn't get more agents
    than it has runs left.
    """
    n_agents = sum(len(sweep.agents) for sweep in sweeps)
    while n_agents < parallel_sweeps:
        candidates = [
            sweep
            for sweep in sweeps
            if sweep.checked
            and not sweep.settled
            and sweep.run_cap - sweep.n_runs > len(sweep.agents)
        ]
        if not candidates:
            return
        sweep = min(candidates, key=lambda candidate: len(candidate.agents))
        if sweep.agent_started and not sweep.agents:
            sweep.n_restarts += 1
        _start_agent(sweep)
        sweep.agent_started = True
        n_agents += 1


def _run_agents(sweeps, parallel_sweeps):
    """Run agents until every sweep has either finished or been given up on."""
    while not all(sweep.settled for sweep in sweeps):
        for sweep in sweeps:
            sweep.agents = [agent for agent in sweep.agents if agent.poll() is None]
            if not sweep.settled:
                _check_sweep(sweep)
        _start_agents(sweeps, parallel_sweeps)
        if not all(sweep.settled for sweep in sweeps):
            time.sleep(POLL_SECONDS)


def _stop_agents(sweeps):
    """
    Stop every agent and its run. An agent takes a first SIGINT as asking it to
    wait for its run to end, and a second one as asking it to terminate it. A
    busy agent can fail to do that, so whatever is left is signaled directly:
    each agent leads a session, which its run stays in after the agent is gone.
    """
    agents = [agent for sweep in sweeps for agent in sweep.agents]
    if not agents:
        return

    for _ in range(2):
        for agent in agents:
            if agent.poll() is None:
                agent.send_signal(signal.SIGINT)
        time.sleep(1)

    deadline = time.monotonic() + AGENT_STOP_SECONDS
    for agent in agents:
        try:
            agent.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            agent.kill()

    sessions = ",".join(str(agent.pid) for agent in agents)
    for signal_name in ("TERM", "KILL"):
        # pkill fails if there was nothing left to signal.
        if subprocess.run(["pkill", f"-{signal_name}", "-s", sessions]).returncode:
            break
        time.sleep(AGENT_TERM_SECONDS)


def _best_run(sweep):
    """The finished run of `sweep` with the lowest objective, or None."""
    runs = [
        run
        for run in _fetch_sweep(sweep.path).runs
        if run.state == "finished" and run.summary.get(OBJECTIVE) is not None
    ]
    return min(runs, key=lambda run: run.summary[OBJECTIVE], default=None)


def _write_tuned_config(sweeps):
    """
    Write `TUNED_CONFIG_PATH`: the experiment config as it is now, with the
    best values of `sweeps` set in the entries of their datasets.

    Returns
    -------
    n_written : int
        How many of `sweeps` had a best run to write.
    """
    text = DEFAULT_CONFIG_PATH.read_text()
    n_written = 0
    for sweep in sweeps:
        best = _best_run(sweep)
        if best is None:
            _log(f"{sweep.name}: no run was scored, so it has nothing to write.")
            continue
        values = {key: best.config[key] for key in sweep.tuned_keys}
        text = set_experiment_values(
            text,
            sweep.exp_info["EXPERIMENT_NAME"],
            resolve_tuned_values(sweep.exp_info, values),
        )
        n_written += 1
        _log(f"{sweep.name}: best objective {best.summary[OBJECTIVE]:.6g}, {values}")

    if n_written:
        TUNED_CONFIG_PATH.write_text(text)
    return n_written


def _forget_sweeps(sweeps):
    """Drop `sweeps` from the state, so that their results aren't reused."""
    state = _load_tune_state()
    for sweep in sweeps:
        state.pop(sweep.name, None)
    if state:
        _save_tune_state(state)
    else:
        TUNE_STATE_FILE.unlink(missing_ok=True)


def parse_tune_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="fair-irl-tune",
        description=(
            "Tune hyperparameters with W&B sweeps: create the sweeps, run their"
            f" agents, and write the best values found to {TUNED_CONFIG_PATH.name}"
            f" next to {DEFAULT_CONFIG_PATH.name}. Running it again after an"
            " interruption continues the same sweeps."
        ),
    )
    parser.add_argument(
        "--algorithm",
        help="Only tune this algorithm. Default: every one selected in the config.",
    )
    parser.add_argument(
        "--dataset",
        help="Only tune on this dataset. Default: every one selected in the config.",
    )
    return parser.parse_args(argv)


def _interrupt(signum, frame):
    raise KeyboardInterrupt


def tune_main(argv=None):
    """Tune hyperparameters from start to finish. See the module docstring."""
    args = parse_tune_args(argv)

    plan = build_experiment_plan(load_config())
    sweep_configs = _load_sweep_configs()
    targets = _tune_targets(args, plan, sweep_configs)
    if not targets:
        sys.exit("Nothing to tune: no selected algorithm has hyperparameters.")

    sweeps = _open_sweeps(targets, sweep_configs)

    # However this process is told to stop, its agents have to stop with it.
    # Handling SIGINT here too means the agents can receive it even if this
    # process was started in the background, which would have them ignore it.
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, _interrupt)
    try:
        # One technique at a time, because each one's `parallel_sweeps` is the
        # number that suits it with the machine to itself.
        for algorithm in dict.fromkeys(sweep.algorithm for sweep in sweeps):
            algorithm_sweeps = [s for s in sweeps if s.algorithm == algorithm]
            parallel_sweeps = algorithm_sweeps[0].parallel_sweeps
            _log(f"Tuning {algorithm}, up to {parallel_sweeps} sweeps at once")
            _run_agents(algorithm_sweeps, parallel_sweeps)
    except KeyboardInterrupt:
        _log("Stopping the agents. Run the same command again to continue.")
        _stop_agents(sweeps)
        sys.exit(130)

    finished = [sweep for sweep in sweeps if sweep.finished]
    n_written = _write_tuned_config(finished)
    _forget_sweeps(finished)

    if n_written:
        _log(f"Wrote the best values of {n_written} sweeps to {TUNED_CONFIG_PATH}")
    n_failed = len(sweeps) - n_written
    if n_failed:
        sys.exit(f"{n_failed} of {len(sweeps)} sweeps did not produce a result.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    trial = subparsers.add_parser(
        "trial",
        help="Run one sweep trial. Started by the sweep agent, not by hand.",
    )
    trial.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    trial.add_argument(
        "--objective-key",
        default=DEFAULT_OBJECTIVE_KEY,
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
