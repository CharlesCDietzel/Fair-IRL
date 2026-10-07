# Research

## Description
This repository was created on May 2, 2019, by Jack Blandin, the previous maintainer. As of summer 2025, maintenence and development has been taken over by me, Charles Dietzel. 
This repo consists of various implementations of both existing and novel machine learning, reinforcement learning, and fairness algorithms, as well as supporting experiments for various published and unpublished works. 

## Setup

The following steps show how to setup a uv environment so that all dependencies are correctly installed. If you don't have uv installed, you really should install and start using it for your own projects. It is by far the least crappy python package manager. 

```sh
# Create the uv environment from the pyproject.toml config file
uv sync

# Activate the environment
source .venv/bin/activate 
```

Once you have created and sourced the environment, you will also need to set up and start the wandb server. To do this, first install docker engine following the instructions here: https://docs.docker.com/engine/install/ubuntu/. NOTE: These instructions assume you are using Ubuntu. If not, google for how to install docker engine. If you are using Windows, idk man, figure it out. 

After docker is installed and the docker service is started, run ```wandb server start```. This will pull and automatically start the wandb server inside a docker container. 

Once the command finishes, you will see a terminal prompt asking you to provide an API key. Click the URL in the terminal (which will look something like http://localhost:8080/login) to go to a screen where you will need to make an account. Don't worry, the account will only be created on your local machine and will not exist exist on the internet. 

After you make your account, wandb will then give you your API key, which you should copy and paste into the terminal prompt. After you have done this, congratulations! You have completed the setup. 

IMPORTANT NOTE: You will need to re-run ```wandb server start``` each time you restart your computer if you want to run this code. 

# Reproducing Results

Generating the results from scratch takes three steps, all run from the repository root
with the W&B server running:

1. [Tune](#1-tuning-hyperparameters) the hyperparameters of FairIRL Bias
   Reduction and of each baseline with W&B sweeps.
2. [Record](#2-recording-the-tuned-hyperparameters) the best values found in
   `configs/experiment.yaml`.
3. [Run](#3-running-the-final-experiments) the final experiments with
   `uv run fair-irl`.

In order to reproduce the results from the paper, the tuned hyperparameters are already
set up correctly, so you only need to [run](#3-running-the-final-experiments) the
experiment code. 

After the experiment is run, see [here](#figures) for how to generate the paper's plots.


## The experiment config

Every experiment setting lives in `configs/experiment.yaml`. That covers which
datasets, experts and techniques are run, and every technique's
hyperparameters. Its comments document each setting. The file has four parts:

* `SELECTED_DATASETS`, `EXPERT_ALGOS` and `RANDOM_SEED` at the top: which
  experiments run, with which experts, and the global random seed.
* `common`: the settings shared by every experiment.
* `presets`: named groups of settings that an experiment can opt into. The only
  one is `sh_paper`, the Superhuman Fairness paper's conditions (see
  [below](#reproducing-the-superhuman-fairness-paper)).
* `experiments`: one entry per dataset, holding that dataset's own settings.

Each experiment's settings are `common`, then its presets, then its own entry.
Each later layer overrides the earlier ones, so a value set in an
experiment's entry applies to that dataset only.

Run the experiments with either of these commands:

```sh
uv run fair-irl
python3 src/fair_irl/Fair_IRL_Biased_Demonstrations.py  # with .venv activated
```

Use `--config <file>` to run a different config file, or override single
values for every experiment with `--set`. `--set` reads each value as JSON, so
quote lists:

```sh
uv run fair-irl --set 'SELECTED_DATASETS=["COMPAS"]' --set SH_ITERS=10
```

`--set` only accepts keys the config file already contains, so a misspelled
key is an error rather than being silently ignored.

## 1. Tuning hyperparameters

Each technique with hyperparameters has a W&B sweep config in
`configs/sweeps/`. Each sweep tunes only that technique's settings. Every other
setting comes from `configs/experiment.yaml`.

| Technique | Sweep config | Search | Tuned settings |
| --- | --- | --- | --- |
| FairIRL Bias Reduction | `fairirl_bias_reduction.yaml` | grid (all 21 combinations) | `METHOD`, `OPT_DEBIAS_OPTIMIZER` |
| Superhuman Fairness | `superhuman_fairness.yaml` | Bayesian, 50 runs | `SH_ITERS`, `SH_LR_THETA`, `SH_LAMDA` |
| Superhuman Fairness Fixed | `superhuman_fairness_fixed.yaml` | Bayesian, 50 runs | `SH_FIXED_ITERS`, `SH_FIXED_LR_THETA`, `SH_FIXED_LAMDA` |
| Superhuman Fairness Neural Network | `superhuman_fairness_nn.yaml` | Bayesian, 60 runs | `SH_NN_*` (all but `SH_NN_DEVICE`) |
| Superhuman Fairness Neural Network Fixed | `superhuman_fairness_nn_fixed.yaml` | Bayesian, 60 runs | `SH_NN_FIXED_*` (all but `SH_NN_FIXED_DEVICE`) |
| Fair LogLoss DP | `fair_logloss_dp.yaml` | Bayesian, 30 runs | `FAIR_LOGLOSS_DP_C`, `FAIR_LOGLOSS_DP_RANDOM_INIT` |
| Fair LogLoss EqOdds | `fair_logloss_eqodds.yaml` | Bayesian, 30 runs | `FAIR_LOGLOSS_EQODDS_C`, `FAIR_LOGLOSS_EQODDS_RANDOM_INIT` |

`Post Proc DP` and `Post Proc EqOdds` have no hyperparameters.

Every sweep minimizes the same objective, the technique's mean validation
subdominance (`sum_abs_subdominance_val`). The test split is never used for
tuning. FairIRL Bias Reduction is scored on its weight-adjusted runs only, not
on the unadjusted weights it starts from. If any of a trial's runs fail, the
trial is left out of the search, so hyperparameters that make a technique fail
on its hard cases are never chosen.

To tune a technique, create its sweeps, then start a W&B agent for each sweep
id that `create` prints:

```sh
uv run python -m fair_irl.sweep create configs/sweeps/<file>.yaml --project fair-irl-tuning
uv run wandb agent <entity>/fair-irl-tuning/<sweep id>
```

By default, `create` makes one sweep per dataset in `SELECTED_DATASETS`, so
each dataset gets its own hyperparameters. Other options:

* `--joint` creates one sweep that tunes a single set of hyperparameters
  across all the datasets.
* `--dataset <name>` (repeatable) picks the datasets yourself.
* `--dry-run` prints the sweep configs without creating them.

Several agents can work on the same sweep at once. Run the agents from the
repository root, because the datasets are loaded from relative paths.

Use `--project` to give the sweeps their own W&B project. Each sweep trial is a
W&B run of its own, and the per-technique runs it reports go to the same
project. Each of those runs records which trial launched it in its `SWEEP_ID`
and `SWEEP_RUN_ID` config. Keeping all of this out of the main project matters
because the plotting notebook loads the most recent session in its project by
default, and a sweep trial's runs form a session too.

A few things to keep in mind:

* Each trial runs whatever `configs/experiment.yaml` says when the trial
  starts. Don't edit the file while a sweep is running, or later trials will be
  scored under different conditions than earlier ones.
* Every trial runs `N_TRIALS` trials of each dataset bias type in
  `DATASET_BIAS_TYPE_LIST`, so a sweep costs as much as that many final runs,
  times the number of sweep runs.
* The FairIRL grid sweep finishes on its own after its 21 runs. Each Bayesian
  sweep stops at the run cap in the table above. To change a cap, edit the
  `run_cap` in the sweep config before creating the sweep.

See `src/fair_irl/sweep.py` for the details of how a trial is run.

## 2. Recording the tuned hyperparameters

In the W&B UI, open a sweep and sort its runs by `objective` (lower is
better). The best run's config holds the values to use. Each run also logs
`objective/<dataset>`, the objective broken down by dataset, which is useful
for `--joint` sweeps.

Record those values in `configs/experiment.yaml` and remove their
`TODO: Tune this hyperparameter` comments as you go:

* Values from a per-dataset sweep go in that dataset's entry under
  `experiments`, where they override `common` and any preset:

  ```yaml
  experiments:
    COMPAS:
      MIN_FREQ_FILL_PCT: 0.0
      SH_ITERS: 12
      SH_LR_THETA: 0.0034
      FAIR_LOGLOSS_DP_C: 0.021
  ```

* Values from a `--joint` sweep replace the defaults in `common`.
* FairIRL Bias Reduction's `OPT_DEBIAS_OPTIMIZER` is a sweep-only shorthand,
  not a config setting. Record its value by editing the `opt_debias` entry of
  `WEIGHT_ADJUST_LIST` instead. For example, `nevergrad/Powell` becomes
  `[opt_debias, nevergrad, Powell, 500]`.

## 3. Running the final experiments

Set `N_TRIALS` in `common` to the number of trials to average over. 3 is the
number used for the paper results. Check that `SELECTED_DATASETS` lists every
dataset to report on, then run:

```sh
uv run fair-irl
```

The results go to the W&B project `fair-irl`, or to the project named by the
`WANDB_PROJECT` environment variable. The script logs a W&B session id when it
starts and again when it finishes. To plot that session, set `WANDB_SESSION` in
the plotting notebook to that id; the default is the most recent session in
the project.

## Techniques

Nine techniques can be trained and evaluated:

* **FairIRL Bias Reduction** -- this project's own technique.
* **Superhuman Fairness** -- the ICML 2023 technique of Memarrast et al.
  ([paper](https://proceedings.mlr.press/v202/memarrast23a/memarrast23a.pdf),
  [reference implementation](https://github.com/omidMemari/superhumn-fairness)),
  ported in `src/fair_irl/sh/superhuman_fairness.py`.
* **Superhuman Fairness Fixed** -- Superhuman Fairness with the normalization
  of its gradient's feature-matching term fixed. Upstream normalizes the
  expected feature vector by the whole training pool's size but each
  demonstration's by the demonstration's size, which adds a component shared by
  every demonstration that dominates the gradient and drifts the model toward
  predicting every label 0. This variant normalizes both by the
  demonstration's size, so it learns from the demonstration-specific signal
  only. It shares every `SH_*` setting except its own `SH_FIXED_ITERS`,
  `SH_FIXED_LR_THETA` and `SH_FIXED_LAMDA`, so that it can be tuned separately,
  and imitates exactly the same demonstrations as Superhuman Fairness does.
* **Superhuman Fairness Neural Network** -- the neural network version of
  Superhuman Fairness from the reference implementation's
  [`reorg_current` branch](https://github.com/omidMemari/superhumn-fairness/tree/reorg_current),
  ported in `src/fair_irl/sh/superhuman_fairness_nn.py`. It shares the
  `SH_*` settings except its own `SH_NN_*` ones (including `SH_NN_LR_THETA`,
  which here is the network's Adam learning rate), and imitates the same
  demonstrations as Superhuman Fairness. It trains with PyTorch, on a GPU when
  PyTorch can see one and on the CPU otherwise (see `SH_NN_DEVICE`).
* **Superhuman Fairness Neural Network Fixed** -- Superhuman Fairness Neural
  Network with its training loss fixed. Upstream's loss weights each
  demonstration's total probability of predicting 1 by its (positive)
  subdominance, so every step lowers every probability of predicting 1 and the
  model drifts toward predicting every label 0, and its gradient never sees
  the sampled decisions. This variant weights the log-probability of each
  demonstration's sampled decisions by its subdominance minus the mean over
  demonstrations: the score-function estimate of the gradient of the expected
  subdominance, with the shared direction removed. It has its own
  `SH_NN_FIXED_*` counterpart of every `SH_NN_*` setting.
* **Post Proc DP** and **Post Proc EqOdds** -- the post-processing model of
  Hardt et al. (2016), with demographic parity and with equalized odds as the
  fairness constraint.
* **Fair LogLoss DP** and **Fair LogLoss EqOdds** -- the robust fair-log-loss
  model of Rezaei et al. (2020), with the same two constraints.

The last four are the fair-classification baselines the Superhuman Fairness
paper compares itself against, ported from the same repository into
`src/fair_irl/sh/baselines.py` (with the Rezaei et al. classifiers vendored
verbatim in `src/fair_irl/sh/fair_logloss.py`).

That paper's remaining baseline, **MFOpt** (Hsu et al., 2022), is not
available. Its reference repository ships no implementation of it -- only CSVs
of predictions its authors produced elsewhere -- and Hsu et al. published no
code, so there is nothing to port and any implementation here would be a
reconstruction of the paper rather than the published method.

Which techniques a run covers is the `ALGORITHMS` list in
`configs/experiment.yaml`; list any combination:

```yaml
ALGORITHMS:
  - FairIRL Bias Reduction
  - Superhuman Fairness
  - Superhuman Fairness Fixed
  - Superhuman Fairness Neural Network
  - Superhuman Fairness Neural Network Fixed
  - Post Proc DP
  - Post Proc EqOdds
  - Fair LogLoss DP
  - Fair LogLoss EqOdds
```

Every technique is trained on the same dataset, the same injected label bias
and the same train/validation/test split, and is evaluated by the same code, so
their metrics are directly comparable. Each reports its own W&B run, tagged and
configured with its `ALGORITHM`; the plotting notebook selects one with its
`selected_algorithm` variable.

Each baseline draws from its own deterministically seeded random generator
rather than the global one, so enabling any of them leaves the FairIRL results
bit-identical to a FairIRL-only run. A baseline that fails on some split (the
fair-log-loss optimizer can) is logged, its run marked `converged = False` --
which is what the plotting notebook filters on -- and the trial carries on with
the remaining techniques.

The techniques' own parameters live next to `ALGORITHMS` in the same config
file and are documented there: the `SH_*` entries for Superhuman Fairness (which
demonstrations it imitates, which performance/fairness measures it optimizes,
its learning rate and iteration count) and the `FAIR_LOGLOSS_DP_*` and
`FAIR_LOGLOSS_EQODDS_*` entries for the two fair-log-loss baselines, which are
configured separately.

### Reproducing the Superhuman Fairness paper

The Superhuman Fairness paper's own versions of its two datasets are available as the `Adult_SH`
and `COMPAS_SH` datasets. They are the reference implementation's
already-encoded `dataset/<name>/dataset_ref.csv` files, which this repository
does not track; copy them from
[its repository](https://github.com/omidMemari/superhumn-fairness) to
`data/superhuman_fairness/Adult/dataset_ref.csv` and
`data/superhuman_fairness/COMPAS/dataset_ref.csv`. They differ from this
project's `Adult` and `COMPAS` in their rows, encoding, label and protected
attribute (see their loaders in `src/fair_irl/datasets.py`).

Listing either in `SELECTED_DATASETS` runs it under the paper's experimental
conditions, which the `sh_paper` preset in `configs/experiment.yaml` sets: the paper's post-processing demonstrations, measures, hyperparameters and
stratified half/half split, plus, for COMPAS, the learning rate
(`lr_theta = 0.0001`), iteration count (5) and fair log-loss initialization
its runs used. Then set `plot_demo_source = "superhuman"` and
`fair_logloss_paper_metrics = True` in the plotting notebook to draw the
figures the way the paper does.

The Superhuman Fairness paper's own demonstrator -- the post-processing model
its `SH_DEMO_SOURCE = "pp_baseline"` setting learns from -- is also available
to the FairIRL Bias Reduction technique as the `PostProcDemo` entry of
`EXPERT_ALGOS`. It takes its fairness constraint from `SH_DEMO_CONSTRAINTS`,
so selecting it makes both techniques imitate the same demonstrator.

# Figures

Use VSCode or your IDE of choice to view and run the various python notebooks. 
All figures and plots found in the paper are generated by [this notebook](notebooks/irl/Fair_IRL_Biased_Demonstrations_Plotting.ipynb).

# Publications

Currently, there are no publications that correspond to this code. Watch this space! Or don't, I'm not your dad. 
