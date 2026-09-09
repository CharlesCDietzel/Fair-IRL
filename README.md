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

To reproduce the results, run ```python3 src/fair_irl/Fair_IRL_Biased_Demonstrations.py```

## Techniques

Six techniques can be trained and evaluated:

* **FairIRL Bias Reduction** -- this project's own technique.
* **Superhuman Fairness** -- the ICML 2023 technique of Memarrast et al.
  ([paper](https://proceedings.mlr.press/v202/memarrast23a/memarrast23a.pdf),
  [reference implementation](https://github.com/omidMemari/superhumn-fairness)),
  ported in `src/fair_irl/sh/superhuman_fairness.py`.
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
`Fair_IRL_Biased_Demonstrations.py`; list any combination:

```python
"ALGORITHMS": [
    "FairIRL Bias Reduction",
    "Superhuman Fairness",
    "Post Proc DP",
    "Post Proc EqOdds",
    "Fair LogLoss DP",
    "Fair LogLoss EqOdds",
],
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

The techniques' own parameters live next to `ALGORITHMS` in the same file and
are documented there: the `SH_*` entries for Superhuman Fairness (which
demonstrations it imitates, which performance/fairness measures it optimizes,
its learning rate and iteration count) and the `FAIR_LOGLOSS_*` entries for the
fair-log-loss baselines.

# Figures

Use VSCode or your IDE of choice to view and run the various python notebooks. 

# Publications

Currently, there are no publications that correspond to this code. Watch this space! Or don't, I'm not your dad. 
