# Research

## Description
This repository was created on May 2, 2019, by Jack Blandin, the previous maintainer. As of summer 2025, maintenence and development has been taken over by me, Charles Dietzel. 
This repo consists of various implementations of both existing and novel machine learning, reinforcement learning, and fairness algorithms, as well as supporting experiments for various published and unpublished works. 

## Setup

The following steps show how to setup a uv environment so that all dependencies are correctly installed.

```sh
# Create the uv environment from the pyproject.toml config file
uv sync

# Activate the environment
source .venv/bin/activate 
```

Use VSCode or your IDE of choice to view the various python notebooks. 

# Publications

Here is the code supporting published papers:

TMLR, 2024:
- [Experiment code](https://github.com/jackblandin/research/blob/master/experiments/fairness/group-fairness-in-rl-through-multi-objective-rewards--experiments.ipynb)
- [Figure generation code](https://github.com/jackblandin/research/blob/master/experiments/fairness/group-fairness-in-rl-through-multi-objective-rewards--heatmaps.ipynb)

FAccT, 2024:
- [Experiment code](https://github.com/jackblandin/research/blob/master/experiments/irl/Fair%20IRL%20Experiments.ipynb)
