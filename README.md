# Research

## Description
This repository was created on May 2, 2019, by Jack Blandin, the previous maintainer. As of summer 2025, maintenence and development has been taken over by me, Charles Dietzel. 
This repo consists of various implementations of both existing and novel machine learning, reinforcement learning, and fairness algorithms, as well as supporting experiments for various published and unpublished works. 

## Setup

The following steps show how to setup the conda environment so that all dependencies are correctly installed, as well as how to run the Jupyter Notebooks in the context of the conda environment.

```sh
# Create the conda environment from the config file
conda env create -f environment.yml

# Activate the conda environment
conda activate research

# Create an IPython kernel which will allow you to run the Jupyter Notebook in the conda environment
python3.6 -m ipykernel install --user --name research

# Start the jupyter notebook
jupyter notebook
```

Then when you're in the Jupyter Notebook, select `Kernel > Change Kernel > research`.

# Publications

Here is the code supporting published papers:

TMLR, 2024:
- [Experiment code](https://github.com/jackblandin/research/blob/master/experiments/fairness/group-fairness-in-rl-through-multi-objective-rewards--experiments.ipynb)
- [Figure generation code](https://github.com/jackblandin/research/blob/master/experiments/fairness/group-fairness-in-rl-through-multi-objective-rewards--heatmaps.ipynb)

FAccT, 2024:
- [Experiment code](https://github.com/jackblandin/research/blob/master/experiments/irl/Fair%20IRL%20Experiments.ipynb)
