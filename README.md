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

Once you have created and sourced the environment, you will also need to set up and start the wandb server. To do this, first install docker engine following the instructions here: https://docs.docker.com/engine/install/ubuntu/. NOTE: These instructions assume you are using Ubuntu. If not, google for how to install docker engine. If you are using Windows, idk man, figure it out. 

After docker is installed and the docker service is started, run ```wandb server start```. This will pull and automatically start the wandb server inside a docker container. 

Once the command finishes, you will see a terminal prompt asking you to provide an API key. Click the URL in the terminal (which will look something like http://localhost:8080/login) to go to a screen where you will need to make an account. Don't worry, the account will only be created on your local machine and will not exist exist on the internet. 

After you make your account, wandb will then give you your API key, which you should copy and paste into the terminal prompt. After you have done this, congratulations! You have completed the setup. 

IMPORTANT NOTE: You will need to re-run ```wandb server start``` each time you restart your computer if you want to run this code. 

# Reproducing Results

To reproduce the results, run ```python3 src/fair_irl/Fair_IRL_Biased_Demonstrations.py```

# Figures

Use VSCode or your IDE of choice to view and run the various python notebooks. 

# Publications

Currently, there are no publications that correspond to this code. Watch this space! Or don't, I'm not your dad. 
