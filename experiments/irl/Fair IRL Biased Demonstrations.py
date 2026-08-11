# Add parent directory to current path. Needed for research imports.
import os.path
import sys
import random
import subprocess
from pathlib import Path
import cProfile
import pstats

try:
    # If running as a script, __file__ is available
    p = Path(__file__).resolve().parent.parent.parent
except NameError:
    # Fallback for Jupyter Notebook execution
    p = Path(os.getcwd()).resolve().parent.parent
if p not in sys.path:
    sys.path.insert(0, str(p))

import logging
import numpy as np
import pandas as pd

# TODO: Remove this once we are on pandas 3.0
pd.options.mode.copy_on_write = "warn"

import warnings

from experiments.irl.datasets import *
from experiments.irl.experiment_utils import *
from research.irl.fair_irl import *
from research.utils import *

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")

from IPython.display import display, HTML

display(HTML("<style>.container { width:2000px !important; }</style>"))
pd.set_option("display.max_columns", None)
# pd.set_option('display.max_colwidth', None)

# Prevent long logging lines from wrapping
# display(HTML("<style>div.output_area pre {white-space: pre;}</style>"))
np.set_printoptions(linewidth=np.inf)

random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)


# Common Config
exp_list = []
common_exp_info = {
    "FEAT_EXP_OBJECTIVE_NAMES": [
        "Acc",
        "AccPar",
        "DemPar",
        "EqOpp",
        # "FPRPar",
        # "EqOdds",
        "TNRPar",
        # "FNRPar",
        # "PR_Z0",
        # "PR_Z1",
        # "NR_Z0",
        # "NR_Z1",
        # "TPR_Z0",
        # "TPR_Z1",
        # "TNR_Z0",
        # "TNR_Z1",
        # "FPR_Z0",
        # "FPR_Z1",
        # "FNR_Z0",
        # "FNR_Z1",
    ],
    "PERF_MEAS_OBJECTIVE_NAMES": [
        "Acc",
        "AccPar",
        "DemPar",
        "EqOpp",
        "FPRPar",
        "EqOdds",
        "TNRPar",
        "FNRPar",
        # "PR_Z0",
        # "PR_Z1",
        # "TPR_Z0",
        # "TPR_Z1",
        # "TNR_Z0",
        # "TNR_Z1",
        # "FPR_Z0",
        # "FPR_Z1",
        # "FNR_Z0",
        # "FNR_Z1",
        "PredPar",
        "NegPredPar",
    ],
    # Expert demo parameters
    #     'DATASET': 'ACSIncome__CA',
    #     'TARGET_DATASET': 'ACSIncome__IL',
    "EXPERT_CANNOT_PREDICT_IN_TARGET": False,
    "USE_HIDDEN_FEATURES_SOURCE": True,
    "USE_HIDDEN_FEATURES_TARGET": False,
    "N_EXPERT_DEMOS": 1,
    "EXPERT_ALGO": None,
    # "MIN_FREQ_FILL_PCT": 0.3,
    "RESTRICT_Y_ACTION": True,
    # IRL Loop parameters
    "IRL_METHOD": None,
    "METHOD": "highs",
    # Plotting parameters
    "NOISE_FACTOR": 0.01,
    "ANNOTATE": True,
    "NON_EXPERT_ALGOS": [
        # 'OptAccNoisy',
        # 'HardtDemParNoisy',
        # 'HardtEqOppNoisy',
        # 'HardtFPRNoisy',
        # 'HardtTNRNoisy',
        # "Dummy",
        # 'DummyNoisy',
        # "DegradeAbsolute",
        "DegradeRelative",
        # "DegradeNoisy",
        # "DegradeNoisyAbsolute",
        # "DegradeNoisyRelative",
    ],
    "N_TRIALS": 1,  # TODO: CHANGE THIS BACK TO 3 FOR FINAL PAPER RESULTS
    "N_SUBDOMINANCE_GROUPS": 100,
    "EPSILON": 0.005,
    "IGNORE_RESULTS_EPSILON": np.inf,
    "MAX_ITER": 40,
    "ALLOW_NEG_WEIGHTS": True,
    "DOT_WEIGHTS_FEAT_EXP": True,
    "EARLY_STOP_NO_NEW_BEST_ITERS": 15,
    "N_DATASET_SAMPLES": None,
    "RANDOM_SEED": random_seed,
}

# ### COMPAS
base_exp_info = {
    "EXPERIMENT_NAME": "COMPAS",
    "MIN_FREQ_FILL_PCT": 0.08,  # MIN_FREQ_FILL_PCT values have been chosen
    # so that the runtime per irl loop is roughly equal for all datasets
    # "N_SUBDOMINANCE_GROUPS": 20,  # N_SUBDOMINANCE_GROUPS values have been
    # chosen so that none of the feature expectations before or after weight
    # adjustment have zero values.
    # (That would cause issues for the subdominance calculations)
}
base_exp_info |= common_exp_info

source_states = [
    "COMPAS",
]

exp_dict = {}
exp_dict["base_exp_info"] = base_exp_info
exp_dict["source_states"] = source_states
exp_list.append(exp_dict)

# Boston (DISABLED, SINCE NOT ENOUGH DATA FOR GOOD SUBDOMINANCE CALCS)
# base_exp_info = {
#     'EXPERIMENT_NAME': 'Boston',
#     "MIN_FREQ_FILL_PCT": 0.0,
#     "N_SUBDOMINANCE_GROUPS": 2,  # Even with only 2 subdominance groups,
#     # Boston is small and doesn't quite have enough samples to properly
#     # compute fairness features in some instances. However, this is the
#     # best we can do without reducing the number of subdominance groups
#     # to 1, and the compute_alphas() function requires at least 2.
#     # The good news is, since we are using sum-aggregated absolute
#     # subdominance, the effect of a few bad fairness feature estimates
#     # should be minimal. Therefore, we choose to keep this dataset in.
#     # Nevermind, taking it out.
# }
# base_exp_info |= common_exp_info

# source_states = [
#     'Boston',
# ]

# exp_dict = {}
# exp_dict["base_exp_info"] = base_exp_info
# exp_dict["source_states"] = source_states
# exp_list.append(exp_dict)

# Adult
base_exp_info = {
    "EXPERIMENT_NAME": "Adult",
    "MIN_FREQ_FILL_PCT": 0.24,
    # "N_SUBDOMINANCE_GROUPS": 10,
}
base_exp_info |= common_exp_info

source_states = [
    "Adult",
]

exp_dict = {}
exp_dict["base_exp_info"] = base_exp_info
exp_dict["source_states"] = source_states
exp_list.append(exp_dict)

# ACSIncome: MA
base_exp_info = {
    "EXPERIMENT_NAME": "ACSIncome__MA",
    "MIN_FREQ_FILL_PCT": 0.07,
    # "N_SUBDOMINANCE_GROUPS": 20,
}
base_exp_info |= common_exp_info

source_states = [
    "ACSIncome__MA",
]

exp_dict = {}
exp_dict["base_exp_info"] = base_exp_info
exp_dict["source_states"] = source_states
exp_list.append(exp_dict)

# ACSIncome: MS
base_exp_info = {
    "EXPERIMENT_NAME": "ACSIncome__MS",
    "MIN_FREQ_FILL_PCT": 0.05,
    # "N_SUBDOMINANCE_GROUPS": 10,
}
base_exp_info |= common_exp_info

source_states = [
    "ACSIncome__MS",
]

exp_dict = {}
exp_dict["base_exp_info"] = base_exp_info
exp_dict["source_states"] = source_states
exp_list.append(exp_dict)

# ACSIncome: CA
base_exp_info = {
    "EXPERIMENT_NAME": "ACSIncome__CA",
    "MIN_FREQ_FILL_PCT": 0.2,
    # "N_SUBDOMINANCE_GROUPS": 20,
}
base_exp_info |= common_exp_info

source_states = [
    "ACSIncome__CA",
]

exp_dict = {}
exp_dict["base_exp_info"] = base_exp_info
exp_dict["source_states"] = source_states
exp_list.append(exp_dict)

# ACSIncome: IL
base_exp_info = {
    "EXPERIMENT_NAME": "ACSIncome__IL",
    "MIN_FREQ_FILL_PCT": 0.09,
    # "N_SUBDOMINANCE_GROUPS": 50,
}
base_exp_info |= common_exp_info

source_states = [
    "ACSIncome__IL",
]

exp_dict = {}
exp_dict["base_exp_info"] = base_exp_info
exp_dict["source_states"] = source_states
exp_list.append(exp_dict)

# ACSIncome: AL
base_exp_info = {
    "EXPERIMENT_NAME": "ACSIncome__AL",
    "MIN_FREQ_FILL_PCT": 0.06,
    # "N_SUBDOMINANCE_GROUPS": 20,
}
base_exp_info |= common_exp_info

source_states = [
    "ACSIncome__AL",
]

exp_dict = {}
exp_dict["base_exp_info"] = base_exp_info
exp_dict["source_states"] = source_states
exp_list.append(exp_dict)

# ACSIncome: HI
base_exp_info = {
    "EXPERIMENT_NAME": "ACSIncome__HI",
    "MIN_FREQ_FILL_PCT": 0.1,
    # "N_SUBDOMINANCE_GROUPS": 10,
}
base_exp_info |= common_exp_info

source_states = [
    "ACSIncome__HI",
]

exp_dict = {}
exp_dict["base_exp_info"] = base_exp_info
exp_dict["source_states"] = source_states
exp_list.append(exp_dict)

# Set Experts
expert_algos = [
    "OptAcc",
    # "CatBoostOptAcc",
    # "XGBoostOptAcc",
    # "HardtDemPar",
    # "HardtEqOpp",
    # "HardtTNRPar",
    # "HardtFPRPar",
    # "HardtFNRPar",
    # "HardtEqOdds",
    # 'BoundedGroupLoss',
    # 'COMPAS',
]

ALL_FEAT_PERF_OBJECTIVE_NAMES = [
    "Acc",
    "AccPar",
    "DemPar",
    "EqOpp",
    "FPRPar",
    "EqOdds",
    "TNRPar",
    "FNRPar",
    "PR_Z0",
    "PR_Z1",
    "NR_Z0",
    "NR_Z1",
    "TPR_Z0",
    "TPR_Z1",
    "TNR_Z0",
    "TNR_Z1",
    "FPR_Z0",
    "FPR_Z1",
    "FNR_Z0",
    "FNR_Z1",
    "PredPar",
    "NegPredPar",
]

#  all available bias types = ((), ("threshold_swapping",), ("unbalanced_redlining",), ("balanced_redlining",), ("perfectly_balanced_redlining",), ("broken_redlining"))
bias_types_list = (
    (),
    # ("threshold_swapping",), # This doesn't work for non postprocessing classifiers,
    # and also generates different kinds of bias for each type of Postprocessing classifier,
    # so don't use it for the final paper results.
    # (("unbalanced_redlining", 0.2),),
    (("balanced_redlining", 0.2),),
    # (("perfectly_balanced_redlining", 0.2),),
    # (("corruption_bias", "CatBoost", 0.001, "gaussian", 1.0),),
)
# bias_types_list = (("perfectly_balanced_redlining"))

weight_adjusts_list = (
    (),  # need initial empty list for this one, otherwise weight adjustment code won't work correctly
    # (("mul_negative_weights", 0.0),),
    # (("mul_negative_weights", 0.1),),
    # (("mul_negative_weights", 0.2),),
    # (("mul_negative_weights", 0.3),),
    # (("mul_negative_weights", 0.4),),
    # (("mul_negative_weights", 0.5),),
    # (("mul_negative_weights", 0.6),),
    # (("mul_negative_weights", 0.7),),
    # (("mul_negative_weights", 0.8),),
    # (("mul_negative_weights", 0.9),),
    # ("sqrt_negative_weights"),
    # ("ln_negative_weights"),
    # Optimization-based weight debiasing (see _ml_apply_weight_adjustment_debias in experiment_utils.py)
    (("opt_debias", "optuna", "CMA-ES"),),
    # (("opt_debias", "pybobyqa", "Multi-Start BOBYQA"),),
    # (("opt_debias", "nevergrad", "BayesOpt"),),
)

subdominance_perf_metrics_list = ("Acc",)
subdominance_fair_metrics_list = ("AccPar", "DemPar", "EqOpp", "TNRPar")
# subdominance_fair_metrics_list = (
#     "AccPar",
#     "DemPar",
#     "EqOpp",
#     "FPRPar",
#     "EqOdds",
#     "TNRPar",
#     "FNRPar",
#     "PR_Z0",
#     "PR_Z1",
#     "NR_Z0",
#     "NR_Z1",
#     "TPR_Z0",
#     "TPR_Z1",
#     "TNR_Z0",
#     "TNR_Z1",
#     "FPR_Z0",
#     "FPR_Z1",
#     "FNR_Z0",
#     "FNR_Z1",
#     "PredPar",
#     "NegPredPar",
#     )

# Experiment to test how experts with different fairness/accuracy tradeoffs affect the learned fairness weights and feature expectations.
extra_bias_types_list = ((),)
extra_weight_adjusts_list = (
    (),
    (("ml_debias",),),
)
expert_algos_extra = [
    "DemParRed_0.01",
    "DemParRed_0.02",
    "DemParRed_0.03",
    "DemParRed_0.04",
    "DemParRed_0.05",
    "DemParRed_0.06",
    "DemParRed_0.07",
    "DemParRed_0.08",
    "DemParRed_0.09",
    "DemParRed_0.1",
]
datasets_extra = ["Adult"]

# Experiment to test how different experts affect the learned weights and feature expectations on one dataset. Prove that experts which optimize for one thing increase the weight on that thing, and decrease other weights.
extra2_bias_types_list = ((),)
extra2_weight_adjusts_list = ((),)
expert_algos_extra2 = [
    "OptAcc",
    "HardtDemPar",
    "HardtEqOpp",
    "HardtTNRPar",
    "HardtEqOdds",
    "DemParRed",
    "EqOppRed",
    "EqOddsRed",
]
datasets_extra2 = ["Adult"]

# ONLY USED FOR TESTING, NOT INCLUDED IN PAPER RESULTS
extra3_bias_types_list = (
    (),
    # ("threshold_swapping",), # This doesn't work for non postprocessing classifiers,
    # and also generates different kinds of bias for each type of Postprocessing classifier,
    # so don't use it for the final paper results.
    # (("unbalanced_redlining", 0.2),),
    # (("balanced_redlining", 0.2),),
    # (("perfectly_balanced_redlining", 0.1),),
    # (("perfectly_balanced_redlining", 0.2),),
    # (("perfectly_balanced_redlining", 0.3),),
    # (("perfectly_balanced_redlining", 0.4),),
    # (("perfectly_balanced_redlining", 0.5),),
    # (("perfectly_balanced_redlining", 0.6),),
    # (("perfectly_balanced_redlining", 0.7),),
    # (("perfectly_balanced_redlining", 0.8),),
    # (("perfectly_balanced_redlining", 0.9),),
    # (("corruption_bias", "CatBoost", 0.001, "gaussian", 1.0),),
)
extra3_weight_adjusts_list = (
    (),  # need initial empty list for this one, otherwise weight adjustment code won't work correctly
    (("mul_negative_weights", 0.0),),
    # (("mul_negative_weights", 0.1),),
    # (("mul_negative_weights", 0.2),),
    # (("mul_negative_weights", 0.3),),
    # (("mul_negative_weights", 0.4),),
    # (("mul_negative_weights", 0.5),),
    # (("mul_negative_weights", 0.6),),
    # (("mul_negative_weights", 0.7),),
    # (("mul_negative_weights", 0.8),),
    # (("mul_negative_weights", 0.9),),
    # ("sqrt_negative_weights"),
    # ("ln_negative_weights"),
    (("opt_debias", "optuna", "CMA-ES"),),
    # (("opt_debias", "pybobyqa", "Multi-Start BOBYQA"),),
    # (("opt_debias", "nevergrad", "BayesOpt"),),
)
expert_algos_extra3 = ["OptAcc"]
datasets_extra3 = [
    "COMPAS",
    # "Boston",
    "Adult",
    "ACSIncome__MA",
    "ACSIncome__MS",
    "ACSIncome__CA",
    "ACSIncome__IL",
    "ACSIncome__AL",
    "ACSIncome__HI",
]
# datasets_extra3 = ["Adult"]

# Run experiments
extensions = [".csv", ".csv", ".json"]
folder_paths = [
    "./data/experiment_output/fair_irl/exp_conv_details/",
    "./data/experiment_output/fair_irl/exp_results/",
    "./data/experiment_output/fair_irl/exp_info/",
]
for extension, folder_path in zip(extensions, folder_paths):
    folder = Path(folder_path)
    for file_path in folder.glob(f"*{extension}"):
        if file_path.is_file():
            file_path.unlink()  # Delete the file
# Create config for each experiment
for exp_dict in exp_list:
    base_exp_info = exp_dict["base_exp_info"]
    source_states = exp_dict["source_states"]
    exp_info = dict(base_exp_info)
    experiments = []
    # for expert_algo in expert_algos_extra:
    #     for bias_types in extra_bias_types_list:
    #         for source_dataset in source_states:
    #             if source_dataset in datasets_extra:
    #                 experiments.append(
    #                     {
    #                         "EXPERT_ALGO": expert_algo,
    #                         "BIAS_TYPES": bias_types,
    #                         "IRL_METHOD": "FairIRL",
    #                         "DATASET": source_dataset,
    #                         "WEIGHT_ADJUSTS_LIST": extra_weight_adjusts_list,
    #                         "SUBDOMINANCE_PERF_METRICS_LIST": subdominance_perf_metrics_list,
    #                         "SUBDOMINANCE_FAIR_METRICS_LIST": subdominance_fair_metrics_list,
    #                     }
    #                 )
    # for expert_algo in expert_algos_extra2:
    #     for bias_types in extra2_bias_types_list:
    #         for source_dataset in source_states:
    #             if source_dataset in datasets_extra2:
    #                 experiments.append(
    #                     {
    #                         "EXPERT_ALGO": expert_algo,
    #                         "BIAS_TYPES": bias_types,
    #                         "IRL_METHOD": "FairIRL",
    #                         "DATASET": source_dataset,
    #                         "WEIGHT_ADJUSTS_LIST": extra2_weight_adjusts_list,
    #                         "SUBDOMINANCE_PERF_METRICS_LIST": subdominance_perf_metrics_list,
    #                         "SUBDOMINANCE_FAIR_METRICS_LIST": subdominance_fair_metrics_list,
    #                     }
    #                 )
    for expert_algo in expert_algos_extra3:
        for bias_types in extra3_bias_types_list:
            for source_dataset in source_states:
                if source_dataset in datasets_extra3:
                    experiments.append(
                        {
                            "EXPERT_ALGO": expert_algo,
                            "BIAS_TYPES": bias_types,
                            "IRL_METHOD": "FairIRL",
                            "DATASET": source_dataset,
                            "WEIGHT_ADJUSTS_LIST": extra3_weight_adjusts_list,
                            "SUBDOMINANCE_PERF_METRICS_LIST": subdominance_perf_metrics_list,
                            "SUBDOMINANCE_FAIR_METRICS_LIST": subdominance_fair_metrics_list,
                        }
                    )
    # for expert_algo in expert_algos:
    #     for bias_types in bias_types_list:
    #         for source_dataset in source_states:
    #             experiments.append(
    #                 {
    #                     "EXPERT_ALGO": expert_algo,
    #                     "BIAS_TYPES": bias_types,
    #                     "IRL_METHOD": "FairIRL",
    #                     "DATASET": source_dataset,
    #                     "WEIGHT_ADJUSTS_LIST": weight_adjusts_list,
    #                     "SUBDOMINANCE_PERF_METRICS_LIST": subdominance_perf_metrics_list,
    #                     "SUBDOMINANCE_FAIR_METRICS_LIST": subdominance_fair_metrics_list,
    #                 }
    #             )
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
            source_X[f] = source_X[f].astype(str)

        source_X_cols = (
            source_feature_types["boolean"]
            + source_feature_types["categoric"]
            + source_feature_types["continuous"]
        )

        if exp_info["USE_HIDDEN_FEATURES_SOURCE"]:
            source_X_cols += source_feature_types["hidden"]
        _source_X = source_X[source_X_cols]

        logging.info(
            f"For dataset: {experiment['DATASET']} and expert algo: {experiment['EXPERT_ALGO']}:"
        )

        # with cProfile.Profile() as pr:
        run_bias_experiment(
            exp_info,
            source_X=_source_X,
            source_y=source_y,
            source_feature_types=source_feature_types,
        )
    #     stats = pstats.Stats(pr)
    #     stats.sort_stats("cumulative").print_stats(100)
    #     stats.sort_stats("time").print_stats(100)
    #     break
    # break

logging.info("TRAINING FINISHED SUCESSFULLY!")


def play_notification():
    subprocess.run(
        ["powershell.exe", "-Command", "[System.Media.SystemSounds]::Hand.Play()"]
    )


play_notification()
