import cProfile
import logging
import pstats
import random
import subprocess
import warnings

import numpy as np
import pandas as pd
from IPython.display import HTML, display

from fair_irl.datasets import *
from fair_irl.experiment_utils import *
from fair_irl.irl.fair_irl import *
from fair_irl.utils import *


def play_notification():
    subprocess.run(
        ["powershell.exe", "-Command", "[System.Media.SystemSounds]::Hand.Play()"]
    )


def main():
    logging.basicConfig(level=logging.INFO)
    warnings.filterwarnings("ignore")

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
        # Policy learning parameters
        "IRL_METHOD": None,
        "METHOD": "highs",
        # The most iterations the `opt_debias` weight adjustment runs before
        # giving up on finding a better weight set. This is only a safety net
        # -- the loop is expected to stop on its own as soon as an iteration
        # fails to improve.
        # Setting this to 1 for now because the Iteration loop does not appear
        # to improve performance versus the initial weight set.
        "OPT_DEBIAS_MAX_ITERATIONS": 1,
        # Plotting parameters
        "NOISE_FACTOR": 0.01,
        "ANNOTATE": True,
        "N_TRIALS": 1,  # TODO: CHANGE THIS BACK TO 3 FOR FINAL PAPER RESULTS
        "N_SUBDOMINANCE_GROUPS": 100,
        "DOT_WEIGHTS_FEAT_EXP": True,
        "N_DATASET_SAMPLES": None,
        "RANDOM_SEED": random_seed,
        # Which techniques to train and evaluate. Valid entries are
        # "FairIRL Bias Reduction" and "Superhuman Fairness"; list either or
        # both. Each listed technique gets its own W&B run per dataset bias
        # type, produced by the same evaluation code on the same data split, so
        # that their metrics are directly comparable.
        "ALGORITHMS": [
            "FairIRL Bias Reduction",
            "Superhuman Fairness",
        ],
        ##
        # Superhuman Fairness baseline parameters. Ignored unless
        # "Superhuman Fairness" is listed in ALGORITHMS above.
        ##
        # Where the reference decisions ("demonstrations") the baseline
        # imitates come from:
        #   "expert_demos" -- this project's own expert (EXPERT_ALGO), on the
        #       same demonstration groups the subdominance metric scores every
        #       policy against, so that both techniques imitate the same expert
        #       on the same rows.
        #   "pp_baseline"  -- the original paper's own demonstrator: a logistic
        #       regression post-processed by a fairlearn ThresholdOptimizer.
        "SH_DEMO_SOURCE": "pp_baseline",
        # The performance/fairness measures the baseline optimizes -- the `-f`
        # flag of the original implementation. Entries may be this project's
        # objective names (e.g. "Acc", "DemPar", "TNRPar") or the original
        # paper's metric names ("inacc", "dp", "eqodds", "prp", "eqopp", "fnr",
        # "fpr", "ppv", "npv", "error_rate_diff"). None uses this experiment's
        # SUBDOMINANCE_PERF_METRICS_LIST + SUBDOMINANCE_FAIR_METRICS_LIST, so
        # that the baseline optimizes exactly what both techniques are
        # evaluated on. The paper's own configuration is
        # ["inacc", "dp", "eqodds", "prp"].
        "SH_FEATURES": ["inacc", "dp", "eqodds", "prp"],
        # "SH_FEATURES": None,
        # How many demonstrations to imitate. None uses every subdominance
        # group ("expert_demos") or the paper's 50 ("pp_baseline").
        "SH_NUM_DEMOS": None,
        # The original's `iters`, `lr_theta` and `lamda`.
        "SH_ITERS": 30,
        "SH_LR_THETA": 0.01,
        "SH_LAMDA": 0.001,
        # The fairness constraint the "pp_baseline" demonstrator satisfies.
        # Unused by the "expert_demos" source.
        "SH_DEMO_CONSTRAINTS": "demographic_parity",
    }

    # ### COMPAS
    base_exp_info = {
        "EXPERIMENT_NAME": "COMPAS",
        "MIN_FREQ_FILL_PCT": 0.08,  # MIN_FREQ_FILL_PCT values have been chosen
        # so that the runtime per learned policy is roughly equal for all
        # datasets
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
        "OptClfMDPPol"
        # "OptAcc",
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

    #  all available bias types = ("unbalanced_redlining", "balanced_redlining", "perfectly_balanced_redlining", "corruption_bias")
    # The unbiased run is always performed first, ahead of every bias type
    # listed here.
    dataset_bias_type_list = (
        # ("unbalanced_redlining", 0.2),
        # ("balanced_redlining", 0.2),
        # ("perfectly_balanced_redlining", 0.2),
        # ("corruption_bias", "CatBoost", 0.0001, "gaussian", 1.0),
        ("corruption_bias", "CatBoost", 0.001, "gaussian", 1.0),
        # ("corruption_bias", "CatBoost", 0.01, "gaussian", 1.0),
    )
    # dataset_bias_types_list = (("perfectly_balanced_redlining"))

    weight_adjust_list = (
        # ("mul_negative_weights", 0.0),
        # ("mul_negative_weights", 0.1),
        # ("mul_negative_weights", 0.2),
        # ("mul_negative_weights", 0.3),
        # ("mul_negative_weights", 0.4),
        # ("mul_negative_weights", 0.5),
        # ("mul_negative_weights", 0.6),
        # ("mul_negative_weights", 0.7),
        # ("mul_negative_weights", 0.8),
        # ("mul_negative_weights", 0.9),
        # Optimization-based weight debiasing (see _ml_apply_weight_adjustment_debias in experiment_utils.py)
        # ("opt_debias", "optuna", "CMA-ES", 500),
        # ("opt_debias", "pybobyqa", "Multi-Start BOBYQA", 500),
        ("opt_debias", "nevergrad", "BayesOpt", 200),
        # ("opt_debias", "nevergrad", "Nelder-Mead", 500),
        # ("opt_debias", "nevergrad", "Powell", 500),
    )

    subdominance_perf_metrics_list = ("Acc",)
    subdominance_fair_metrics_list = ("AccPar", "DemPar", "EqOpp", "TNRPar")

    selected_datasets = [
        "COMPAS",
        "Adult",
        "ACSIncome__MA",
        "ACSIncome__MS",
        "ACSIncome__CA",
        "ACSIncome__IL",
        "ACSIncome__AL",
        "ACSIncome__HI",
    ]

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
    for exp_dict in exp_list:
        source_states = exp_dict["source_states"]
        if source_states[0] not in selected_datasets:
            logging.info(
                f"Skipping experiment {exp_dict['base_exp_info']['EXPERIMENT_NAME']} since it is not in the selected datasets list."
            )
            continue
        base_exp_info = exp_dict["base_exp_info"]
        exp_info = dict(base_exp_info)
        experiments = []
        for expert_algo in expert_algos:
            for source_dataset in source_states:
                experiments.append(
                    {
                        "EXPERT_ALGO": expert_algo,
                        "DATASET_BIAS_TYPE_LIST": dataset_bias_type_list,
                        "IRL_METHOD": "FairIRL",
                        "DATASET": source_dataset,
                        "WEIGHT_ADJUST_LIST": weight_adjust_list,
                        "SUBDOMINANCE_PERF_METRICS_LIST": subdominance_perf_metrics_list,
                        "SUBDOMINANCE_FAIR_METRICS_LIST": subdominance_fair_metrics_list,
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
                session_id=session_id,
            )
        #     stats = pstats.Stats(pr)
        #     stats.sort_stats("cumulative").print_stats(100)
        #     stats.sort_stats("time").print_stats(100)
        #     break
        # break

    logging.info(f"TRAINING FINISHED SUCESSFULLY! W&B session: {session_id}")

    # play_notification()


if __name__ == "__main__":
    main()
