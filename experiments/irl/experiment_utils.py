import datetime
import itertools
import json
import logging
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.base
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import (
    ExponentiatedGradient,
    EqualizedOdds,
    ErrorRateParity,
    BoundedGroupLoss,
    ZeroOneLoss,
)
from matplotlib.ticker import FormatStrFormatter
from research.rl.env.clf_mdp import *
from research.rl.env.clf_mdp_policy import *
from research.rl.env.objectives import *
from research.irl.fair_irl import *
from research.ml.svm import SVM
from research.utils import *
from scipy.spatial.distance import cosine
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from .datasets import *


# Color palette for plotting
cp = sns.color_palette()


# Objective looup
OBJ_LOOKUP_BY_NAME = {
    "Acc": AccuracyObjective,
    "AccPar": AccuracyParityObjective,
    "DemPar": DemographicParityObjective,
    "EqOpp": EqualOpportunityObjective,
    "FPRPar": FalsePositiveRateParityObjective,
    "TNRPar": TrueNegativeRateParityObjective,
    "FNRPar": FalseNegativeRateParityObjective,
    "PredPar": PredictiveParityObjective,
    "NegPredPar": NegativePredictiveParityObjective,
    "PR_Z0": GroupPositiveRateZ0Objective,
    "PR_Z1": GroupPositiveRateZ1Objective,
    "NR_Z0": GroupNegativeRateZ0Objective,
    "NR_Z1": GroupNegativeRateZ1Objective,
    "TPR_Z0": GroupTruePositiveRateZ0Objective,
    "TPR_Z1": GroupTruePositiveRateZ1Objective,
    "TNR_Z0": GroupTrueNegativeRateZ0Objective,
    "TNR_Z1": GroupTrueNegativeRateZ1Objective,
    "FPR_Z0": GroupFalsePositiveRateZ0Objective,
    "FPR_Z1": GroupFalsePositiveRateZ1Objective,
    "FNR_Z0": GroupFalseNegativeRateZ0Objective,
    "FNR_Z1": GroupFalseNegativeRateZ1Objective,
}


class FairLearnSkLearnWrapper:
    """
    Wrapper around scikit-learn classifiers to make them compatible with
    fairlearn classifiers, which require an additional `sensitive_features`
    attribute to be passed in when calling `fit` and `predict`.

    Attributes
    ----------
    initial_clf : sklearn base estimator
        A clone of the initial `clf`. Used to create new clones later on
        without any of the attributes updated.
    has_access_to_sensitive_features : bool, default True
        If False, sensitive features are not available at prediction time. E.g.
        for fairlearn reductions (as opposed to threshold optimizers).
    clone_on_fit : bool, default False
        If True, resets the base classifier.
    """

    def __init__(
        self,
        clf,
        sensitive_features,
        has_access_to_sensitive_features=True,
        clone_on_fit=False,
    ):
        self.clf = clf
        self.initial_clf = sklearn.base.clone(self.clf)
        self.sensitive_features = sensitive_features
        self.has_access_to_sensitive_features = has_access_to_sensitive_features
        self.clone_on_fit = clone_on_fit

    def fit(self, X, y, sample_weight=None, **kwargs):
        if self.clone_on_fit:
            self.clf = sklearn.base.clone(self.initial_clf)

        self.clf.fit(
            X,
            y,
            sensitive_features=X[self.sensitive_features],
            **kwargs,
        )
        return self

    def predict(self, X, sample_weight=None, **kwargs):
        if self.has_access_to_sensitive_features:
            preds = self.clf.predict(
                X,
                sensitive_features=X[self.sensitive_features],
                **kwargs,
            )
        else:
            preds = self.clf.predict(
                X,
                **kwargs,
            )

        return preds


class UnfairNoisyClassifier:
    """
    Wrapper around scikit-learn classifiers to intentionally make them slightly
    less performant. This is useful when generating initial policies where
    having reasonably fair and accurate (but not optimal) classifiers helps set
    the weights in the right directions.

    It randomplly flips negative predictions to positive predictions, based on
    the probabilities defined in the input.
    """

    def __init__(self, clf, prob):
        self.clf = clf
        self.prob = prob

    def fit(self, X, y, **kwargs):
        self.clf.fit(X, y, **kwargs)
        return self

    def predict(self, X, **kwargs):
        preds = self.clf.predict(X, **kwargs)

        z0_override_indexes = np.argwhere(
            np.random.rand(len(X)) < self.prob[0]
        ).flatten()

        z1_override_indexes = np.argwhere(
            np.random.rand(len(X)) < self.prob[1]
        ).flatten()

        for idx in z0_override_indexes:
            z = X.iloc[idx]["z"]
            if z == 0:
                preds[idx] = 1 - preds[idx]

        for idx in z1_override_indexes:
            z = X.iloc[idx]["z"]
            if z == 1:
                preds[idx] = 1 - preds[idx]

        return preds


def generate_expert_algo_lookup(feature_types):
    """
    Parameters
    ----------
    feature_types : dict<str, array-like>
        Mapping of column names to their type of feature. Used to when
        constructing the sklearn pipeline.

    Returns
    -------
    expert_algo_lookup : dict<str, sklearn.pipeline>
        The expert algo lookup dictionary that maps the string name for an
        algorithm to the actual implementation.
    """
    # OptAcc
    opt_acc_pipe = sklearn_clf_pipeline(
        feature_types,
        # clf_inst=DecisionTreeClassifier(min_samples_leaf=10, max_depth=4),
        # clf_inst=DecisionTreeClassifier(min_samples_leaf=5, max_depth=10),
        # RandomForestClassifier(),
        XGBClassifier(),
    )

    # HardtDemPar
    dem_par_thresh_opt = ThresholdOptimizer(
        constraints="demographic_parity",
        predict_method="predict",
        prefit=False,
        estimator=sklearn_clf_pipeline(
            feature_types=feature_types,
            # clf_inst=DecisionTreeClassifier(min_samples_leaf=10, max_depth=4),
            # clf_inst=RandomForestClassifier(), # Messes up DemPar. Why?
            clf_inst=XGBClassifier(),
        ),
    )
    dem_par_wrapper = FairLearnSkLearnWrapper(
        clf=dem_par_thresh_opt,
        sensitive_features="z",
    )

    # HardtEqOpp
    eq_opp_thresh_opt = ThresholdOptimizer(
        constraints="true_positive_rate_parity",
        predict_method="predict",
        prefit=False,
        estimator=sklearn_clf_pipeline(
            feature_types=feature_types,
            # clf_inst=DecisionTreeClassifier(min_samples_leaf=10, max_depth=4),
            # clf_inst=RandomForestClassifier(),
            clf_inst=XGBClassifier(),
        ),
    )
    eq_opp_wrapper = FairLearnSkLearnWrapper(
        clf=eq_opp_thresh_opt,
        sensitive_features="z",
    )

    # HardtEqOdds
    eq_odds_thresh_opt = ThresholdOptimizer(
        constraints="equalized_odds",
        predict_method="predict",
        prefit=False,
        estimator=sklearn_clf_pipeline(
            feature_types=feature_types,
            # clf_inst=RandomForestClassifier(),
            clf_inst=XGBClassifier(),
            # clf_inst=DecisionTreeClassifier(min_samples_leaf=10000, max_depth=20),
        ),
    )
    eq_odds_thresh_wrapper = FairLearnSkLearnWrapper(
        clf=eq_opp_thresh_opt,
        sensitive_features="z",
    )

    # Equal Odds Reduction
    eq_odds = EqualizedOdds(difference_bound=0.01)
    eq_odds_exp_grad = ExponentiatedGradient(
        sample_weight_name="classifier__sample_weight",
        constraints=eq_odds,
        estimator=sklearn_clf_pipeline(
            feature_types=feature_types,
            # clf_inst=DecisionTreeClassifier(min_samples_leaf=10, max_depth=4),
            # clf_inst=RandomForestClassifier(),
            clf_inst=XGBClassifier(),
        ),
    )
    eq_odds_red_wrapper = FairLearnSkLearnWrapper(
        clf=eq_odds_exp_grad,
        sensitive_features="z",
        has_access_to_sensitive_features=False,
        clone_on_fit=True,
    )

    # Bounded Group Loss Reduction
    bgl = BoundedGroupLoss(ZeroOneLoss(), upper_bound=0.03)
    bgl_exp_grad = ExponentiatedGradient(
        sample_weight_name="classifier__sample_weight",
        constraints=bgl,
        estimator=sklearn_clf_pipeline(
            feature_types=feature_types,
            # clf_inst=DecisionTreeClassifier(min_samples_leaf=5, max_depth=10),
            # clf_inst=RandomForestClassifier(),
            clf_inst=XGBClassifier(),
        ),
    )
    bgl_wrapper = FairLearnSkLearnWrapper(
        clf=bgl_exp_grad,
        sensitive_features="z",
        has_access_to_sensitive_features=False,
        clone_on_fit=True,
    )

    # False Positive Rate Noisy
    fpr_thresh_opt = ThresholdOptimizer(
        constraints="false_positive_rate_parity",
        predict_method="predict",
        prefit=False,
        estimator=sklearn_clf_pipeline(
            feature_types=feature_types,
            # clf_inst=DecisionTreeClassifier(min_samples_leaf=10, max_depth=4),
            # clf_inst=RandomForestClassifier(),
            clf_inst=XGBClassifier(),
        ),
    )
    fpr_wrapper = FairLearnSkLearnWrapper(
        clf=fpr_thresh_opt,
        sensitive_features="z",
    )

    # True Negative Rate Noisy
    tnr_thresh_opt = ThresholdOptimizer(
        constraints="true_negative_rate_parity",
        predict_method="predict",
        prefit=False,
        estimator=sklearn_clf_pipeline(
            feature_types=feature_types,
            # clf_inst=DecisionTreeClassifier(min_samples_leaf=10, max_depth=4),
            # clf_inst=RandomForestClassifier(),
            clf_inst=XGBClassifier(),
        ),
    )
    tnr_wrapper = FairLearnSkLearnWrapper(
        clf=tnr_thresh_opt,
        sensitive_features="z",
    )

    # False Positive Rate Noisy
    fnr_thresh_opt = ThresholdOptimizer(
        constraints="false_negative_rate_parity",
        predict_method="predict",
        prefit=False,
        estimator=sklearn_clf_pipeline(
            feature_types=feature_types,
            # clf_inst=DecisionTreeClassifier(min_samples_leaf=10, max_depth=4),
            # clf_inst=RandomForestClassifier(),
            clf_inst=XGBClassifier(),
        ),
    )
    fnr_wrapper = FairLearnSkLearnWrapper(
        clf=fnr_thresh_opt,
        sensitive_features="z",
    )

    dummy_pipe = DummyClassifier(strategy="uniform")

    compas_score_high = ManualClassifier(
        # lambda row: int(row['score_text'] == 'High')
        lambda row: int(row["decile_score"] >= 6)
    )

    expert_algo_lookup = {
        # Experts
        "OptAcc": opt_acc_pipe,
        "HardtDemPar": dem_par_wrapper,
        "HardtEqOpp": eq_opp_wrapper,
        "HardtEqOdds": eq_odds_thresh_wrapper,
        "HardtFPRPar": fpr_wrapper,
        "HardtTNRPar": tnr_wrapper,
        "HardtFNRPar": fnr_wrapper,
        "Dummy": dummy_pipe,
        "RedEqOdds": eq_odds_red_wrapper,
        "BoundedGroupLoss": bgl_wrapper,
        "COMPAS": compas_score_high,
        # Initial policies
        "OptAccNoisy": UnfairNoisyClassifier(clf=opt_acc_pipe, prob=[0.15, 0.25]),
        "HardtDemParNoisy": UnfairNoisyClassifier(
            clf=dem_par_wrapper, prob=[0.05, 0.25]
        ),
        "HardtEqOppNoisy": UnfairNoisyClassifier(clf=eq_opp_wrapper, prob=[0.05, 0.25]),
        "HardtFPRNoisy": UnfairNoisyClassifier(clf=fpr_wrapper, prob=[0.05, 0.05]),
        "HardtTNRNoisy": UnfairNoisyClassifier(clf=tnr_wrapper, prob=[0.05, 0.05]),
        "DummyNoisy": UnfairNoisyClassifier(clf=dummy_pipe, prob=[0.05, 0.05]),
    }

    return expert_algo_lookup


# def generate_all_exp_results_df(
#     feat_obj_set,
#     perf_obj_set,
#     n_trials,
#     data_demo,
#     exp_algo,
#     irl_method,
# ):
#     """
#     Generate dataframe for experiment parameters and results

#     Parameters
#     ----------
#     feat_obj_set : ObjectiveSet
#         The feature expectation objective set.
#     perf_obj_set : ObjectiveSet
#         The performance measure objective set.
#     n_trials : int
#         Number of experiment trials to run.
#     data_demo : str
#         The name of the dataset used to generate expert demonstrations.
#     exp_algo : str
#         The name of the algorithm used to train the expert demonstrator.
#     irl_method : str
#         The name of the IRL algorithm used to recover the rewards.

#     Returns
#     -------
#     all_exp_df : pandas.DataFrame
#         A dataframe with relevant columns, but no data.
#     """
#     all_exp_df_cols = ["n_trials", "data_demo", "exp_algo", "irl_method"]

#     # Feature expectation objectives
#     for obj in feat_obj_set.objectives:
#         all_exp_df_cols.append(f"muE_{obj.name}_mean")
#         all_exp_df_cols.append(f"muE_{obj.name}_std")

#     for obj in feat_obj_set.objectives:
#         all_exp_df_cols.append(f"wL_{obj.name}_mean")
#         all_exp_df_cols.append(f"wL_{obj.name}_std")

#     for obj in feat_obj_set.objectives:
#         all_exp_df_cols.append(f"muL_err_{obj.name}_mean")
#         all_exp_df_cols.append(f"muL_err_{obj.name}_std")

#     # Performance measure objectives
#     for obj in perf_obj_set.objectives:
#         all_exp_df_cols.append(f"muE_{obj.name}_mean")
#         all_exp_df_cols.append(f"muE_{obj.name}_std")

#     for obj in perf_obj_set.objectives:
#         all_exp_df_cols.append(f"wL_{obj.name}_mean")
#         all_exp_df_cols.append(f"wL_{obj.name}_std")

#     for obj in perf_obj_set.objectives:
#         all_exp_df_cols.append(f"muL_err_{obj.name}_mean")
#         all_exp_df_cols.append(f"muL_err_{obj.name}_std")

#     all_exp_df_cols.append("muL_err_l2norm_mean")
#     all_exp_df_cols.append("muL_err_l2norm_std")

#     all_exp_df = pd.DataFrame(columns=all_exp_df_cols)

#     all_exp_df.loc[0, "n_trials"] = n_trials
#     all_exp_df.loc[0, "data_demo"] = "Adult"
#     all_exp_df.loc[0, "exp_algo"] = exp_algo
#     all_exp_df.loc[0, "irl_method"] = "IRL_METHOD"

#     return all_exp_df


def generate_single_exp_results_df(feat_obj_set, perf_obj_set, results):
    """
    Generate dataframe for a single experiment. Keeps track of the results of
    the best learned policy.

    Parameters
    ----------
    feat_obj_set : ObjectiveSet
        The feature expectation objective set.
    perf_obj_set : ObjectiveSet
        The performance measure objective set.
    data : list<list>
        The results.
    results : pandas.DataFrame
        A dataframe with relevant weight, feat exp, and error columns for the
        best learned policy. Each row is produced by the `new_trial_result()`
        method.
    """
    exp_df_cols = []

    for obj in feat_obj_set.objectives:
        exp_df_cols.append(f"muE_unbiased_{obj.name}_mean")
        exp_df_cols.append(f"muE_unbiased_{obj.name}_std")

    for obj in feat_obj_set.objectives:
        exp_df_cols.append(f"muE_{obj.name}_mean")
        exp_df_cols.append(f"muE_{obj.name}_std")

    for obj in feat_obj_set.objectives:
        exp_df_cols.append(f"muE_hold_{obj.name}_mean")
        exp_df_cols.append(f"muE_hold_{obj.name}_std")

    for obj in feat_obj_set.objectives:
        exp_df_cols.append(f"wL_{obj.name}")

    for obj in feat_obj_set.objectives:
        exp_df_cols.append(f"muL_{obj.name}")
        exp_df_cols.append(f"muL_best_{obj.name}")
        exp_df_cols.append(f"muL_hold_{obj.name}")
        exp_df_cols.append(f"muL_best_hold_{obj.name}")

    for obj in feat_obj_set.objectives:
        exp_df_cols.append(f"muL_err_{obj.name}")
        exp_df_cols.append(f"muL_hold_err_{obj.name}")

    exp_df_cols.append("max_abs_subdominance")
    exp_df_cols.append("sum_abs_subdominance")
    exp_df_cols.append("max_rel_subdominance")
    exp_df_cols.append("sum_rel_subdominance")

    exp_df_cols.append("muL_err_l2norm")
    exp_df_cols.append("muL_hold_err_l2norm")

    for obj in perf_obj_set.objectives:
        exp_df_cols.append(f"muE_perf_hold_{obj.name}")
        exp_df_cols.append(f"muL_perf_best_hold_{obj.name}")

    exp_df_cols.append(f"source_trial_runtime")
    exp_df_cols.append(f"source_irl_loop_runtime")
    exp_df_cols.append(f"source_trial_inputsize")

    exp_df = pd.DataFrame(results, columns=exp_df_cols)

    return exp_df


def new_trial_result(
    best_idx,
    feat_obj_set,
    perf_obj_set,
    muE_unbiased,
    muE,
    muE_hold,
    muE_perf_hold,
    df_irl,
    source_trial_runtime,
    source_irl_loop_runtime,
    source_trial_inputsize,
):
    """
    Generates a row of "results", which are collected and persisted for each
    experiment.

    Parameters
    ---------
    feat_obj_set : research.irl.fair_irl.ObjectiveSet
        The set of objectives for the feature expectations.
    perf_obj_set : research.irl.fair_irl.ObjectiveSet
        The set of objectives for the performance measures.
    muE : array-like<float>, shape(n_expert_demos, n_objectives)
        The feature expectations of the expert demonstrations, specifically on
        the demo set.
    muE_hold : array-like<float>, shape(n_expert_demos, n_objectives)
        The feature expectations of the expert demonstrations on the hold out
        set.
    muE_perf_hold : pandas.DataFrame
        Performance measures of expert holdout demos.
    df_irl : pandas.DataFrame
        A collection of results where each row represents either an expert demo
        (and therefore a positive SVM training example) or a learned policy
        (and therefore a negative SVM training example). Includes relevant
        items like learned weights, feature expectations, error, etc.
    source_trial_runtime : float
        The runtime to complete the source trial, in seconds.
    source_irl_loop_runtime : float
        The runtime to complete the source IRL loop, in seconds.
    source_trial_inputsize : int
        The size of the input space. Specifically `mdp.n_states_` (includes y)
    muE_target : array-like<float>. Shape(n_expert_demos, n_objectives).
        The feature expectations of running the expert algo on the target
        domain demo set.
    muE_perf_target : array-like<float>. Shape(n_expert_demos, n_objectives).
        The performance measures of running the expert algo on the target
        domain demo set.
    muL_target_hold
        The feature expectations of running the learned algo on the target
        domain holdout set.
    muL_perf_target_hold
        The performance measures of running the expert algo on the target
        domain holdout set.

    Returns
    -------
    result : list<list<numeric>>
        The new result row.
    """
    result = []

    # Do this for feature expectation obj set
    # Adds n_feat_exp * 2 values
    for i, obj in enumerate(feat_obj_set.objectives):
        muE_unbiased_mean = np.mean(muE_unbiased[:, i])
        muE_unbiased_std = np.std(muE_unbiased[:, i])
        result += [muE_unbiased_mean, muE_unbiased_std]

    # Adds n_feat_exp * 2 values
    for i, obj in enumerate(feat_obj_set.objectives):
        muE_mean = np.mean(muE[:, i])
        muE_std = np.std(muE[:, i])
        result += [muE_mean, muE_std]

    # Adds n_feat_exp * 2 values
    for i, obj in enumerate(feat_obj_set.objectives):
        muE_hold_mean = np.mean(muE_hold[:, i])
        muE_hold_std = np.std(muE_hold[:, i])
        result += [muE_hold_mean, muE_hold_std]

    # best_t = (
    #     df_irl.query("(is_expert == 0) and (is_init_policy == 0)")
    #     .sort_values("t")["t"]
    #     .values[0]
    # )
    # try:
    #     best_idx = (
    #         df_irl[(df_irl["is_expert"] == 0) & (df_irl["is_init_policy"] == 0)]
    #         .query("abs(t - @best_t) <= .0001")
    #         .sort_index()
    #         .index[0]
    #     )
    # except:
    #     pass
    best_row = df_irl.loc[best_idx]

    # Adds n_feat_exp * 1 values
    for i, obj in enumerate(feat_obj_set.objectives):
        result.append(best_row[f"{obj.name}_weight"])

    # Adds n_feat_exp * 4 values
    for obj in feat_obj_set.objectives:
        result.append(best_row[f"{obj.name}"])
        result.append(best_row[f"muL_best_{obj.name}"])
        result.append(best_row[f"muL_hold_{obj.name}"])
        result.append(best_row[f"muL_best_hold_{obj.name}"])

    # Adds n_feat_exp * 2 values
    for i, obj in enumerate(feat_obj_set.objectives):
        _muL_err = abs(best_row[f"{obj.name}"] - np.mean(muE_hold[:, i]))
        _muL_hold_err = abs(best_row[f"muL_hold_{obj.name}"] - np.mean(muE_hold[:, i]))
        result.append(_muL_err)
        result.append(_muL_hold_err)

    # Adds subdominance metrics * 4 values
    result.append(best_row["max_abs_subdominance"])
    result.append(best_row["sum_abs_subdominance"])
    result.append(best_row["max_rel_subdominance"])
    result.append(best_row["sum_rel_subdominance"])

    # Adds n_feat_exp * 2 values
    result.append(best_row["mu_delta_l2norm"])
    result.append(best_row["mu_delta_l2norm_hold"])

    # Append performance measures of expert_holdout and best learned policies
    # Adds n_perf_meas * 2 values
    for i, obj in enumerate(perf_obj_set.objectives):
        perf_hold_mean = np.mean(muE_perf_hold[:, i])
        result.append(perf_hold_mean)
        result.append(best_row[f"muL_perf_best_hold_{obj.name}"])

    result.append(source_trial_runtime)
    result.append(source_irl_loop_runtime)
    result.append(source_trial_inputsize)

    return result


def compute_alphas(raw_demos_feat_loss, clf_demos_feat_loss):
    # Orignal code taken from https://github.com/omidMemari/superhumn-fairness/blob/main/main.py#L672
    lamda = 0.001
    alphas = np.ones(raw_demos_feat_loss.shape[1])

    for k in range(raw_demos_feat_loss.shape[1]):  # for each feature
        sorted_demos = []
        for j in range(raw_demos_feat_loss.shape[0]):  # for each demo
            sample_loss = clf_demos_feat_loss[j][
                k
            ]  # Contains the various feature expectations for each classifier demo
            demo_loss = raw_demos_feat_loss[j][
                k
            ]  # Contains the various feature expectations for each human demo
            sorted_demos.append((demo_loss, sample_loss))

        sorted_demos.sort(
            key=lambda x: x[0]
        )  # dominated_demos.sort(key = lambda x: x[0], reverse=True)   # sort based on demo loss
        # print(self.feature[k])
        # print("demo_loss, sample_loss: ")
        # print(sorted_demos)
        sorted_demos = np.array(sorted_demos)
        alphas[k] = (
            100  # max(self.alpha) #np.mean(self.alpha) # default value in case it didn't change using previous alpha values
        )
        # print("alpha {}", k)
        # print(alpha)
        for m, demo in enumerate(sorted_demos):
            if demo[0] > demo[1]:
                alphas[k] = min(
                    100, 1.0 / (demo[0] - demo[1])
                )  ### limit max alpha to 100
            if (demo[1] + lamda) <= np.mean(
                [x[0] for x in sorted_demos[0 : m + 1]]
            ):  # if (demo[2]) <= np.mean([x[1] for x in dominated_demos[0:m+1]] and demo[0] > 0):
                break
    # print("alpha : ")
    # print(alpha)
    # model_params = {"eval": self.eval}
    # find_gamma_superhuman(self.demo_list, self.model_params)
    return alphas


def compute_subdominance(
    alphas, raw_demos_feat_loss, clf_demos_feat_loss, relative=True, sum_agg=True
):
    # Computation taken from https://proceedings.mlr.press/v162/ziebart22a/ziebart22a.pdf eq. 5-8 and def. 5
    lamda = 0.001  # I assume this is the correct lambda value to use, but it is not specified in the paper.
    beta = 1.0
    # compute the average across subdom for each demo
    total_subdom_list = []
    for demo in range(raw_demos_feat_loss.shape[0]):
        subdom_list = []
        for feature in range(raw_demos_feat_loss.shape[1]):
            alpha = alphas[feature]
            raw_demo_loss = raw_demos_feat_loss[demo][feature]
            clf_demo_loss = clf_demos_feat_loss[demo][feature]
            if relative:
                subdom = (alpha * ((clf_demo_loss / raw_demo_loss) - 1)) + beta
            else:
                subdom = (alpha * (clf_demo_loss - raw_demo_loss)) + beta
            subdom_list.append(subdom)
        if sum_agg:
            total_subdom = sum(subdom_list)
        else:
            total_subdom = max(subdom_list)
        total_subdom += (lamda / 2.0) * (np.linalg.norm(alphas) ** 2)
        total_subdom_list.append(total_subdom)
    final_subdom = sum(total_subdom_list) / len(total_subdom_list)
    return final_subdom


def run_trial_source_domain(
    exp_info,
    X=None,
    y=None,
    feature_types=None,
    plot_svm_iters=False,
):
    """
    Runs 1 trial to learn weights in the source domain.

    X, y, feature_types don't need to be passed. If they are, then
    `generate_dataset()` is not invoked.

    Parameters
    ----------
    exp_info : dict
        Metadata about the experiment.
    X : pandas.DataFrame, Optional
        The X (including z) columns.
    y : pandas.Series, Optional
        Just the y column.
    feature_types : dict<str, array-like>, Optional
        Mapping of column names to their type of feature. Used to when
        constructing sklearn pipelines.
    plot_svm_iters : bool, default False
        If True, plots the SVM iterations.

    Returns
    -------
    did_converge : bool
        If true, irl loop converged and results returned are non-null.
    muE : array-like<float>. Shape(n_expert_demos, n_objectives)
        Expert demonstration feature expectations.
    muE_hold : array-like<float>. Shape(n_expert_demos, n_objectives)
        Expert feature expectations on the hold out set.
    muE_perf_hold : pandas.DataFrame
        Results of the performance measure during the expert's holdout demos.
    df_irl : pandas.DataFrame
        A collection of results where each row represents either an expert demo
        (and therefore a positive SVM training example) or a learned policy
        (and therefore a negative SVM training example). Includes relevant
        items like learned weights, feature expectations, error, etc.
    weights : array-like<float>. Shape(n_irl_loop_iters,  n_objectives).
        The learned weights for each iteration of the IRL loop.
    t_hold : array-like<float>. Shape(n_irl_loop_iters)
        The irl error on the holdout set.
    clf_pol : research.rl.env.clf_mdp.ClassificationMDPPolicy
        The classification MDP optimal policy.
    irl_loop_runtime : float
        The runtime of the IRL loop, in seconds.

    """
    weight_adjusts_list = exp_info["WEIGHT_ADJUSTS_LIST"]

    # Initiate objectives
    objectives = []
    for obj_name in exp_info["FEAT_EXP_OBJECTIVE_NAMES"]:
        objectives.append(OBJ_LOOKUP_BY_NAME[obj_name]())
    feat_obj_set = ObjectiveSet(objectives)
    del objectives
    # Reset the objective set since they get fitted in each trial run
    feat_obj_set.reset()

    objectives = []
    for obj_name in exp_info["PERF_MEAS_OBJECTIVE_NAMES"]:
        objectives.append(OBJ_LOOKUP_BY_NAME[obj_name]())
    perf_obj_set = ObjectiveSet(objectives)
    del objectives
    # Reset the objective set since they get fitted in each trial run
    perf_obj_set.reset()

    # Set observability level
    if "FO" in exp_info["IRL_METHOD"]:
        CAN_OBSERVE_Y = True
    else:
        CAN_OBSERVE_Y = False

    # Read in dataset
    if X is None or y is None or feature_types is None:
        X, y, feature_types = generate_dataset(
            exp_info["DATASET"],
            n_samples=exp_info["N_DATASET_SAMPLES"],
        )

    bias_types = exp_info["BIAS_TYPES"]

    # These are the feature types that will be used as inputs for the expert
    # classifier.
    expert_demo_feature_types = feature_types

    # These are the feature types that will be used in the classifier that will
    # predict `y` given `X` when learning the optimal policy for a given reward
    # function.
    irl_loop_feature_types = feature_types

    expert_algo_lookup = generate_expert_algo_lookup(expert_demo_feature_types)

    #
    # Split data into 3 sets.
    #     1. Demo: for computing expert demos AND learning muL - muE diffs
    #     2. Hold – computes the unbiased values for muL and t (dataset is
    #        never shown to the IRL learning algo.)
    #
    # But furst check that split doesn't have to big of a difference between demos and
    # holdout, otherwise it messes up interpretations.
    #
    split_is_okay = False
    max_muE_cosine_dist_split = 0.001

    while not split_is_okay:
        X_demo, X_hold, y_demo, y_hold = train_test_split(X, y, test_size=0.2)

        # Generate expert demonstrations to learn from
        muE, demoE = generate_demos_k_folds(
            exp_info,
            X=X_demo,
            y=y_demo,
            clf=expert_algo_lookup[exp_info["EXPERT_ALGO"]],
            obj_set=feat_obj_set,
            n_demos=exp_info["N_EXPERT_DEMOS"],
            bias_types=bias_types,
        )

        # Fit expert on entire training dataset. This is used when computing demos
        # on the holdout set for performance measurement.
        expert_hold_clf = expert_algo_lookup[exp_info["EXPERT_ALGO"]]
        expert_hold_clf.fit(X_demo, y_demo)
        expert_hold_clf = add_classifier_bias(expert_hold_clf, bias_types)
        demoE_hold = generate_demo(
            clf=expert_hold_clf,
            X_test=X_demo,
            y_test=y_demo,
            can_observe_y=CAN_OBSERVE_Y,
        )
        demoE_hold = add_demo_bias(
            demoE_hold, unfairness_types=bias_types, dataset=exp_info["DATASET"]
        )
        muE_hold = np.array([feat_obj_set.compute_demo_feature_exp(demoE_hold)])
        muE_perf_hold = np.array([perf_obj_set.compute_demo_feature_exp(demoE_hold)])

        for demo_i, _ in enumerate(muE):
            split_is_okay = True
            if cosine(muE[demo_i], muE_hold[demo_i]) > max_muE_cosine_dist_split:
                split_is_okay = False
                logging.info(
                    f"INFO: Split check failed: {cosine(muE[demo_i], muE_hold[demo_i])} > {max_muE_cosine_dist_split}, retrying split..."
                )

    muE_unbiased, demoE_unbiased = generate_demos_k_folds(
        exp_info,
        X=X_demo,
        y=y_demo,
        clf=expert_algo_lookup[exp_info["EXPERT_ALGO"]],
        obj_set=feat_obj_set,
        n_demos=exp_info["N_EXPERT_DEMOS"],
        bias_types=[],
    )

    raw_clf = expert_algo_lookup[exp_info["EXPERT_ALGO"]]
    raw_clf.fit(X_demo, y_demo)
    ref_demo = generate_demo(raw_clf, X_demo, y_demo)

    logging.info(f"muE_unbiased:\n{muE_unbiased}")
    logging.info(f"muE:\n{muE}")
    logging.info(f"muE_hold:\n{muE_hold}")
    logging.info(f"muE_perf_hold:\n{muE_perf_hold}")

    ##
    # Run IRL loop.
    # Create a clf dataset where inputs are feature expectations and outputs
    # are whether the policy is expert or learned through IRL iterations. Then
    # train an SVM classifier on this dataset. Then extract the weights of the
    # svm and use them as the weights for the "reward" function. Then use this
    # reward function to learn a policy (classifier). Then compute the feature
    # expectations from this classifer on the irl hold-out set. Then compute
    # the error between the feature expectations of this learned clf and the
    # demonstration feature exp. If this error is less than epsilon, stop. The
    # reward function is the final set of weights.
    ##

    # Initiate variables needed to run IRL Loop
    x_cols = (
        irl_loop_feature_types["boolean"]
        + irl_loop_feature_types["categoric"]
        + irl_loop_feature_types["continuous"]
    )
    x_cols.remove("z")
    feat_obj_set_cols = [obj.name for obj in feat_obj_set.objectives]
    perf_obj_set_cols = [obj.name for obj in perf_obj_set.objectives]

    # Set aggregator variables for the main IRL while loop
    max_abs_subdom_hist = []
    sum_abs_subdom_hist = []
    max_rel_subdom_hist = []
    sum_rel_subdom_hist = []
    t = []  # Errors for each iteration
    t_hold = []  # Errors on hold out set for each iteration
    mu_delta_l2norm_hist = []
    mu_delta_l2norm_hold_hist = []
    weights = []
    i = 0
    demo_history = []
    demo_hold_history = []
    mu_history = []
    mu_hold_history = []
    mu_perf_hold_history = []
    mu_best_history = []
    mu_best_hold_history = []
    mu_perf_best_hold_history = []

    # Generate initial learned policies to serve as negative training examples
    # for the SVM IRL classifier.
    mu = []
    for non_expert_algo in exp_info["NON_EXPERT_ALGOS"]:
        # Demo set
        _mu, _demos = generate_demos_k_folds(
            exp_info,
            X=X_demo,
            y=y_demo,
            clf=expert_algo_lookup[non_expert_algo],
            obj_set=feat_obj_set,
            n_demos=1,
            bias_types=bias_types,
        )
        mu.append(_mu[0])

        # Holdout set
        _mu_hold, _demos_hold = generate_demos_k_folds(
            exp_info,
            X=X_demo,  # Is this supposed to be X_hold and y_hold?
            y=y_demo,
            clf=expert_algo_lookup[non_expert_algo],
            obj_set=feat_obj_set,
            n_demos=1,
            bias_types=bias_types,
        )
        mu_deltas_hold = _mu_hold - muE_hold
        mu_deltas_hold_l2norm = np.linalg.norm(mu_deltas_hold, ord=2)
        mu_delta_l2norm_hold_hist.append(mu_deltas_hold_l2norm)

    mu_delta_l2norm_hold_hist_backup = mu_delta_l2norm_hold_hist.copy()

    mu = np.array(mu)
    logging.info(f"muL:\n{mu}")
    # X_biased_exp = pd.DataFrame(muE_unbiased, columns=feat_obj_set_cols)
    X_irl_exp = pd.DataFrame(muE, columns=feat_obj_set_cols)
    y_irl_exp = pd.Series(np.ones(exp_info["N_EXPERT_DEMOS"]), dtype=int)
    X_irl_learn = pd.DataFrame(mu, columns=feat_obj_set_cols)
    y_irl_learn = pd.Series(np.zeros(len(mu)), dtype=int)

    # Start the IRL Loop
    logging.debug("")
    logging.debug("Starting IRL Loop ...")

    done = False  # When true, breaks IRL loop
    is_stuck_in_loop = False
    return_vals = []
    irl_loop_start = datetime.datetime.now()
    for weight_adjusts in weight_adjusts_list:
        while not done:
            if weight_adjusts == ():
                logging.info(f"\tIRL Loop iteration {i+1}/{exp_info['MAX_ITER']} ...")

                # Train SVM classifier that distinguishes which demonstrations are
                # expert and which were generated from this loop.
                logging.debug("\tFitting SVM classifier...")
                X_irl = pd.concat([X_irl_exp, X_irl_learn], axis=0).reset_index(
                    drop=True
                )
                y_irl = pd.concat([y_irl_exp, y_irl_learn], axis=0).reset_index(
                    drop=True
                )
                try:
                    allow_pos_weights = not exp_info["ALLOW_NEG_WEIGHTS"]
                    randomly_simluate_stuck = np.random.rand()
                    # if randomly_simluate_stuck < 0 or is_stuck_in_loop:
                    #     is_stuck_in_loop = False
                    #     # Randomly sample 1 or 2 negative samples (learned policies)
                    #     n_samples = max(1, np.random.choice(int(np.log((i+1)**2))+1))
                    #     pos_idx = X_irl[0:exp_info['N_EXPERT_DEMOS']].index
                    #     neg_idx = X_irl[exp_info['N_EXPERT_DEMOS']:].sample(n_samples).index
                    #     _X_irl = pd.concat([X_irl.loc[pos_idx], X_irl.loc[neg_idx]])
                    #     _y_irl = pd.concat([y_irl.loc[pos_idx], y_irl.loc[neg_idx]])
                    #     logging.info(f"\t\tStuck in loop or randomized sample: sampling {n_samples} random negative demos")
                    #     print('Sampled IRL points for SVM')
                    #     irl_display = _X_irl.copy()
                    #     irl_display['is_expert'] = _y_irl.copy()
                    #     display(irl_display)
                    # else:
                    #     _X_irl = X_irl
                    #     _y_irl = y_irl
                    svm = SVM(positive_weights_only=allow_pos_weights).fit(X_irl, y_irl)
                except ValueError as e:
                    if e.args[0] != "No support vectors found.":
                        raise (e)
                    logging.info("\t\tAllowing negative weights.")
                    svm = SVM(positive_weights_only=False).fit(X_irl, y_irl)

                wi = svm.weights(norm="l1")

                # New as of 01/07/2024
                # If weights have any negative values, renormalize the set to be
                # positive but still same rank order. I think this will help when IRL
                # can't land on a 1, 0, 0 weight b/c the negative weights prevent the
                # 1,0,0 from beign learned.
                # if np.any(wi < 0):
                #     wi = (wi - wi.min()) / np.linalg.norm(wi, ord=1)

                ##
                # Learn a policy (clf_pol) from the reward (SVM) weights.
                ##

                # Fit a classifier that predicts `y` from `X`.
                # TODO 12/16/2023: remove this. we no longer use it.
                logging.debug("\tFitting `y|x` predictor for clf policy...")
                clf = sklearn_clf_pipeline(
                    feature_types=irl_loop_feature_types,
                    clf_inst=RandomForestClassifier(),
                    # clf_inst=XGBClassifier(),
                )
                clf.fit(X_demo, y_demo)

                demo_df = pd.DataFrame(X_demo)
                #
                # JDB 06/27/2023 – NOTE: mu0s need to be calculated using the predicted
                # labels, not the true labels. Otherwise the optimizer solves for a
                # policy that doesn't reflect observability. I discovered this issue by
                # giving weights [0, 1, 0] into the run_trial_source_domain and
                # noticing that the ouptput had poor demographic parity.
                #
                demo_df["y"] = y_demo
            else:
                # Reset IRL loop history vars
                max_abs_subdom_hist = []
                sum_abs_subdom_hist = []
                max_rel_subdom_hist = []
                sum_rel_subdom_hist = []
                t = []  # Errors for each iteration
                t_hold = []  # Errors on hold out set for each iteration
                mu_delta_l2norm_hist = []
                mu_delta_l2norm_hold_hist = mu_delta_l2norm_hold_hist_backup.copy()
                weights = []
                i = 0
                demo_history = []
                demo_hold_history = []
                mu_history = []
                mu_hold_history = []
                mu_perf_hold_history = []
                mu_best_history = []
                mu_best_hold_history = []
                mu_perf_best_hold_history = []
                X_irl_exp = pd.DataFrame(muE, columns=feat_obj_set_cols)
                y_irl_exp = pd.Series(np.ones(exp_info["N_EXPERT_DEMOS"]), dtype=int)
                X_irl_learn = pd.DataFrame(mu, columns=feat_obj_set_cols)
                y_irl_learn = pd.Series(np.zeros(len(mu)), dtype=int)
                wi = unadjusted_best_weight.copy()
                for weight_adjust in weight_adjusts:
                    mul_factor = weight_adjust[1]
                    if weight_adjust[0] == "mul_negative_weights":
                        wi[wi < 0] = wi[wi < 0] * mul_factor
                    # elif weight_adjust[0] == "sqrt_negative_weights":
                    #     wi[wi < 0] = 1 - np.sqrt((-wi[wi < 0]) + 1)
                    # elif weight_adjust[0] == "ln_negative_weights":
                    #     wi[wi < 0] = -np.log((-wi[wi < 0]) + 1)
                    # Other function ideas:
                    # y\ =\ -\sqrt{-x}
                    # y\ =-\ x\ ^{2}
                    # y=-\ln\left(1-x\left(e-1\right)\right)
                    # y=\frac{-e^{-x}+1}{e-1}
                done = True
            # Learn a policy that maximizes the reward function.
            # logging.debug('\tComputing the optimal policy given reward weights and `y|x` classifier...')
            weights.append(wi)

            reward_weights = {
                obj.name: wi[j] for j, obj in enumerate(feat_obj_set.objectives)
            }
            clf_pol = compute_optimal_policy(
                clf_df=demo_df,  # NOT the dataset used to train the C_{Y_Z,X} clf
                clf=clf,
                x_cols=x_cols,
                obj_set=feat_obj_set,
                reward_weights=reward_weights,
                skip_error_terms=True,
                method=exp_info["METHOD"],
                min_freq_fill_pct=exp_info["MIN_FREQ_FILL_PCT"],
                restrict_y=exp_info["RESTRICT_Y_ACTION"],
            )

            ##
            # Measure and record the error of the learned policy, and keep it as
            # a negative training example for next IRL Loop iteration.
            ##

            # Compute feature expectations of the learned policy
            logging.debug("\tGenerating learned demostration...")
            demo = generate_demo(clf_pol, X_demo, y_demo, can_observe_y=CAN_OBSERVE_Y)
            demo_hold = generate_demo(clf_pol, X_hold, y_hold, can_observe_y=False)
            demo_history.append(demo)
            demo_hold_history.append(demo_hold)
            muj = feat_obj_set.compute_demo_feature_exp(demo)
            muj_hold = feat_obj_set.compute_demo_feature_exp(demo_hold)
            muj_perf_hold = perf_obj_set.compute_demo_feature_exp(demo_hold)
            mu_history.append(muj)
            mu_hold_history.append(muj_hold)
            mu_perf_hold_history.append(muj_perf_hold)
            logging.info(
                f"\t\t muL[{i}] \t\t= {str(np.round(muj, 2)).replace('0.', '.')}"
            )
            logging.debug(f"\t\t muL_hold[{i}] = {np.round(muj_hold, 2)}")

            # Append policy's feature expectations to irl clf dataset
            X_irl_learn_i = pd.DataFrame(np.array([muj]), columns=feat_obj_set_cols)
            y_irl_learn_i = pd.Series(np.zeros(1), dtype=int)
            X_irl_learn = pd.concat([X_irl_learn, X_irl_learn_i], axis=0)
            y_irl_learn = pd.concat([y_irl_learn, y_irl_learn_i], axis=0)

            # Compute error of the learned policy: t[i] = wT(muE-mu[j])
            # This is equivalent to computing the SVM margin.
            ti, best_j, mu_delta, mu_delta_l2norm = irl_error(
                wi,
                muE,
                mu_history,
                dot_weights_feat_exp=exp_info["DOT_WEIGHTS_FEAT_EXP"],
                svm_margin=svm.margin(),
            )
            # Do it for the hold-out set as well.
            ti_hold, best_j_hold, mu_delta_hold, mu_delta_l2norm_hold = irl_error(
                wi,
                muE_hold,
                mu_hold_history,
                dot_weights_feat_exp=exp_info["DOT_WEIGHTS_FEAT_EXP"],
                svm_margin=svm.margin(),
            )
            # Compute subdominance metric for the learned policy
            # Begin by subdividing the demos into groups
            # TODO: add holdout demo groups as well and compute subdominance
            # TODO: while i'm at it, make a proper train/validation/test split
            raw_demo = ref_demo.copy()  # should be the same shape
            clf_demo = demo.copy()  # should also be the same shape
            raw_demo = raw_demo.sample(frac=1).reset_index(drop=True)
            clf_demo = clf_demo.sample(frac=1).reset_index(drop=True)
            raw_demos = np.array_split(raw_demo, exp_info["N_SUBDOMINANCE_GROUPS"])
            clf_demos = np.array_split(clf_demo, exp_info["N_SUBDOMINANCE_GROUPS"])
            raw_demos_feat_loss = np.array(
                [
                    compute_relevant_feat_loss(exp_info, raw_demo_group)
                    for raw_demo_group in raw_demos
                ]
            )
            clf_demos_feat_loss = np.array(
                [
                    compute_relevant_feat_loss(exp_info, clf_demo_group)
                    for clf_demo_group in clf_demos
                ]
            )
            alphas = compute_alphas(raw_demos_feat_loss, clf_demos_feat_loss)
            # Compute max-aggregated absolute subdominance:
            max_abs_subdominance = compute_subdominance(
                alphas,
                raw_demos_feat_loss,
                clf_demos_feat_loss,
                relative=False,
                sum_agg=False,
            )
            # Compute sum-aggregated absolute subdominance:
            sum_abs_subdominance = compute_subdominance(
                alphas,
                raw_demos_feat_loss,
                clf_demos_feat_loss,
                relative=False,
                sum_agg=True,
            )
            # Compute max-aggregated relative subdominance:
            max_rel_subdominance = compute_subdominance(
                alphas,
                raw_demos_feat_loss,
                clf_demos_feat_loss,
                relative=True,
                sum_agg=False,
            )
            # Compute sum-aggregated relative subdominance:
            sum_rel_subdominance = compute_subdominance(
                alphas,
                raw_demos_feat_loss,
                clf_demos_feat_loss,
                relative=True,
                sum_agg=True,
            )

            max_abs_subdom_hist.append(max_abs_subdominance)
            sum_abs_subdom_hist.append(sum_abs_subdominance)
            max_rel_subdom_hist.append(max_rel_subdominance)
            sum_rel_subdom_hist.append(sum_rel_subdominance)
            mu_best_history.append(mu_history[best_j])
            mu_best_hold_history.append(mu_hold_history[best_j])
            mu_perf_best_hold_history.append(mu_perf_hold_history[best_j])
            t.append(ti)
            t_hold.append(ti_hold)
            mu_delta_l2norm_hist.append(mu_delta_l2norm)
            mu_delta_l2norm_hold_hist.append(mu_delta_l2norm_hold)
            logging.info(
                f"\t\t Best mu_delta[{i}] \t= {str(np.round(mu_delta, 2)).replace('0.', '.')}"
            )
            logging.info(
                f"\t\t Best mu_delta_hold[i] \t= {str(np.round(mu_delta_hold, 2)).replace('0.', '.')}"
            )
            logging.info(f"\t\t t[{i}] \t\t= {t[i]:.5f}")
            logging.info(f"\t\t t_hold[i] \t= {t_hold[i]:.5f}")
            logging.info(
                f"\t\t weights[{i}] \t= {str(np.round(weights[i], 2)).replace('0.', '.')}"
            )

            # If reached maximum iterations
            if i >= exp_info["MAX_ITER"] - 1:
                logging.info(f"\nReached max iters.")
                done = True
                break

            # If haven't reached max iterations but error is below epsilon
            elif ti < exp_info["EPSILON"] and ti <= min(t):
                # Check if accuracy weight is zero
                # TODO: remove this
                if np.allclose(weights[i][0], 0, atol=1e-5):
                    logging.info("\t\tAccuracy weight is zero, continuing")
                    i += 1
                    continue

                # If non-neg weights allowed or all weights non-neg
                if exp_info["ALLOW_NEG_WEIGHTS"] or np.all(wi > -1e-5):
                    done = True
                    break

            # If error is going back up, stop
            elif len(t) > 1 and ti > t[-2]:
                logging.info(f"\n\nError is going back up. Stoping.")
                done = True
                break
            # Check if a new best error has been found in last i/2 iterations. This
            # is essentially a "smart" convervence to not waste time on more
            # iterations if it's not improving.
            elif (i - np.argsort(t)[0]) > exp_info["EARLY_STOP_NO_NEW_BEST_ITERS"]:
                logging.info(
                    f"\n\t\tNo new best in last {exp_info['EARLY_STOP_NO_NEW_BEST_ITERS']} iterations, so stopping early."
                )
                done = True
                break
            # Check if loop is stuck in local optimum by checking if error and
            # weights are the same as previous iteration.
            elif (
                i > 0
                and abs(t[i] - t[i - 1]) < 1e-3
                and np.allclose(weights[i], weights[i - 1], atol=1e-3)
            ):
                logging.info("\t\tStuck in loop")
                is_stuck_in_loop = True
                i += 1
            # Check if error is infinite (output by fair_irl.irl_error() when
            # accuracy weight is zero) and treat it like it's stuck.
            elif i > 0 and t[i] == np.inf:
                logging.info("\t\tInfinite error: Treating like stuck in loop")
                is_stuck_in_loop = True
                i += 1
            else:
                i += 1

            # End IRL Loop
        done = False

        irl_loop_runtime = (datetime.datetime.now() - irl_loop_start).total_seconds()

        # If solution not sufficient for use, exit early
        # if min(t) > exp_info['IGNORE_RESULTS_EPSILON']:
        if min(mu_delta_l2norm_hist) > exp_info["IGNORE_RESULTS_EPSILON"]:
            # logging.info(f"IGNORING RESULTS BECAUSE BEST ERROR {min(t):.3f} > {exp_info['IGNORE_RESULTS_EPSILON']:.3f}")
            logging.info(
                f"IGNORING RESULTS BECAUSE BEST ERROR {min(mu_delta_l2norm_hist):.3f} > {exp_info['IGNORE_RESULTS_EPSILON']:.3f}"
            )
            return_val = (None, None, None, None, None, None, None, None, None)
            return_vals.append(return_val)
            return return_vals

        # Find best weights based on smallest error (with nonnegative weights)
        t_arg_smallest_to_largest = np.argsort(t)
        mu_delta_l2_norm_arg_smallest_to_largest = np.argsort(mu_delta_l2norm_hist)
        best_iter = t_arg_smallest_to_largest[0]
        # best_iter = mu_delta_l2_norm_arg_smallest_to_largest[0]
        # print('best_iter', best_iter)
        if not exp_info["ALLOW_NEG_WEIGHTS"]:
            best_t = None
            best_t_i = 0
            best_t_done = False
            while not best_t_done:
                if best_t_i >= len(t):
                    np.info(f"best_weight:\t {np.round(weights[best_iter], 3)}")
                    # display(weights)
                    raise ValueError("Only negative weights learned")
                best_iter = t_arg_smallest_to_largest[best_t_i]
                best_t = t[best_iter]
                # if np.all(weights[best_iter] > -1e-5) and best_t <= exp_info['IGNORE_RESULTS_EPSILON']:
                if (
                    np.all(weights[best_iter] > -1e-5)
                    and mu_delta_l2norm_hist[best_iter]
                    <= exp_info["IGNORE_RESULTS_EPSILON"]
                ):
                    best_t_done = True
                best_t_i += 1

        ##
        # Book keeping stuff for the trial.
        ##

        # Compare the best learned policy with the expert demonstrations
        best_demo = demo_history[best_iter]
        best_weight = weights[best_iter]
        if weight_adjusts == ():
            unadjusted_best_weight = weights[best_iter].copy()
        logging.debug("Best iteration: " + str(best_iter))
        logging.info(
            f"Best Learned Policy yhat (not real yhat since doesn't factor mu0): {best_demo['yhat'].mean():.3f}"
        )
        logging.info(f"best weight:\t {np.round(best_weight, 3)}")

        best_row = exp_info["N_EXPERT_DEMOS"] + exp_info["N_INIT_POLICIES"] + best_iter

        # Generate a dataframe for results gathering.
        X_irl = pd.concat([X_irl_exp, X_irl_learn], axis=0).reset_index(drop=True)
        y_irl = pd.concat([y_irl_exp, y_irl_learn], axis=0).reset_index(drop=True)
        df_irl = X_irl.copy()
        df_irl["is_expert"] = y_irl.copy()
        for i, col in enumerate(feat_obj_set_cols):
            df_irl[f"muL_best_{col}"] = (
                np.zeros(
                    exp_info["N_EXPERT_DEMOS"] + exp_info["N_INIT_POLICIES"]
                ).tolist()
                + np.array(mu_best_history)[:, i].tolist()
            )
            df_irl[f"muL_hold_{col}"] = (
                np.zeros(
                    exp_info["N_EXPERT_DEMOS"] + +exp_info["N_INIT_POLICIES"]
                ).tolist()
                + np.array(mu_hold_history)[:, i].tolist()
            )
            df_irl[f"muL_best_hold_{col}"] = (
                np.zeros(
                    exp_info["N_EXPERT_DEMOS"] + exp_info["N_INIT_POLICIES"]
                ).tolist()
                + np.array(mu_best_hold_history)[:, i].tolist()
            )
        for i, col in enumerate(perf_obj_set_cols):
            df_irl[f"muL_perf_best_hold_{col}"] = (
                np.zeros(
                    exp_info["N_EXPERT_DEMOS"] + exp_info["N_INIT_POLICIES"]
                ).tolist()
                + np.array(mu_perf_best_hold_history)[:, i].tolist()
            )
        df_irl["is_init_policy"] = (
            np.zeros(exp_info["N_EXPERT_DEMOS"]).tolist()
            + np.ones(exp_info["N_INIT_POLICIES"]).tolist()
            + np.zeros(len(t)).tolist()
        )
        df_irl["learn_idx"] = list(-1 * np.ones(exp_info["N_EXPERT_DEMOS"])) + list(
            np.arange(exp_info["N_INIT_POLICIES"] + len(t))
        )
        for i, col in enumerate(feat_obj_set_cols):
            df_irl[f"{col}_weight"] = np.zeros(
                exp_info["N_EXPERT_DEMOS"] + exp_info["N_INIT_POLICIES"]
            ).tolist() + [w[i] for w in weights]
        df_irl["max_abs_subdominance"] = (
            np.inf
            * np.ones(
                exp_info["N_EXPERT_DEMOS"] + exp_info["N_INIT_POLICIES"], dtype=float
            )
        ).tolist() + max_abs_subdom_hist
        df_irl["sum_abs_subdominance"] = (
            np.inf
            * np.ones(
                exp_info["N_EXPERT_DEMOS"] + exp_info["N_INIT_POLICIES"], dtype=float
            )
        ).tolist() + sum_abs_subdom_hist
        df_irl["max_rel_subdominance"] = (
            np.inf
            * np.ones(
                exp_info["N_EXPERT_DEMOS"] + exp_info["N_INIT_POLICIES"], dtype=float
            )
        ).tolist() + max_rel_subdom_hist
        df_irl["sum_rel_subdominance"] = (
            np.inf
            * np.ones(
                exp_info["N_EXPERT_DEMOS"] + exp_info["N_INIT_POLICIES"], dtype=float
            )
        ).tolist() + sum_rel_subdom_hist
        df_irl["t"] = (
            list(
                np.inf
                * (np.ones(exp_info["N_EXPERT_DEMOS"] + exp_info["N_INIT_POLICIES"]))
            )
            + t
        )
        df_irl["t_hold"] = (
            list(
                np.inf
                * (np.ones(exp_info["N_EXPERT_DEMOS"] + exp_info["N_INIT_POLICIES"]))
            )
            + t_hold
        )
        df_irl["mu_delta_l2norm"] = (
            np.inf
            * np.ones(
                exp_info["N_EXPERT_DEMOS"] + exp_info["N_INIT_POLICIES"], dtype=float
            )
        ).tolist() + mu_delta_l2norm_hist
        df_irl["mu_delta_l2norm_hold"] = (
            (np.inf * np.ones(exp_info["N_EXPERT_DEMOS"], dtype=float)).tolist()
            # 01/21/2024: Started recording mu_delta_l2norm_hold_hist for the
            # initial policies since I am plotting this now for our paper.
            # +exp_info['N_INIT_POLICIES']).tolist()
            + mu_delta_l2norm_hold_hist
        )
        logging.debug("Experiment Summary")
        # display(df_irl.round(3))

        # End regular IRL trial
        return_val = (
            best_row,
            muE_unbiased,
            muE,
            muE_hold,
            muE_perf_hold,
            df_irl,
            weights,
            t_hold,
            clf_pol,
            irl_loop_runtime,
        )
        return_vals.append(return_val)
    return return_vals


def compute_relevant_feat_loss(exp_info, demo):
    objectives = []
    for obj_name in exp_info["SUBDOMINANCE_METRICS_LIST"]:
        objectives.append(OBJ_LOOKUP_BY_NAME[obj_name]())
    feat_obj_set = ObjectiveSet(objectives)
    del objectives

    # Generate feature expectations of the demo.
    demo_feat_exp = np.array(feat_obj_set.compute_demo_feature_exp(demo))
    # Invert expectations to be loss, where lower is better
    demo_feat_exp = -demo_feat_exp + 1
    return demo_feat_exp


def run_bias_experiment(
    exp_info, source_X=None, source_y=None, source_feature_types=None
):
    """
    Runs experiment for source domain and optionally target domain based on
    the parameters in `exp_info`.

    Parameters
    ----------
    exp_info : dict
        Experiment parameters.

    Returns
    -------
    source_clf_pol : research.rl.env.clf_mdp.ClassificationMDPPolicy
        The classification MDP optimal policy for the source domain. This is
        returned only for debugging or inpsection purposes.
    target_clf_pol : research.rl.env.clf_mdp.ClassificationMDPPolicy
        The classification MDP optimal policy for the target domain. This is
        returned only for debugging or inpsection purposes.

    Persists
    --------
    exp_df : pandas.DataFrame
        Saves experiment results as a CSV where each row in the CSV represents
        the relevant results of one trial. The file is stored as
        "/data/experiment_output/fair_irl/exp_results/{timestamp}.csv"
    exp_info : dict
        Saves the experiment parameters and metadata metadata as a JSON file
        "./../../data/experiment_output/fair_irl/exp_info/{timestamp}.json"
    source_X : pandas.DataFrame, Optional
        The X (including z) columns for the source domain.
    source_y : pandas.Series, Optional
        Just the y column for the source domain.
    source_feature_types : dict<str, array-like>, Optional
        Mapping of column names to their type of feature. Used to when
        constructing sklearn pipelines for the source domain.
    target_X : pandas.DataFrame, Optional
        The X (including z) columns for the target domain.
    target_y : pandas.Series, Optional
        Just the y column for the target domain.
    target_feature_types : dict<str, array-like>, Optional
        Mapping of column names to their type of feature. Used to when
        constructing sklearn pipelines for the target domain.
    """
    logging.info(f"exp_info: {exp_info}")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"Experiment timestamp: {timestamp}")

    objectives = []
    for obj_name in exp_info["FEAT_EXP_OBJECTIVE_NAMES"]:
        objectives.append(OBJ_LOOKUP_BY_NAME[obj_name]())
    feat_obj_set = ObjectiveSet(objectives)
    del objectives

    objectives = []
    for obj_name in exp_info["PERF_MEAS_OBJECTIVE_NAMES"]:
        objectives.append(OBJ_LOOKUP_BY_NAME[obj_name]())
    perf_obj_set = ObjectiveSet(objectives)
    del objectives

    results = []
    weight_adjust_names = []
    for weight_adjusts in exp_info["WEIGHT_ADJUSTS_LIST"]:
        results.append([])
        weight_adjust_names.append("".join(str(weight_adjusts)))
    trial_i = 0

    while trial_i < exp_info["N_TRIALS"]:
        # logging.info(f"\n\nTRIAL {trial_i}\n")

        source_trial_start = datetime.datetime.now()

        # Run trials to learn weights on source domain
        return_vals = run_trial_source_domain(
            exp_info,
            X=source_X,
            y=source_y,
            feature_types=source_feature_types,
        )
        for i, (weight_adjusts, return_val) in enumerate(
            zip(exp_info["WEIGHT_ADJUSTS_LIST"], return_vals)
        ):
            (
                converge_idx,
                muE_unbiased,
                muE,
                muE_feat_hold,
                muE_perf_hold,
                df_irl,
                weights,
                t_hold,
                source_clf_pol,
                source_irl_loop_runtime,
            ) = return_val

            source_trial_runtime = (
                datetime.datetime.now() - source_trial_start
            ).total_seconds()

            source_trial_inputsize = None
            if hasattr(source_clf_pol, "mdp"):
                source_trial_inputsize = source_clf_pol.mdp.n_states_

            # If feature expectation error wasn't sufficiently small, skip.
            if converge_idx is None:
                logging.info(f"\nTrial {trial_i} did not converge.\n")
                trial_i += 1
                continue

            # Aggregate trial results
            _result = new_trial_result(
                converge_idx,
                feat_obj_set,
                perf_obj_set,
                muE_unbiased,
                muE,
                muE_feat_hold,
                muE_perf_hold,
                df_irl,
                source_trial_runtime,
                source_irl_loop_runtime,
                source_trial_inputsize,
            )
            results[i].append(
                _result
            )  # each "i" list should contain one result for each trial

            # Persist the irl loop details so we can look at convergence
            df_irl.to_csv(
                f"./../../data/experiment_output/fair_irl/exp_conv_details/{timestamp}_{weight_adjust_names[i]}_trial{trial_i}.csv",
                index=None,
            )

        trial_i += 1

    # Persist trial results
    for i, weight_adjusts in enumerate(exp_info["WEIGHT_ADJUSTS_LIST"]):
        exp_df = generate_single_exp_results_df(
            feat_obj_set,
            perf_obj_set,
            results[i],
        )
        exp_df.to_csv(
            f"./../../data/experiment_output/fair_irl/exp_results/{timestamp}_{weight_adjust_names[i]}.csv",
            index=None,
        )

        # Persist trial info
        exp_info["timestamp"] = timestamp
        exp_info["WEIGHT_ADJUSTS"] = weight_adjusts
        fp = f"./../../data/experiment_output/fair_irl/exp_info/{timestamp}_{weight_adjust_names[i]}.json"
        json.dump(exp_info, open(fp, "w"))
