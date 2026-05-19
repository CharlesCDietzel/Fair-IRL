import copy
import datetime
import itertools
import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.base
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import (
    ClassificationMoment,
    Moment,
    DemographicParity,
    ErrorRate,
    ZeroOneLoss,
    TruePositiveRateParity,
    BoundedGroupLoss,
    EqualizedOdds,
    ErrorRateParity,
    ExponentiatedGradient,
    ZeroOneLoss,
)
from matplotlib.ticker import FormatStrFormatter
from scipy.spatial.distance import cosine
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from research.irl.fair_irl import *
from research.ml.svm import SVM
from research.rl.env.clf_mdp import *
from research.rl.env.clf_mdp_policy import *
from research.rl.env.objectives import *
from research.utils import *

from .datasets import *

# Color palette for plotting
cp = sns.color_palette()


# Objective lookup
OBJ_LOOKUP_BY_NAME = {
    "Acc": AccuracyObjective,
    "AccPar": AccuracyParityObjective,
    "DemPar": DemographicParityObjective,
    "EqOpp": EqualOpportunityObjective,
    "FPRPar": FalsePositiveRateParityObjective,
    "EqOdds": EqualizedOddsObjective,
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
        clf=eq_odds_thresh_opt,
        sensitive_features="z",
    )

    # Demographic Parity Reduction with difference_bound=0.01
    dem_par = DemographicParity(difference_bound=0.01)
    dem_par_exp_grad = ExponentiatedGradient(
        sample_weight_name="classifier__sample_weight",
        constraints=dem_par,
        estimator=sklearn_clf_pipeline(
            feature_types=feature_types,
            # clf_inst=DecisionTreeClassifier(min_samples_leaf=10, max_depth=4),
            # clf_inst=RandomForestClassifier(),
            clf_inst=XGBClassifier(),
        ),
    )
    dem_par_red_wrapper = FairLearnSkLearnWrapper(
        clf=dem_par_exp_grad,
        sensitive_features="z",
        has_access_to_sensitive_features=False,
        clone_on_fit=True,
    )

    # Demographic Parity Reduction with difference_bound=0.01
    dem_par_1 = DemographicParity(difference_bound=0.01)
    dem_par_exp_grad_1 = ExponentiatedGradient(
        sample_weight_name="classifier__sample_weight",
        constraints=dem_par_1,
        estimator=sklearn_clf_pipeline(
            feature_types=feature_types,
            # clf_inst=DecisionTreeClassifier(min_samples_leaf=10, max_depth=4),
            # clf_inst=RandomForestClassifier(),
            clf_inst=XGBClassifier(),
        ),
    )
    dem_par_red_wrapper_1 = FairLearnSkLearnWrapper(
        clf=dem_par_exp_grad_1,
        sensitive_features="z",
        has_access_to_sensitive_features=False,
        clone_on_fit=True,
    )

    # Demographic Parity Reduction with difference_bound=0.02
    dem_par_2 = DemographicParity(difference_bound=0.02)
    dem_par_exp_grad_2 = ExponentiatedGradient(
        sample_weight_name="classifier__sample_weight",
        constraints=dem_par_2,
        estimator=sklearn_clf_pipeline(
            feature_types=feature_types,
            # clf_inst=DecisionTreeClassifier(min_samples_leaf=10, max_depth=4),
            # clf_inst=RandomForestClassifier(),
            clf_inst=XGBClassifier(),
        ),
    )
    dem_par_red_wrapper_2 = FairLearnSkLearnWrapper(
        clf=dem_par_exp_grad_2,
        sensitive_features="z",
        has_access_to_sensitive_features=False,
        clone_on_fit=True,
    )

    # Demographic Parity Reduction with difference_bound=0.03
    dem_par_3 = DemographicParity(difference_bound=0.03)
    dem_par_exp_grad_3 = ExponentiatedGradient(
        sample_weight_name="classifier__sample_weight",
        constraints=dem_par_3,
        estimator=sklearn_clf_pipeline(
            feature_types=feature_types,
            # clf_inst=DecisionTreeClassifier(min_samples_leaf=10, max_depth=4),
            # clf_inst=RandomForestClassifier(),
            clf_inst=XGBClassifier(),
        ),
    )
    dem_par_red_wrapper_3 = FairLearnSkLearnWrapper(
        clf=dem_par_exp_grad_3,
        sensitive_features="z",
        has_access_to_sensitive_features=False,
        clone_on_fit=True,
    )

    # Demographic Parity Reduction with difference_bound=0.04
    dem_par_4 = DemographicParity(difference_bound=0.04)
    dem_par_exp_grad_4 = ExponentiatedGradient(
        sample_weight_name="classifier__sample_weight",
        constraints=dem_par_4,
        estimator=sklearn_clf_pipeline(
            feature_types=feature_types,
            # clf_inst=DecisionTreeClassifier(min_samples_leaf=10, max_depth=4),
            # clf_inst=RandomForestClassifier(),
            clf_inst=XGBClassifier(),
        ),
    )
    dem_par_red_wrapper_4 = FairLearnSkLearnWrapper(
        clf=dem_par_exp_grad_4,
        sensitive_features="z",
        has_access_to_sensitive_features=False,
        clone_on_fit=True,
    )

    # Demographic Parity Reduction with difference_bound=0.05
    dem_par_5 = DemographicParity(difference_bound=0.05)
    dem_par_exp_grad_5 = ExponentiatedGradient(
        sample_weight_name="classifier__sample_weight",
        constraints=dem_par_5,
        estimator=sklearn_clf_pipeline(
            feature_types=feature_types,
            # clf_inst=DecisionTreeClassifier(min_samples_leaf=10, max_depth=4),
            # clf_inst=RandomForestClassifier(),
            clf_inst=XGBClassifier(),
        ),
    )
    dem_par_red_wrapper_5 = FairLearnSkLearnWrapper(
        clf=dem_par_exp_grad_5,
        sensitive_features="z",
        has_access_to_sensitive_features=False,
        clone_on_fit=True,
    )

    # Demographic Parity Reduction with difference_bound=0.06
    dem_par_6 = DemographicParity(difference_bound=0.06)
    dem_par_exp_grad_6 = ExponentiatedGradient(
        sample_weight_name="classifier__sample_weight",
        constraints=dem_par_6,
        estimator=sklearn_clf_pipeline(
            feature_types=feature_types,
            # clf_inst=DecisionTreeClassifier(min_samples_leaf=10, max_depth=4),
            # clf_inst=RandomForestClassifier(),
            clf_inst=XGBClassifier(),
        ),
    )
    dem_par_red_wrapper_6 = FairLearnSkLearnWrapper(
        clf=dem_par_exp_grad_6,
        sensitive_features="z",
        has_access_to_sensitive_features=False,
        clone_on_fit=True,
    )

    # Demographic Parity Reduction with difference_bound=0.07
    dem_par_7 = DemographicParity(difference_bound=0.07)
    dem_par_exp_grad_7 = ExponentiatedGradient(
        sample_weight_name="classifier__sample_weight",
        constraints=dem_par_7,
        estimator=sklearn_clf_pipeline(
            feature_types=feature_types,
            # clf_inst=DecisionTreeClassifier(min_samples_leaf=10, max_depth=4),
            # clf_inst=RandomForestClassifier(),
            clf_inst=XGBClassifier(),
        ),
    )
    dem_par_red_wrapper_7 = FairLearnSkLearnWrapper(
        clf=dem_par_exp_grad_7,
        sensitive_features="z",
        has_access_to_sensitive_features=False,
        clone_on_fit=True,
    )

    # Demographic Parity Reduction with difference_bound=0.08
    dem_par_8 = DemographicParity(difference_bound=0.08)
    dem_par_exp_grad_8 = ExponentiatedGradient(
        sample_weight_name="classifier__sample_weight",
        constraints=dem_par_8,
        estimator=sklearn_clf_pipeline(
            feature_types=feature_types,
            # clf_inst=DecisionTreeClassifier(min_samples_leaf=10, max_depth=4),
            # clf_inst=RandomForestClassifier(),
            clf_inst=XGBClassifier(),
        ),
    )
    dem_par_red_wrapper_8 = FairLearnSkLearnWrapper(
        clf=dem_par_exp_grad_8,
        sensitive_features="z",
        has_access_to_sensitive_features=False,
        clone_on_fit=True,
    )

    # Demographic Parity Reduction with difference_bound=0.09
    dem_par_9 = DemographicParity(difference_bound=0.09)
    dem_par_exp_grad_9 = ExponentiatedGradient(
        sample_weight_name="classifier__sample_weight",
        constraints=dem_par_9,
        estimator=sklearn_clf_pipeline(
            feature_types=feature_types,
            # clf_inst=DecisionTreeClassifier(min_samples_leaf=10, max_depth=4),
            # clf_inst=RandomForestClassifier(),
            clf_inst=XGBClassifier(),
        ),
    )
    dem_par_red_wrapper_9 = FairLearnSkLearnWrapper(
        clf=dem_par_exp_grad_9,
        sensitive_features="z",
        has_access_to_sensitive_features=False,
        clone_on_fit=True,
    )

    # Demographic Parity Reduction with difference_bound=0.1
    dem_par_10 = DemographicParity(difference_bound=0.1)
    dem_par_exp_grad_10 = ExponentiatedGradient(
        sample_weight_name="classifier__sample_weight",
        constraints=dem_par_10,
        estimator=sklearn_clf_pipeline(
            feature_types=feature_types,
            # clf_inst=DecisionTreeClassifier(min_samples_leaf=10, max_depth=4),
            # clf_inst=RandomForestClassifier(),
            clf_inst=XGBClassifier(),
        ),
    )
    dem_par_red_wrapper_10 = FairLearnSkLearnWrapper(
        clf=dem_par_exp_grad_10,
        sensitive_features="z",
        has_access_to_sensitive_features=False,
        clone_on_fit=True,
    )

    # Equal Opportunity Reduction
    eq_opp = TruePositiveRateParity(difference_bound=0.01)
    eq_opp_exp_grad = ExponentiatedGradient(
        sample_weight_name="classifier__sample_weight",
        constraints=eq_opp,
        estimator=sklearn_clf_pipeline(
            feature_types=feature_types,
            # clf_inst=DecisionTreeClassifier(min_samples_leaf=10, max_depth=4),
            # clf_inst=RandomForestClassifier(),
            clf_inst=XGBClassifier(),
        ),
    )
    eq_opp_red_wrapper = FairLearnSkLearnWrapper(
        clf=eq_opp_exp_grad,
        sensitive_features="z",
        has_access_to_sensitive_features=False,
        clone_on_fit=True,
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
        "DemParRed": dem_par_red_wrapper,
        "DemParRed_0.01": dem_par_red_wrapper_1,
        "DemParRed_0.02": dem_par_red_wrapper_2,
        "DemParRed_0.03": dem_par_red_wrapper_3,
        "DemParRed_0.04": dem_par_red_wrapper_4,
        "DemParRed_0.05": dem_par_red_wrapper_5,
        "DemParRed_0.06": dem_par_red_wrapper_6,
        "DemParRed_0.07": dem_par_red_wrapper_7,
        "DemParRed_0.08": dem_par_red_wrapper_8,
        "DemParRed_0.09": dem_par_red_wrapper_9,
        "DemParRed_0.1": dem_par_red_wrapper_10,
        "EqOppRed": eq_opp_red_wrapper,
        "EqOddsRed": eq_odds_red_wrapper,
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
    results : list<list>
        The results.

    Returns
    -------
    exp_df : pandas.DataFrame
        A dataframe with relevant weight, feat exp, and error columns for the
        best learned policy. Each row is produced by the `new_trial_result()`
        method.
    """
    exp_df_cols = []

    for obj in feat_obj_set.objectives:
        exp_df_cols.append(f"muE_dataset_{obj.name}_mean")
        exp_df_cols.append(f"muE_dataset_{obj.name}_std")

    for obj in feat_obj_set.objectives:
        exp_df_cols.append(f"muE_{obj.name}_mean")
        exp_df_cols.append(f"muE_{obj.name}_std")

    for obj in feat_obj_set.objectives:
        exp_df_cols.append(f"muE_val_{obj.name}_mean")
        exp_df_cols.append(f"muE_val_{obj.name}_std")

    for obj in feat_obj_set.objectives:
        exp_df_cols.append(f"muE_test_{obj.name}_mean")
        exp_df_cols.append(f"muE_test_{obj.name}_std")

    for obj in feat_obj_set.objectives:
        exp_df_cols.append(f"wL_{obj.name}")

    for obj in feat_obj_set.objectives:
        exp_df_cols.append(f"muL_{obj.name}")
        exp_df_cols.append(f"muL_val_{obj.name}")
        exp_df_cols.append(f"muL_test_{obj.name}")

    for obj in feat_obj_set.objectives:
        exp_df_cols.append(f"muL_err_{obj.name}")
        exp_df_cols.append(f"muL_val_err_{obj.name}")
        exp_df_cols.append(f"muL_test_err_{obj.name}")

    # Subdominance (Train, Val, Test)
    exp_df_cols.append("max_abs_subdominance")
    exp_df_cols.append("sum_abs_subdominance")
    exp_df_cols.append("max_rel_subdominance")
    exp_df_cols.append("sum_rel_subdominance")

    exp_df_cols.append("max_abs_subdominance_val")
    exp_df_cols.append("sum_abs_subdominance_val")
    exp_df_cols.append("max_rel_subdominance_val")
    exp_df_cols.append("sum_rel_subdominance_val")

    exp_df_cols.append("max_abs_subdominance_test")
    exp_df_cols.append("sum_abs_subdominance_test")
    exp_df_cols.append("max_rel_subdominance_test")
    exp_df_cols.append("sum_rel_subdominance_test")

    exp_df_cols.append("muL_err_l2norm")
    exp_df_cols.append("muL_val_err_l2norm")
    exp_df_cols.append("muL_test_err_l2norm")

    # Training Errors (Train, Val, Test) for best iteration
    exp_df_cols.append("best_t_train")
    exp_df_cols.append("best_t_val")
    exp_df_cols.append("best_t_test")

    for obj in perf_obj_set.objectives:
        exp_df_cols.append(f"muE_perf_{obj.name}")
        exp_df_cols.append(f"muE_perf_val_{obj.name}")
        exp_df_cols.append(f"muE_perf_test_{obj.name}")
        exp_df_cols.append(f"muL_perf_{obj.name}")
        exp_df_cols.append(f"muL_perf_val_{obj.name}")
        exp_df_cols.append(f"muL_perf_test_{obj.name}")

    exp_df_cols.append(f"trial_runtime")
    exp_df_cols.append(f"avg_runtime_per_irl_loop")
    exp_df_cols.append(f"trial_inputsize")

    exp_df = pd.DataFrame(results, columns=exp_df_cols)

    return exp_df


def new_trial_result(
    best_row_idx,
    feat_obj_set,
    perf_obj_set,
    muE_dataset,
    muE,
    muE_val,
    muE_test,
    muE_perf,
    muE_perf_val,
    muE_perf_test,
    df_irl,
    trial_runtime,
    avg_runtime_per_irl_loop,
    trial_inputsize,
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
    muE_val : array-like<float>, shape(n_expert_demos, n_objectives)
        The feature expectations of the expert demonstrations on the validation
        set.
    muE_test : array-like<float>, shape(n_expert_demos, n_objectives)
        The feature expectations of the expert demonstrations on the test set.
    muE_perf_val : pandas.DataFrame
        Performance measures of expert validation demos.
    muE_perf_test : pandas.DataFrame
        Performance measures of expert test demos.
    df_irl : pandas.DataFrame
        A collection of results where each row represents either an expert demo
        (and therefore a positive SVM training example) or a learned policy
        (and therefore a negative SVM training example). Includes relevant
        items like learned weights, feature expectations, error, etc.
    source_trial_runtime : float
        The runtime to complete the source trial, in seconds.
    source_avg_runtime_per_irl_loop : float
        The runtime to complete the source IRL loop, in seconds.
    source_trial_inputsize : int
        The size of the input space. Specifically `mdp.n_states_` (includes y)

    Returns
    -------
    result : list<list<numeric>>
        The new result row.
    """
    result = []

    # Do this for feature expectation obj set
    for i, obj in enumerate(feat_obj_set.objectives):
        muE_dataset_mean = np.mean(muE_dataset[:, i])
        muE_dataset_std = np.std(muE_dataset[:, i])
        result += [muE_dataset_mean, muE_dataset_std]

    for i, obj in enumerate(feat_obj_set.objectives):
        muE_mean = np.mean(muE[:, i])
        muE_std = np.std(muE[:, i])
        result += [muE_mean, muE_std]

    for i, obj in enumerate(feat_obj_set.objectives):
        muE_val_mean = np.mean(muE_val[:, i])
        muE_val_std = np.std(muE_val[:, i])
        result += [muE_val_mean, muE_val_std]

    for i, obj in enumerate(feat_obj_set.objectives):
        muE_test_mean = np.mean(muE_test[:, i])
        muE_test_std = np.std(muE_test[:, i])
        result += [muE_test_mean, muE_test_std]

    best_row = df_irl.loc[best_row_idx]

    for i, obj in enumerate(feat_obj_set.objectives):
        result.append(best_row[f"{obj.name}_weight"])

    for obj in feat_obj_set.objectives:
        result.append(best_row[f"muL_{obj.name}"])
        result.append(best_row[f"muL_val_{obj.name}"])
        result.append(best_row[f"muL_test_{obj.name}"])

    for i, obj in enumerate(feat_obj_set.objectives):
        _muL_err = abs(best_row[f"muL_{obj.name}"] - np.mean(muE_val[:, i]))
        _muL_val_err = abs(best_row[f"muL_val_{obj.name}"] - np.mean(muE_val[:, i]))
        _muL_test_err = abs(best_row[f"muL_test_{obj.name}"] - np.mean(muE_test[:, i]))
        result.append(_muL_err)
        result.append(_muL_val_err)
        result.append(_muL_test_err)

    # Subdominance (Train, Val, Test)
    result.append(best_row["max_abs_subdominance"])
    result.append(best_row["sum_abs_subdominance"])
    result.append(best_row["max_rel_subdominance"])
    result.append(best_row["sum_rel_subdominance"])

    result.append(best_row["max_abs_subdominance_val"])
    result.append(best_row["sum_abs_subdominance_val"])
    result.append(best_row["max_rel_subdominance_val"])
    result.append(best_row["sum_rel_subdominance_val"])

    result.append(best_row["max_abs_subdominance_test"])
    result.append(best_row["sum_abs_subdominance_test"])
    result.append(best_row["max_rel_subdominance_test"])
    result.append(best_row["sum_rel_subdominance_test"])

    result.append(best_row["mu_delta_l2norm"])
    result.append(best_row["mu_delta_l2norm_val"])
    result.append(best_row["mu_delta_l2norm_test"])

    # Training Errors (Train, Val, Test)
    result.append(best_row["t"])
    result.append(best_row["t_val"])
    result.append(best_row["t_test"])

    # Append performance measures of expert_val, expert_test and best learned policies
    for i, obj in enumerate(perf_obj_set.objectives):
        perf_mean = np.mean(muE_perf[:, i])
        perf_val_mean = np.mean(muE_perf_val[:, i])
        perf_test_mean = np.mean(muE_perf_test[:, i])
        result.append(perf_mean)
        result.append(perf_val_mean)
        result.append(perf_test_mean)
        result.append(best_row[f"muL_perf_{obj.name}"])
        result.append(best_row[f"muL_perf_val_{obj.name}"])
        result.append(best_row[f"muL_perf_test_{obj.name}"])

    result.append(trial_runtime)
    result.append(avg_runtime_per_irl_loop)
    result.append(trial_inputsize)

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
    exp_info,
    alphas,
    raw_demos_feat_loss,
    clf_demos_feat_loss,
    relative=True,
    sum_agg=True,
):
    # Computation based on https://proceedings.mlr.press/v162/ziebart22a/ziebart22a.pdf eq. 5-8 and def. 5
    beta = 1.0
    num_perf_metrics = len(exp_info["SUBDOMINANCE_PERF_METRICS_LIST"])
    num_fair_metrics = len(exp_info["SUBDOMINANCE_FAIR_METRICS_LIST"])
    perf_metric_weight = 0.5 / num_perf_metrics
    fair_metric_weight = 0.5 / num_fair_metrics
    # compute the average across subdom for each demo
    total_subdom_list = []
    for demo_idx in range(raw_demos_feat_loss.shape[0]):
        subdom_list = []
        for feature_idx in range(raw_demos_feat_loss.shape[1]):
            alpha = alphas[feature_idx]
            raw_demo_loss = raw_demos_feat_loss[demo_idx][feature_idx]
            clf_demo_loss = clf_demos_feat_loss[demo_idx][feature_idx]
            if relative:
                epsilon = 1e-10  # to avoid divide by zero
                if raw_demo_loss < epsilon:
                    raw_demo_loss = epsilon
                subdom = np.maximum(
                    (alpha * ((clf_demo_loss / raw_demo_loss) - 1)) + beta, 0
                )
            else:
                subdom = np.maximum((alpha * (clf_demo_loss - raw_demo_loss)) + beta, 0)
            subdom_list.append(subdom)

        if sum_agg:
            # CUSTOM MODIFICATION FOR OUR USE CASE:
            # If we are aggregating by sum, reweight the performance measure (accuracy) subdominance
            # and the fairness measure subdominance equally
            for i, subdom in enumerate(subdom_list):
                if i < num_perf_metrics:
                    subdom_list[i] = subdom * perf_metric_weight
                else:
                    subdom_list[i] = subdom * fair_metric_weight
            total_subdom = sum(subdom_list)
        else:
            total_subdom = max(subdom_list)
        total_subdom_list.append(total_subdom)

    final_subdom = sum(total_subdom_list) / len(total_subdom_list)
    # Since we are using subdominance as an evaluation metric, we don't need to add regularization.
    # final_subdom += (lamda / 2.0) * (np.linalg.norm(alphas) ** 2)
    return final_subdom


# Helper to compute subdominance for a specific set
def compute_iteration_subdominance(
    exp_info,
    raw_demo_ref,
    clf_demo_cur,
):
    # Compute subdominance metric for the learned policy
    raw_demo = raw_demo_ref.copy()
    clf_demo = clf_demo_cur.copy()
    # raw_demo = raw_demo.sample(frac=1).reset_index(drop=True)
    # clf_demo = clf_demo.sample(frac=1).reset_index(drop=True)
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
    max_abs_subdom = compute_subdominance(
        exp_info,
        alphas,
        raw_demos_feat_loss,
        clf_demos_feat_loss,
        relative=False,
        sum_agg=False,
    )
    # Compute sum-aggregated absolute subdominance:
    sum_abs_subdom = compute_subdominance(
        exp_info,
        alphas,
        raw_demos_feat_loss,
        clf_demos_feat_loss,
        relative=False,
        sum_agg=True,
    )
    # Compute max-aggregated relative subdominance:
    max_rel_subdom = compute_subdominance(
        exp_info,
        alphas,
        raw_demos_feat_loss,
        clf_demos_feat_loss,
        relative=True,
        sum_agg=False,
    )
    # Compute sum-aggregated relative subdominance:
    sum_rel_subdom = compute_subdominance(
        exp_info,
        alphas,
        raw_demos_feat_loss,
        clf_demos_feat_loss,
        relative=True,
        sum_agg=True,
    )
    return max_abs_subdom, sum_abs_subdom, max_rel_subdom, sum_rel_subdom


def run_trial_source_domain(
    exp_info,
    X=None,
    y=None,
    feature_types=None,
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
    muE_val : array-like<float>. Shape(n_expert_demos, n_objectives)
        Expert feature expectations on the validation set.
    muE_test : array-like<float>. Shape(n_expert_demos, n_objectives)
        Expert feature expectations on the test set.
    muE_perf_val : pandas.DataFrame
        Results of the performance measure during the expert's validation demos.
    muE_perf_test : pandas.DataFrame
        Results of the performance measure during the expert's test demos.
    df_irl : pandas.DataFrame
        A collection of results where each row represents either an expert demo
        (and therefore a positive SVM training example) or a learned policy
        (and therefore a negative SVM training example). Includes relevant
        items like learned weights, feature expectations, error, etc.
    weights : array-like<float>. Shape(n_irl_loop_iters,  n_objectives).
        The learned weights for each iteration of the IRL loop.
    t_val : array-like<float>. Shape(n_irl_loop_iters)
        The irl error on the validation set.
    t_test : array-like<float>. Shape(n_irl_loop_iters)
        The irl error on the test set.
    clf_pol : research.rl.env.clf_mdp.ClassificationMDPPolicy
        The classification MDP optimal policy.
    avg_runtime_per_irl_loop : float
        The runtime of the IRL loop, in seconds.

    """
    weight_adjusts_list = exp_info["WEIGHT_ADJUSTS_LIST"]
    n_init_policies = len(exp_info["NON_EXPERT_ALGOS"])

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
    #     2. Val – computes the unbiased values for muL and t (dataset is
    #        never shown to the IRL learning algo.)
    #     3. Test – holdout set for final evaluation (never shown to IRL algo)
    #
    # But first check that split doesn't have too big of a difference between demos and
    # holdout, otherwise it messes up interpretations.
    #
    split_is_okay = False
    max_muE_cosine_dist_split = 0.005

    # Hold variables for later use in IRL loop
    demoE_val = None
    demoE_test = None

    while not split_is_okay:
        # 1. First split: Split into 60/40 train and val_test
        X_train, X_val_test, y_train, y_val_test = train_test_split(
            X, y, train_size=0.60
        )

        # 2. Second split: Split the training_val set 50/50 between val and test
        # (In total, this results in 60% for train, 20% for val, and 20% for test)
        X_val, X_test, y_val, y_test = train_test_split(
            X_val_test, y_val_test, test_size=0.50
        )

        # Generate expert demonstrations to learn from
        muE, demoE = generate_demos_k_folds(
            exp_info,
            X=X_train,
            y=y_train,
            clf=copy.deepcopy(expert_algo_lookup[exp_info["EXPERT_ALGO"]]),
            obj_set=feat_obj_set,
            n_demos=exp_info["N_EXPERT_DEMOS"],
            bias_types=bias_types,
        )
        muE_perf = np.array([perf_obj_set.compute_demo_feature_exp(demoE[0])])

        # Expert validation set computations
        expert_val_clf = copy.deepcopy(expert_algo_lookup[exp_info["EXPERT_ALGO"]])
        expert_val_clf.fit(X_val, y_val)
        expert_val_clf = add_classifier_bias(expert_val_clf, bias_types)
        demoE_val = generate_demo(
            clf=expert_val_clf,
            X_test=X_val,
            y_test=y_val,
            can_observe_y=CAN_OBSERVE_Y,
        )
        demoE_val = add_demo_bias(
            demoE_val, unfairness_types=bias_types, dataset=exp_info["DATASET"]
        )
        muE_val = np.array([feat_obj_set.compute_demo_feature_exp(demoE_val)])
        muE_perf_val = np.array([perf_obj_set.compute_demo_feature_exp(demoE_val)])

        # Expert test set computations
        expert_test_clf = copy.deepcopy(expert_algo_lookup[exp_info["EXPERT_ALGO"]])
        expert_test_clf.fit(X_test, y_test)
        expert_test_clf = add_classifier_bias(expert_test_clf, bias_types)
        demoE_test = generate_demo(
            clf=expert_test_clf,
            X_test=X_test,
            y_test=y_test,
            can_observe_y=CAN_OBSERVE_Y,
        )
        demoE_test = add_demo_bias(
            demoE_test, unfairness_types=bias_types, dataset=exp_info["DATASET"]
        )
        muE_test = np.array([feat_obj_set.compute_demo_feature_exp(demoE_test)])
        muE_perf_test = np.array([perf_obj_set.compute_demo_feature_exp(demoE_test)])

        split_is_okay = True
        for demo_i in range(len(muE)):
            if cosine(muE[demo_i], muE_val[0]) > max_muE_cosine_dist_split:
                split_is_okay = False
                logging.info(
                    f"INFO: muE_val Split check failed: {cosine(muE[demo_i], muE_val[0])} > {max_muE_cosine_dist_split}, retrying split..."
                )
                break
            if cosine(muE[demo_i], muE_test[0]) > max_muE_cosine_dist_split:
                split_is_okay = False
                logging.info(
                    f"INFO: muE_test Split check failed: {cosine(muE[demo_i], muE_test[0])} > {max_muE_cosine_dist_split}, retrying split..."
                )
                break

    muE_dataset, _ = generate_demos_k_folds(
        exp_info,
        X=X,
        y=y,
        clf=copy.deepcopy(expert_algo_lookup[exp_info["EXPERT_ALGO"]]),
        obj_set=feat_obj_set,
        n_demos=exp_info["N_EXPERT_DEMOS"],
        bias_types=[],
    )
    # For subdominance calculations later
    expert_train_clf = copy.deepcopy(expert_algo_lookup[exp_info["EXPERT_ALGO"]])
    expert_train_clf.fit(X_train, y_train)
    expert_train_clf = add_classifier_bias(expert_train_clf, bias_types)
    demoE_train = generate_demo(expert_train_clf, X_train, y_train)
    demoE_train = add_demo_bias(
        demoE_train, unfairness_types=bias_types, dataset=exp_info["DATASET"]
    )

    logging.info(f"muE_dataset:\n{muE_dataset}")
    logging.info(f"muE:\n{muE}")
    logging.info(f"muE_val:\n{muE_val}")
    logging.info(f"muE_test:\n{muE_test}")
    logging.info(f"muE_perf:\n{muE_perf}")
    logging.info(f"muE_perf_val:\n{muE_perf_val}")
    logging.info(f"muE_perf_test:\n{muE_perf_test}")

    ##
    # Run IRL loop.
    # Create a clf dataset where inputs are feature expectations and outputs
    # are whether the policy is expert or learned through IRL iterations. Then
    # train an SVM classifier on this dataset. Then extract the weights of the
    # svm and use them as the weights for the "reward" function. Then use this
    # reward function to learn a policy (classifier). Then compute the feature
    # expectations from this classifer on the irl validation set. Then compute
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

    # Added for Val and Test subdominance
    max_abs_subdom_val_hist = []
    sum_abs_subdom_val_hist = []
    max_rel_subdom_val_hist = []
    sum_rel_subdom_val_hist = []

    max_abs_subdom_test_hist = []
    sum_abs_subdom_test_hist = []
    max_rel_subdom_test_hist = []
    sum_rel_subdom_test_hist = []

    t = []  # Errors for each iteration
    t_val = []  # Errors on validation set for each iteration
    t_test = []  # Errors on test set for each iteration
    muL_delta_l2norm_hist = []
    muL_delta_l2norm_val_hist = []
    muL_delta_l2norm_test_hist = []
    weights = []
    i = 0
    demo_history = []
    demo_val_history = []
    demo_test_history = []
    muL_history = []
    muL_val_history = []
    muL_test_history = []
    muL_perf_history = []
    muL_perf_val_history = []
    muL_perf_test_history = []

    # Generate initial learned policies to serve as negative training examples
    # for the SVM IRL classifier.
    muL = []
    for non_expert_algo in exp_info["NON_EXPERT_ALGOS"]:
        # Training set
        _muL, _demos = generate_demos_k_folds(
            exp_info,
            X=X_train,
            y=y_train,
            clf=copy.deepcopy(expert_algo_lookup[non_expert_algo]),
            obj_set=feat_obj_set,
            n_demos=1,
            bias_types=bias_types,
        )
        muL.append(_muL[0])

        # Validation set
        _muL_val, _demos_val = generate_demos_k_folds(
            exp_info,
            X=X_val,
            y=y_val,
            clf=copy.deepcopy(expert_algo_lookup[non_expert_algo]),
            obj_set=feat_obj_set,
            n_demos=1,
            bias_types=bias_types,
        )
        # muL_deltas_val = _muL_val - muE_val
        # muL_deltas_val_l2norm = np.linalg.norm(muL_deltas_val, ord=2)
        # muL_delta_l2norm_val_hist.append(muL_deltas_val_l2norm)

        # Test set
        _muL_test, _demos_test = generate_demos_k_folds(
            exp_info,
            X=X_test,
            y=y_test,
            clf=copy.deepcopy(expert_algo_lookup[non_expert_algo]),
            obj_set=feat_obj_set,
            n_demos=1,
            bias_types=bias_types,
        )
        # muL_deltas_test = _muL_test - muE_test
        # muL_deltas_test_l2norm = np.linalg.norm(muL_deltas_test, ord=2)
        # muL_delta_l2norm_test_hist.append(muL_deltas_test_l2norm)

    # muL_delta_l2norm_val_hist_backup = muL_delta_l2norm_val_hist.copy()
    # muL_delta_l2norm_test_hist_backup = muL_delta_l2norm_test_hist.copy()

    muL = np.array(muL)
    logging.info(f"muL:\n{muL}")
    X_irl_exp = pd.DataFrame(muE, columns=feat_obj_set_cols)
    y_irl_exp = pd.Series(np.ones(exp_info["N_EXPERT_DEMOS"]), dtype=int)
    X_irl_learn = pd.DataFrame(muL, columns=feat_obj_set_cols)
    y_irl_learn = pd.Series(np.zeros(len(muL)), dtype=int)

    # Start the IRL Loop
    logging.debug("")
    logging.debug("Starting IRL Loop ...")

    done = False  # When true, breaks IRL loop
    is_stuck_in_loop = False
    num_loops = 0
    return_vals = []
    all_irl_loop_start = datetime.datetime.now()
    for weight_adjusts in weight_adjusts_list:
        while not done:
            this_irl_loop_start = datetime.datetime.now()
            num_loops += 1
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
                    svm = SVM(positive_weights_only=allow_pos_weights).fit(X_irl, y_irl)
                except ValueError as e:
                    if e.args[0] != "No support vectors found.":
                        raise (e)
                    logging.info("\t\tAllowing negative weights.")
                    svm = SVM(positive_weights_only=False).fit(X_irl, y_irl)

                wi = svm.weights(norm="l1")

                ##
                # Learn a policy (clf_pol) from the reward (SVM) weights.
                ##

                # Fit a classifier that predicts `y` from `X`.
                logging.debug("\tFitting `y|x` predictor for clf policy...")
                clf = sklearn_clf_pipeline(
                    feature_types=irl_loop_feature_types,
                    clf_inst=RandomForestClassifier(),
                )
                clf.fit(X_train, y_train)

                demo_df = pd.DataFrame(X_train)
                demo_df["y"] = y_train
            else:
                # Reset IRL loop history vars
                max_abs_subdom_hist = []
                sum_abs_subdom_hist = []
                max_rel_subdom_hist = []
                sum_rel_subdom_hist = []

                max_abs_subdom_val_hist = []
                sum_abs_subdom_val_hist = []
                max_rel_subdom_val_hist = []
                sum_rel_subdom_val_hist = []

                max_abs_subdom_test_hist = []
                sum_abs_subdom_test_hist = []
                max_rel_subdom_test_hist = []
                sum_rel_subdom_test_hist = []

                t = []  # Errors for each iteration
                t_val = []  # Errors on validation set for each iteration
                t_test = []  # Errors on test set for each iteration
                muL_delta_l2norm_hist = []
                # muL_delta_l2norm_val_hist = muL_delta_l2norm_val_hist_backup.copy()
                # muL_delta_l2norm_test_hist = muL_delta_l2norm_test_hist_backup.copy()
                muL_delta_l2norm_val_hist = []
                muL_delta_l2norm_test_hist = []
                weights = []
                i = 0
                demo_history = []
                demo_val_history = []
                demo_test_history = []
                muL_history = []
                muL_val_history = []
                muL_test_history = []
                muL_perf_history = []
                muL_perf_val_history = []
                muL_perf_test_history = []
                X_irl_exp = pd.DataFrame(muE, columns=feat_obj_set_cols)
                y_irl_exp = pd.Series(np.ones(exp_info["N_EXPERT_DEMOS"]), dtype=int)
                X_irl_learn = pd.DataFrame(muL, columns=feat_obj_set_cols)
                y_irl_learn = pd.Series(np.zeros(len(muL)), dtype=int)
                wi = unadjusted_best_weight.copy()
                for weight_adjust in weight_adjusts:
                    mul_factor = weight_adjust[1]
                    if weight_adjust[0] == "mul_negative_weights":
                        wi[wi < 0] = wi[wi < 0] * mul_factor
                done = True
            # Learn a policy that maximizes the reward function.
            weights.append(wi)

            reward_weights = {
                obj.name: wi[j] for j, obj in enumerate(feat_obj_set.objectives)
            }
            clf_pol = compute_optimal_policy(
                clf_df=demo_df,
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
            demo = generate_demo(clf_pol, X_train, y_train, can_observe_y=CAN_OBSERVE_Y)
            demo_val = generate_demo(clf_pol, X_val, y_val, can_observe_y=False)
            demo_test = generate_demo(clf_pol, X_test, y_test, can_observe_y=False)
            demo_history.append(demo)
            demo_val_history.append(demo_val)
            demo_test_history.append(demo_test)
            muL_ = feat_obj_set.compute_demo_feature_exp(demo)
            muL_val = feat_obj_set.compute_demo_feature_exp(demo_val)
            muL_test = feat_obj_set.compute_demo_feature_exp(demo_test)
            muL_perf = perf_obj_set.compute_demo_feature_exp(demo)
            muL_perf_val = perf_obj_set.compute_demo_feature_exp(demo_val)
            muL_perf_test = perf_obj_set.compute_demo_feature_exp(demo_test)
            muL_history.append(muL_)
            muL_val_history.append(muL_val)
            muL_test_history.append(muL_test)
            muL_perf_history.append(muL_perf)
            muL_perf_val_history.append(muL_perf_val)
            muL_perf_test_history.append(muL_perf_test)
            logging.info(
                f"\t\t muL[{i}] \t\t= {str(np.round(muL_, 2)).replace('0.', '.')}"
            )
            logging.debug(f"\t\t muL_val[{i}] = {np.round(muL_val, 2)}")
            logging.debug(f"\t\t muL_test[{i}] = {np.round(muL_test, 2)}")

            # Append policy's feature expectations to irl clf dataset
            X_irl_learn_i = pd.DataFrame(np.array([muL_]), columns=feat_obj_set_cols)
            y_irl_learn_i = pd.Series(np.zeros(1), dtype=int)
            X_irl_learn = pd.concat([X_irl_learn, X_irl_learn_i], axis=0)
            y_irl_learn = pd.concat([y_irl_learn, y_irl_learn_i], axis=0)

            # Compute error of the learned policy: t[i] = wT(muE-muL[j])
            # This is equivalent to computing the SVM margin.
            # NOTE: Since the svm model does not get retrained when weights are adjusted,
            # the ti values returned here are incorrect when weights are being adjusted.
            # However, this does not practically matter since there is only one iteration
            # when weights are being adjusted. Also they are never actually used for anything.
            # NOTE: the best_j values returned here are sometimes incorrect, instead of returning
            # the index of the lowest error iteration, it just returns the most recent index.
            # However, this also does not practically matter since best_j is only used to select
            # the data to put in the muL_best_ lists, which are only ever used for diagnostics.
            # Honestly, the names muL_"best" and "best"_j are misleading since they are not
            # necessarily referencing the best performing iteration, just the most recent iteration.
            # But I can't be bothered to change the name at this point, so it shall stay for now.
            ti, best_j, muL_delta, muL_delta_l2norm = irl_error(
                wi,
                muE,
                muL_history,
                dot_weights_feat_exp=exp_info["DOT_WEIGHTS_FEAT_EXP"],
                svm_margin=svm.margin(),
            )
            if len(t) >= 1 and ti > t[-1]:  # This is the final IRL loop iteration
                pass
            # Do it for the validation set as well.
            ti_val, best_j_val, muL_delta_val, muL_delta_l2norm_val = irl_error(
                wi,
                muE_val,
                muL_val_history,
                dot_weights_feat_exp=exp_info["DOT_WEIGHTS_FEAT_EXP"],
                svm_margin=svm.margin(),
            )
            # Do it for the test set as well.
            ti_test, best_j_test, muL_delta_test, muL_delta_l2norm_test = irl_error(
                wi,
                muE_test,
                muL_test_history,
                dot_weights_feat_exp=exp_info["DOT_WEIGHTS_FEAT_EXP"],
                svm_margin=svm.margin(),
            )

            # --- Subdominance Calculations ---
            # Train
            (
                max_abs_subdominance,
                sum_abs_subdominance,
                max_rel_subdominance,
                sum_rel_subdominance,
            ) = compute_iteration_subdominance(exp_info, demoE_train, demo)
            max_abs_subdom_hist.append(max_abs_subdominance)
            sum_abs_subdom_hist.append(sum_abs_subdominance)
            max_rel_subdom_hist.append(max_rel_subdominance)
            sum_rel_subdom_hist.append(sum_rel_subdominance)

            # Val
            max_abs_subdom_v, sum_abs_subdom_v, max_rel_subdom_v, sum_rel_subdom_v = (
                compute_iteration_subdominance(exp_info, demoE_val, demo_val)
            )
            max_abs_subdom_val_hist.append(max_abs_subdom_v)
            sum_abs_subdom_val_hist.append(sum_abs_subdom_v)
            max_rel_subdom_val_hist.append(max_rel_subdom_v)
            sum_rel_subdom_val_hist.append(sum_rel_subdom_v)

            # Test
            max_abs_subdom_t, sum_abs_subdom_t, max_rel_subdom_t, sum_rel_subdom_t = (
                compute_iteration_subdominance(exp_info, demoE_test, demo_test)
            )
            max_abs_subdom_test_hist.append(max_abs_subdom_t)
            sum_abs_subdom_test_hist.append(sum_abs_subdom_t)
            max_rel_subdom_test_hist.append(max_rel_subdom_t)
            sum_rel_subdom_test_hist.append(sum_rel_subdom_t)

            # -------------------------------

            t.append(ti)
            t_val.append(ti_val)
            t_test.append(ti_test)
            muL_delta_l2norm_hist.append(muL_delta_l2norm)
            muL_delta_l2norm_val_hist.append(muL_delta_l2norm_val)
            muL_delta_l2norm_test_hist.append(muL_delta_l2norm_test)
            logging.info(
                f"\t\t Best muL_delta[{i}] \t= {str(np.round(muL_delta, 2)).replace('0.', '.')}"
            )
            logging.info(
                f"\t\t Best muL_delta_val[{i}] \t= {str(np.round(muL_delta_val, 2)).replace('0.', '.')}"
            )
            logging.info(
                f"\t\t Best muL_delta_test[{i}] \t= {str(np.round(muL_delta_test, 2)).replace('0.', '.')}"
            )
            logging.info(f"\t\t t[{i}] \t\t= {t[i]:.5f}")
            logging.info(f"\t\t t_val[{i}] \t= {t_val[i]:.5f}")
            logging.info(f"\t\t t_test[{i}] \t= {t_test[i]:.5f}")
            logging.info(
                f"\t\t weights[{i}] \t= {str(np.round(weights[i], 2)).replace('0.', '.')}"
            )

            runtime_this_irl_loop = (
                datetime.datetime.now() - this_irl_loop_start
            ).total_seconds()
            logging.info(f"\t\t Runtime for this IRL loop: {runtime_this_irl_loop}")

            # If reached maximum iterations
            if i >= exp_info["MAX_ITER"] - 1:
                logging.info(f"\nReached max iters.")
                done = True
                break

            # Check if error is below epsilon
            elif ti < exp_info["EPSILON"] and ti <= min(t):
                if np.allclose(weights[i][0], 0, atol=1e-5):
                    logging.info("\t\tAccuracy weight is zero, continuing")
                    i += 1
                    continue

                if exp_info["ALLOW_NEG_WEIGHTS"] or np.all(wi > -1e-5):
                    done = True
                    break

            # If error is going back up, stop
            elif len(t) > 1 and ti > t[-2]:
                logging.info(f"\n\nError is going back up. Stoping.")
                done = True
                break
            # Check if a new best error has been found in last i/2 iterations.
            elif (i - np.argsort(t)[0]) > exp_info["EARLY_STOP_NO_NEW_BEST_ITERS"]:
                logging.info(
                    f"\n\t\tNo new best in last {exp_info['EARLY_STOP_NO_NEW_BEST_ITERS']} iterations, so stopping early."
                )
                done = True
                break
            # Check if loop is stuck in local optimum
            elif (
                i > 0
                and abs(t[i] - t[i - 1]) < 1e-3
                and np.allclose(weights[i], weights[i - 1], atol=1e-3)
            ):
                logging.info("\t\tStuck in loop")
                is_stuck_in_loop = True
                i += 1
            # Check if error is infinite
            elif i > 0 and t[i] == np.inf:
                logging.info("\t\tInfinite error: Treating like stuck in loop")
                is_stuck_in_loop = True
                i += 1
            else:
                i += 1

            # End IRL Loop
        done = False

        avg_runtime_per_irl_loop = (
            datetime.datetime.now() - all_irl_loop_start
        ).total_seconds() / num_loops

        # If solution not sufficient for use, exit early
        if min(muL_delta_l2norm_hist) > exp_info["IGNORE_RESULTS_EPSILON"]:
            logging.info(
                f"IGNORING RESULTS BECAUSE BEST ERROR {min(muL_delta_l2norm_hist):.3f} > {exp_info['IGNORE_RESULTS_EPSILON']:.3f}"
            )
            return_val = (
                None,  # best_row_idx
                None,  # muE_dataset
                None,  # muE
                None,  # muE_val
                None,  # muE_test
                None,  # muE_perf
                None,  # muE_perf_val
                None,  # muE_perf_test
                None,  # df_irl
                None,  # clf_pol
                None,  # avg_runtime_per_irl_loop
            )
            return_vals.append(return_val)
            return return_vals

        # Find best weights based on smallest subdominance error
        # Using the validation set for selection since it is never seen in training
        # and therefore should give the most unbiased estimate of true error.
        # MODIFIED: Using sum_abs_subdominance_val for best_iter selection
        t_val_arg_smallest_to_largest = np.argsort(t_val)
        muL_delta_l2_norm_val_arg_smallest_to_largest = np.argsort(
            muL_delta_l2norm_val_hist
        )
        sum_abs_subdominance_val_arg_smallest_to_largest = np.argsort(
            sum_abs_subdom_val_hist
        )
        # best_iter = muL_delta_l2_norm_val_arg_smallest_to_largest[0]
        # best_iter = t_val_arg_smallest_to_largest[0]
        best_iter = sum_abs_subdominance_val_arg_smallest_to_largest[0]

        if not exp_info["ALLOW_NEG_WEIGHTS"]:
            best_t_val = None
            best_t_val_i = 0
            best_t_val_done = False
            while not best_t_val_done:
                if best_t_val_i >= len(t_val):
                    logging.info(f"best_weight:\t {np.round(weights[best_iter], 3)}")
                    raise ValueError("Only negative weights learned")
                best_iter = t_val_arg_smallest_to_largest[best_t_val_i]
                best_t_val = t_val[best_iter]
                if (
                    np.all(weights[best_iter] > -1e-5)
                    and muL_delta_l2norm_hist[best_iter]
                    <= exp_info["IGNORE_RESULTS_EPSILON"]
                ):
                    best_t_val_done = True
                best_t_val_i += 1

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

        best_row_idx = exp_info["N_EXPERT_DEMOS"] + n_init_policies + best_iter

        # Generate a dataframe for results gathering.
        X_irl = pd.concat([X_irl_exp, X_irl_learn], axis=0).reset_index(drop=True)
        y_irl = pd.concat([y_irl_exp, y_irl_learn], axis=0).reset_index(drop=True)
        df_irl = X_irl.copy()
        df_irl["is_expert"] = y_irl.copy()
        for i, col in enumerate(feat_obj_set_cols):
            df_irl.columns.values[i] = f"muL_{col}"
            # df_irl[f"muL_{col}"] = (
            #     np.zeros(exp_info["N_EXPERT_DEMOS"] + n_init_policies).tolist()
            #     + np.array(muL_history)[:, i].tolist()
            # )
            df_irl[f"muL_val_{col}"] = (
                np.zeros(exp_info["N_EXPERT_DEMOS"] + n_init_policies).tolist()
                + np.array(muL_val_history)[:, i].tolist()
            )
            df_irl[f"muL_test_{col}"] = (
                np.zeros(exp_info["N_EXPERT_DEMOS"] + n_init_policies).tolist()
                + np.array(muL_test_history)[:, i].tolist()
            )
        for i, col in enumerate(perf_obj_set_cols):
            df_irl[f"muL_perf_{col}"] = (
                np.zeros(exp_info["N_EXPERT_DEMOS"] + n_init_policies).tolist()
                + np.array(muL_perf_history)[:, i].tolist()
            )
            df_irl[f"muL_perf_val_{col}"] = (
                np.zeros(exp_info["N_EXPERT_DEMOS"] + n_init_policies).tolist()
                + np.array(muL_perf_val_history)[:, i].tolist()
            )
            df_irl[f"muL_perf_test_{col}"] = (
                np.zeros(exp_info["N_EXPERT_DEMOS"] + n_init_policies).tolist()
                + np.array(muL_perf_test_history)[:, i].tolist()
            )
        df_irl["is_init_policy"] = (
            np.zeros(exp_info["N_EXPERT_DEMOS"]).tolist()
            + np.ones(n_init_policies).tolist()
            + np.zeros(len(t)).tolist()
        )
        df_irl["learn_idx"] = list(-1 * np.ones(exp_info["N_EXPERT_DEMOS"])) + list(
            np.arange(n_init_policies + len(t))
        )
        for i, col in enumerate(feat_obj_set_cols):
            df_irl[f"{col}_weight"] = np.zeros(
                exp_info["N_EXPERT_DEMOS"] + n_init_policies
            ).tolist() + [w[i] for w in weights]
        df_irl["max_abs_subdominance"] = (
            np.inf * np.ones(exp_info["N_EXPERT_DEMOS"] + n_init_policies, dtype=float)
        ).tolist() + max_abs_subdom_hist
        df_irl["sum_abs_subdominance"] = (
            np.inf * np.ones(exp_info["N_EXPERT_DEMOS"] + n_init_policies, dtype=float)
        ).tolist() + sum_abs_subdom_hist
        df_irl["max_rel_subdominance"] = (
            np.inf * np.ones(exp_info["N_EXPERT_DEMOS"] + n_init_policies, dtype=float)
        ).tolist() + max_rel_subdom_hist
        df_irl["sum_rel_subdominance"] = (
            np.inf * np.ones(exp_info["N_EXPERT_DEMOS"] + n_init_policies, dtype=float)
        ).tolist() + sum_rel_subdom_hist

        # Added Subdominance for Val and Test
        df_irl["max_abs_subdominance_val"] = (
            np.inf * np.ones(exp_info["N_EXPERT_DEMOS"] + n_init_policies, dtype=float)
        ).tolist() + max_abs_subdom_val_hist
        df_irl["sum_abs_subdominance_val"] = (
            np.inf * np.ones(exp_info["N_EXPERT_DEMOS"] + n_init_policies, dtype=float)
        ).tolist() + sum_abs_subdom_val_hist
        df_irl["max_rel_subdominance_val"] = (
            np.inf * np.ones(exp_info["N_EXPERT_DEMOS"] + n_init_policies, dtype=float)
        ).tolist() + max_rel_subdom_val_hist
        df_irl["sum_rel_subdominance_val"] = (
            np.inf * np.ones(exp_info["N_EXPERT_DEMOS"] + n_init_policies, dtype=float)
        ).tolist() + sum_rel_subdom_val_hist

        df_irl["max_abs_subdominance_test"] = (
            np.inf * np.ones(exp_info["N_EXPERT_DEMOS"] + n_init_policies, dtype=float)
        ).tolist() + max_abs_subdom_test_hist
        df_irl["sum_abs_subdominance_test"] = (
            np.inf * np.ones(exp_info["N_EXPERT_DEMOS"] + n_init_policies, dtype=float)
        ).tolist() + sum_abs_subdom_test_hist
        df_irl["max_rel_subdominance_test"] = (
            np.inf * np.ones(exp_info["N_EXPERT_DEMOS"] + n_init_policies, dtype=float)
        ).tolist() + max_rel_subdom_test_hist
        df_irl["sum_rel_subdominance_test"] = (
            np.inf * np.ones(exp_info["N_EXPERT_DEMOS"] + n_init_policies, dtype=float)
        ).tolist() + sum_rel_subdom_test_hist

        df_irl["t"] = (
            list(np.inf * (np.ones(exp_info["N_EXPERT_DEMOS"] + n_init_policies))) + t
        )
        df_irl["t_val"] = (
            list(np.inf * (np.ones(exp_info["N_EXPERT_DEMOS"] + n_init_policies)))
            + t_val
        )
        df_irl["t_test"] = (
            list(np.inf * (np.ones(exp_info["N_EXPERT_DEMOS"] + n_init_policies)))
            + t_test
        )
        df_irl["mu_delta_l2norm"] = (
            np.inf * np.ones(exp_info["N_EXPERT_DEMOS"] + n_init_policies, dtype=float)
        ).tolist() + muL_delta_l2norm_hist
        df_irl["mu_delta_l2norm_val"] = (
            np.inf * np.ones(exp_info["N_EXPERT_DEMOS"] + n_init_policies, dtype=float)
        ).tolist() + muL_delta_l2norm_val_hist
        df_irl["mu_delta_l2norm_test"] = (
            np.inf * np.ones(exp_info["N_EXPERT_DEMOS"] + n_init_policies, dtype=float)
        ).tolist() + muL_delta_l2norm_test_hist

        logging.debug("Experiment Summary")

        # End regular IRL trial
        return_val = (
            best_row_idx,
            muE_dataset,
            muE,
            muE_val,
            muE_test,
            muE_perf,
            muE_perf_val,
            muE_perf_test,
            df_irl,
            clf_pol,
            avg_runtime_per_irl_loop,
        )
        return_vals.append(return_val)
    return return_vals


def compute_relevant_feat_loss(exp_info, demo):
    objectives = []
    for obj_name in exp_info["SUBDOMINANCE_PERF_METRICS_LIST"]:
        objectives.append(OBJ_LOOKUP_BY_NAME[obj_name]())
    for obj_name in exp_info["SUBDOMINANCE_FAIR_METRICS_LIST"]:
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

        trial_start = datetime.datetime.now()

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
                best_row_idx,
                muE_dataset,
                muE,
                muE_val,
                muE_test,
                muE_perf,
                muE_perf_val,
                muE_perf_test,
                df_irl,
                clf_pol,
                avg_runtime_per_irl_loop,
            ) = return_val

            trial_runtime = (datetime.datetime.now() - trial_start).total_seconds()

            trial_inputsize = None
            if hasattr(clf_pol, "mdp"):
                trial_inputsize = clf_pol.mdp.n_states_

            if best_row_idx is None:
                logging.info(f"\nTrial {trial_i} did not converge.\n")
                trial_i += 1
                continue

            # Aggregate trial results
            _result = new_trial_result(
                best_row_idx,
                feat_obj_set,
                perf_obj_set,
                muE_dataset,
                muE,
                muE_val,
                muE_test,
                muE_perf,
                muE_perf_val,
                muE_perf_test,
                df_irl,
                trial_runtime,
                avg_runtime_per_irl_loop,
                trial_inputsize,
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
