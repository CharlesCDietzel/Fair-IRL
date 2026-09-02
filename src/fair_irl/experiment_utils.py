import copy
import datetime
import logging
import os
import uuid
from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd
import sklearn.base
import optuna
import wandb
import pybobyqa
import nevergrad as ng
from catboost import CatBoostClassifier
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equal_opportunity_difference,
    equalized_odds_difference,
    false_negative_rate,
    false_positive_rate,
    selection_rate,
    true_negative_rate,
    true_positive_rate,
)
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import (
    BoundedGroupLoss,
    DemographicParity,
    EqualizedOdds,
    ExponentiatedGradient,
    TruePositiveRateParity,
    ZeroOneLoss,
)
from line_profiler import LineProfiler
from scipy.spatial.distance import cosine
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.preprocessing import normalize

from fair_irl.irl.fair_irl import *
from fair_irl.rl.clf_mdp import *
from fair_irl.rl.clf_mdp_policy import *
from fair_irl.rl.objectives import *
from fair_irl.utils import *

from .datasets import *

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


def _negative_predictive_value(y_true, y_pred):
    """
    P(y=0 | yhat=0) = TN / (TN + FN). Not available in fairlearn/sklearn, so
    it's implemented manually to be used as a `MetricFrame` metric function.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    denom = tn + fn
    return tn / denom if denom > 0 else np.nan


def _group_rate_loss(metric_fn, z_value):
    """
    Builds a `demo`-loss function for a metric evaluated on a single
    sensitive-feature group (e.g. TPR_Z0), matching the `-mu + 1` inversion
    used by `compute_relevant_feat_loss()`.
    """

    def _loss(demo):
        mf = MetricFrame(
            metrics=metric_fn,
            y_true=demo["y"],
            y_pred=demo["yhat"],
            sensitive_features=demo["z"],
        )
        return 1 - mf.by_group[z_value]

    return _loss


def _pairwise_metric_diff_loss(metric_fn):
    """
    Builds a `demo`-loss function for the absolute between-group difference
    of a metric (e.g. Predictive Parity), matching the `-mu + 1` inversion
    used by `compute_relevant_feat_loss()`.
    """

    def _loss(demo):
        mf = MetricFrame(
            metrics=metric_fn,
            y_true=demo["y"],
            y_pred=demo["yhat"],
            sensitive_features=demo["z"],
        )
        return mf.difference(method="between_groups")

    return _loss


# Fairlearn/sklearn-based equivalent of OBJ_LOOKUP_BY_NAME: maps each obj_name
# to a function of `demo` that reproduces the corresponding entry of
# `demo_feat_exp` as computed in `compute_relevant_feat_loss()`.
FAIRLEARN_OBJ_LOOKUP_BY_NAME = {
    "Acc": lambda demo: 1 - accuracy_score(demo["y"], demo["yhat"]),
    "AccPar": _pairwise_metric_diff_loss(accuracy_score),
    "DemPar": lambda demo: demographic_parity_difference(
        demo["y"], demo["yhat"], sensitive_features=demo["z"]
    ),
    "EqOpp": lambda demo: equal_opportunity_difference(
        demo["y"], demo["yhat"], sensitive_features=demo["z"]
    ),
    "FPRPar": _pairwise_metric_diff_loss(false_positive_rate),
    "EqOdds": lambda demo: equalized_odds_difference(
        demo["y"], demo["yhat"], sensitive_features=demo["z"]
    ),
    "TNRPar": _pairwise_metric_diff_loss(true_negative_rate),
    "FNRPar": _pairwise_metric_diff_loss(false_negative_rate),
    "PredPar": _pairwise_metric_diff_loss(precision_score),
    "NegPredPar": _pairwise_metric_diff_loss(_negative_predictive_value),
    "PR_Z0": _group_rate_loss(selection_rate, 0),
    "PR_Z1": _group_rate_loss(selection_rate, 1),
    "NR_Z0": _group_rate_loss(lambda yt, yp: 1 - selection_rate(yt, yp), 0),
    "NR_Z1": _group_rate_loss(lambda yt, yp: 1 - selection_rate(yt, yp), 1),
    "TPR_Z0": _group_rate_loss(true_positive_rate, 0),
    "TPR_Z1": _group_rate_loss(true_positive_rate, 1),
    "TNR_Z0": _group_rate_loss(true_negative_rate, 0),
    "TNR_Z1": _group_rate_loss(true_negative_rate, 1),
    "FPR_Z0": _group_rate_loss(false_positive_rate, 0),
    "FPR_Z1": _group_rate_loss(false_positive_rate, 1),
    "FNR_Z0": _group_rate_loss(false_negative_rate, 0),
    "FNR_Z1": _group_rate_loss(false_negative_rate, 1),
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
        CatBoostClassifier(allow_writing_files=False, logging_level="Silent"),
    )

    catboost_opt_acc_pipe = sklearn_clf_pipeline(
        feature_types,
        # clf_inst=DecisionTreeClassifier(min_samples_leaf=10, max_depth=4),
        # clf_inst=DecisionTreeClassifier(min_samples_leaf=5, max_depth=10),
        # RandomForestClassifier(),
        CatBoostClassifier(allow_writing_files=False, logging_level="Silent"),
    )

    xgboost_opt_acc_pipe = sklearn_clf_pipeline(
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
            clf_inst=CatBoostClassifier(
                allow_writing_files=False, logging_level="Silent"
            ),
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
            clf_inst=CatBoostClassifier(
                allow_writing_files=False, logging_level="Silent"
            ),
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
            clf_inst=CatBoostClassifier(
                allow_writing_files=False, logging_level="Silent"
            ),
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
            clf_inst=CatBoostClassifier(
                allow_writing_files=False, logging_level="Silent"
            ),
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
            clf_inst=CatBoostClassifier(
                allow_writing_files=False, logging_level="Silent"
            ),
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
            clf_inst=CatBoostClassifier(
                allow_writing_files=False, logging_level="Silent"
            ),
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
            clf_inst=CatBoostClassifier(
                allow_writing_files=False, logging_level="Silent"
            ),
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
            clf_inst=CatBoostClassifier(
                allow_writing_files=False, logging_level="Silent"
            ),
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
            clf_inst=CatBoostClassifier(
                allow_writing_files=False, logging_level="Silent"
            ),
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
            clf_inst=CatBoostClassifier(
                allow_writing_files=False, logging_level="Silent"
            ),
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
            clf_inst=CatBoostClassifier(
                allow_writing_files=False, logging_level="Silent"
            ),
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
            clf_inst=CatBoostClassifier(
                allow_writing_files=False, logging_level="Silent"
            ),
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
            clf_inst=CatBoostClassifier(
                allow_writing_files=False, logging_level="Silent"
            ),
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
            clf_inst=CatBoostClassifier(
                allow_writing_files=False, logging_level="Silent"
            ),
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
            clf_inst=CatBoostClassifier(
                allow_writing_files=False, logging_level="Silent"
            ),
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
            clf_inst=CatBoostClassifier(
                allow_writing_files=False, logging_level="Silent"
            ),
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
            clf_inst=CatBoostClassifier(
                allow_writing_files=False, logging_level="Silent"
            ),
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
            clf_inst=CatBoostClassifier(
                allow_writing_files=False, logging_level="Silent"
            ),
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
            clf_inst=CatBoostClassifier(
                allow_writing_files=False, logging_level="Silent"
            ),
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
            clf_inst=CatBoostClassifier(
                allow_writing_files=False, logging_level="Silent"
            ),
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

    # OptClfMDPPol: optimal classifier policy (see compute_optimal_policy() in
    # fair_irl.py), with reward weights split equally (and positively) across
    # every feature-expectation objective. Its ClassificationMDPPolicy is
    # actually built per-fold in generate_non_overfit_demos() since it depends
    # on the objective set and training fold, neither of which is available
    # here yet.
    opt_clf_mdp_pol_expert = OptClfMDPPolicyExpert(feature_types)

    expert_algo_lookup = {
        # Experts
        "OptAcc": opt_acc_pipe,
        "CatBoostOptAcc": catboost_opt_acc_pipe,
        "XGBoostOptAcc": xgboost_opt_acc_pipe,
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
        "OptClfMDPPol": opt_clf_mdp_pol_expert,
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


# ---------------------------------------------------------------------------
# Weights & Biases experiment tracking
# ---------------------------------------------------------------------------
# Experiment configuration, learned-policy metrics and trial results are all
# reported to W&B rather than written out as CSV/JSON files. The server address and credentials come from the standard
# `wandb` configuration (the `WANDB_BASE_URL` environment variable or
# `~/.config/wandb/settings`), so pointing the experiments at a different
# server needs no code change here.

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "fair-irl")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY") or None

# The 12 subdominance measurements returned, in this order, by
# `_evaluate_policy()` for every learned policy.
SUBDOMINANCE_KEYS = tuple(
    f"{agg}_{kind}_subdominance_{split}"
    for split in ("train", "val", "test")
    for agg, kind in (("max", "abs"), ("sum", "abs"), ("max", "rel"), ("sum", "rel"))
)


def _json_safe(value):
    """
    Convert numpy scalars/arrays and tuples into plain Python types so they can
    be stored in a W&B run config.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return value


def weight_adjusts_name(weight_adjust):
    """
    Build a short, human readable label for a weight adjustment configuration.

    Used as the W&B `job_type` and as part of the run name, so that runs which
    differ only in how the learned weights were adjusted can be grouped and
    compared against each other.

    Parameters
    ----------
    weight_adjusts : tuple
        One entry of `exp_info["WEIGHT_ADJUST_LIST"]`, or `()` for the
        unadjusted weights.

    Returns
    -------
    name : str
        E.g. `"unadjusted"` or `"opt_debias_optuna_CMA-ES"`.
    """
    if not weight_adjust:
        return "unadjusted"

    names = []
    for weight_adjust_component in weight_adjust:
        if isinstance(weight_adjust_component, str):
            names.append(weight_adjust_component)
        else:
            names.append(str(weight_adjust_component))

    return "_".join(names)


def bias_type_name(bias_type):
    """
    Build a short, human readable label for a bias type configuration.

    Used as part of the W&B run name and as a run tag, so that the runs of the
    bias types one trial covers can be told apart from each other.

    Parameters
    ----------
    bias_type : tuple
        One entry of `exp_info["BIAS_TYPE_LIST"]`, or `()` for the unbiased
        demonstrations.

    Returns
    -------
    name : str
        E.g. `"unbiased"` or `"balanced_redlining_0.2"`.
    """
    if not bias_type:
        return "unbiased"

    return "_".join(str(component) for component in bias_type)


def new_session_id():
    """
    Build the identifier shared by every W&B run of one execution of the
    experiment script.

    A single execution spreads its runs across many `run_bias_experiment()`
    calls, each of which gets its own group; this id is what ties them back
    together, so that the plotting notebook can select exactly one execution's
    results instead of mixing several.

    Returns
    -------
    session_id : str
        E.g. `"20260826-010125-3f9ab2"`. The random suffix keeps two executions
        started within the same second from being treated as one session.
    """
    return f"{datetime.datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"


def start_wandb_run(exp_info, bias_type, weight_adjust, trial_i, group, session_id):
    """
    Start the W&B run that records one trial of one bias type and one weight
    adjustment.

    Parameters
    ----------
    exp_info : dict
        Experiment parameters. Logged in full as the run's config, replacing
        the exp_info JSON files this pipeline used to write. It holds the
        listed bias types of the trial as `BIAS_TYPE_LIST`; the single one this
        run covers -- which may be the unbiased `()` that is always run and so
        is not listed there -- is recorded separately as `BIAS_TYPE`.
    bias_type : tuple
        The bias type this run covers; `()` for no bias.
    weight_adjusts : tuple
        The weight adjustment this run covers; `()` for the unadjusted
        weights.
    trial_i : int
        Index of the trial within the experiment.
    group : str
        W&B group shared by every run of the same experiment, so that trials,
        bias types and weight adjustments of one experiment stay together in
        the UI.
    session_id : str
        Identifies the execution of the experiment script this run belongs to.
        Recorded in the config so that the plotting notebook can select one
        execution's runs and never mix results from several.

    Returns
    -------
    run : wandb.sdk.wandb_run.Run
        The started run. The caller is responsible for calling `finish()`.
    """
    bias_name = bias_type_name(bias_type)
    adjust_name = weight_adjusts_name(weight_adjust)

    config = {key: _json_safe(value) for key, value in exp_info.items()}
    config["BIAS_TYPE"] = _json_safe(bias_type)
    config["BIAS_TYPE_NAME"] = bias_name
    config["WEIGHT_ADJUST"] = _json_safe(weight_adjust)
    config["WEIGHT_ADJUST_NAME"] = adjust_name
    config["TRIAL"] = trial_i
    config["SESSION_ID"] = session_id

    return wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        group=group,
        job_type=adjust_name,
        name=f"{group}__{bias_name}__{adjust_name}__trial{trial_i}",
        config=config,
        tags=[
            str(exp_info["DATASET"]),
            str(exp_info["EXPERT_ALGO"]),
            str(exp_info["IRL_METHOD"]),
            bias_name,
            adjust_name,
        ],
    )


@dataclass
class ExpertDemos:
    """
    The expert demonstrations and feature expectations for one data split.

    Groups the biased and unbiased demonstrations that
    `_generate_expert_demonstrations()` produces for a split so that the
    train/validation/test expert data can be passed around as three objects
    instead of eighteen separate arrays.

    Attributes
    ----------
    mu : array-like<float>, shape(n_expert_demos, n_objectives)
        Feature expectations of the (possibly biased) expert demonstrations.
    demo : pandas.DataFrame
        The expert demonstrations themselves.
    mu_unbiased : array-like<float>, shape(n_expert_demos, n_objectives)
        Feature expectations of the same expert before bias was applied.
    demo_unbiased : pandas.DataFrame
        The unbiased expert demonstrations.
    mu_perf : array-like<float>, shape(1, n_perf_objectives)
        Performance measures of the expert demonstrations.
    mu_perf_unbiased : array-like<float>, shape(1, n_perf_objectives)
        Performance measures of the unbiased expert demonstrations.
    """

    mu: np.ndarray
    demo: pd.DataFrame
    mu_unbiased: np.ndarray
    demo_unbiased: pd.DataFrame
    mu_perf: np.ndarray
    mu_perf_unbiased: np.ndarray


@dataclass
class BiasedExpertDemos:
    """
    The expert demonstrations of one bias type, across all three data splits.

    `run_experiment_trial()` runs the unbiased demonstrations and every bias
    type of `exp_info["BIAS_TYPE_LIST"]` against one shared
    train/validation/test split. This groups everything that is specific to a
    single one of those bias types, so that the trial can iterate over one list
    instead of threading six parallel lists through its loop.

    Attributes
    ----------
    bias_type : tuple
        The bias type these demonstrations were generated with; `()` for the
        unbiased demonstrations.
    expert_train, expert_val, expert_test : ExpertDemos
        The expert demonstrations of each data split.
    subdom_groups_train, subdom_groups_val, subdom_groups_test : tuple
        The subdominance groups of each split's expert demos, as
        `(group_idxs, raw_demos, raw_demos_feat_loss)` triples produced by
        `generate_subdominance_groups()`. They are sampled once and passed to
        every subsequent `compute_iteration_subdominance()` call, so that all
        subdominance computations of this bias type share the same groups and
        the same expert feature losses.
    """

    bias_type: tuple
    expert_train: ExpertDemos
    expert_val: ExpertDemos
    expert_test: ExpertDemos
    subdom_groups_train: tuple
    subdom_groups_val: tuple
    subdom_groups_test: tuple


class PolicyResults:
    """
    The evaluation of one learned policy.

    Owns every quantity `_evaluate_policy()` produces for a single set of
    reward weights, so that `run_experiment_trial()` and
    `_build_trial_summary()` exchange one object instead of thirty values.
    `record()` stores an `_evaluate_policy()` result and reports it to the
    active W&B run.

    Attributes
    ----------
    feat_obj_set_cols : list<str>
        Names of the feature expectation objectives, in order.
    perf_obj_set_cols : list<str>
        Names of the performance measure objectives, in order.
    weights : array-like<float>
        The reward weights the policy was learned from. Set by the caller
        before the policy is computed, not by `record()`.
    runtime : float
        Seconds spent learning and evaluating this policy.
    subdominance : dict<str, float>
        Maps each name in `SUBDOMINANCE_KEYS` to its value.
    """

    def __init__(self, feat_obj_set_cols, perf_obj_set_cols):
        self.feat_obj_set_cols = feat_obj_set_cols
        self.perf_obj_set_cols = perf_obj_set_cols

        self.weights = None
        self.runtime = None
        self.demo_train = None
        self.demo_val = None
        self.demo_test = None
        self.muL_train = None
        self.muL_val = None
        self.muL_test = None
        self.muL_perf_train = None
        self.muL_perf_val = None
        self.muL_perf_test = None
        self.t_train = None
        self.t_val = None
        self.t_test = None
        self.muL_delta_l2_train = None
        self.muL_delta_l2_val = None
        self.muL_delta_l2_test = None
        self.muL_delta_abs_l2_train = None
        self.muL_delta_abs_l2_val = None
        self.muL_delta_abs_l2_test = None
        self.subdominance = {key: None for key in SUBDOMINANCE_KEYS}

    def record(self, evaluate_policy_result, start, run):
        """
        Store an `_evaluate_policy()` result, then log it to the console and
        to W&B.

        Parameters
        ----------
        evaluate_policy_result : tuple
            The return value of `_evaluate_policy()`.
        start : datetime.datetime
            When this policy started being learned, used to report its runtime.
        run : wandb.sdk.wandb_run.Run or None
            The W&B run to log to. `None` disables W&B logging.
        """
        (
            self.demo_train,
            self.demo_val,
            self.demo_test,
            self.muL_train,
            self.muL_val,
            self.muL_test,
            self.muL_perf_train,
            self.muL_perf_val,
            self.muL_perf_test,
            muL_delta_train,
            self.muL_delta_l2_train,
            self.muL_delta_abs_l2_train,
            self.t_train,
            _muL_delta_val,
            self.muL_delta_l2_val,
            self.muL_delta_abs_l2_val,
            self.t_val,
            _muL_delta_test,
            self.muL_delta_l2_test,
            self.muL_delta_abs_l2_test,
            self.t_test,
            *subdominance,
        ) = evaluate_policy_result

        for key, value in zip(SUBDOMINANCE_KEYS, subdominance):
            self.subdominance[key] = value

        self.runtime = (datetime.datetime.now() - start).total_seconds()

        self._log(muL_delta_train, run)

    def _log(self, muL_delta_train, run):
        """Print the policy's stats and send them to the W&B run."""
        logging.info(
            f"\t\t muL_train \t\t= {str(np.round(self.muL_train, 2)).replace('0.', '.')}"
        )
        logging.debug(f"\t\t muL_val = {np.round(self.muL_val, 2)}")
        logging.debug(f"\t\t muL_test = {np.round(self.muL_test, 2)}")
        logging.info(f"\t\t t_train \t\t= {self.t_train:.5f}")
        logging.info(f"\t\t muL_delta_l2_train \t= {self.muL_delta_l2_train:.5f}")
        logging.info(
            f"\t\t muL_delta_abs_l2_train \t= {self.muL_delta_abs_l2_train:.5f}"
        )
        logging.info(
            f"\t\t muL_delta_train \t= {str(np.round(muL_delta_train, 2)).replace('0.', '.')}"
        )
        logging.info(
            f"\t\t weights \t= {str(np.round(self.weights, 2)).replace('0.', '.')}"
        )
        logging.info(f"\t\t Runtime for this policy: {self.runtime}")

        if run is None:
            return

        metrics = {
            "policy/t_train": self.t_train,
            "policy/t_val": self.t_val,
            "policy/t_test": self.t_test,
            "policy/mu_delta_l2_train": self.muL_delta_l2_train,
            "policy/mu_delta_l2_val": self.muL_delta_l2_val,
            "policy/mu_delta_l2_test": self.muL_delta_l2_test,
            "policy/mu_delta_abs_l2_train": self.muL_delta_abs_l2_train,
            "policy/mu_delta_abs_l2_val": self.muL_delta_abs_l2_val,
            "policy/mu_delta_abs_l2_test": self.muL_delta_abs_l2_test,
            "policy/runtime": self.runtime,
        }

        for key, value in self.subdominance.items():
            metrics[f"subdominance/{key}"] = value

        for j, col in enumerate(self.feat_obj_set_cols):
            metrics[f"weight/{col}"] = self.weights[j]
            metrics[f"muL_train/{col}"] = self.muL_train[j]
            metrics[f"muL_val/{col}"] = self.muL_val[j]
            metrics[f"muL_test/{col}"] = self.muL_test[j]
            metrics[f"muL_delta_train/{col}"] = muL_delta_train[j]

        for j, col in enumerate(self.perf_obj_set_cols):
            metrics[f"muL_perf_train/{col}"] = self.muL_perf_train[j]
            metrics[f"muL_perf_val/{col}"] = self.muL_perf_val[j]
            metrics[f"muL_perf_test/{col}"] = self.muL_perf_test[j]

        run.log(metrics)


def _build_trial_summary(
    feat_obj_set,
    perf_obj_set,
    expert_train,
    expert_val,
    expert_test,
    results,
    trial_runtime,
    trial_inputsize,
):
    """
    Summarise one trial's learned policy as a flat metric dict.

    The keys used here are the column names the results CSV used to have, so
    the same quantities are reported as before, addressed by name instead of
    by position in a result row.

    Parameters
    ----------
    feat_obj_set : fair_irl.irl.fair_irl.ObjectiveSet
        The set of objectives for the feature expectations.
    perf_obj_set : fair_irl.irl.fair_irl.ObjectiveSet
        The set of objectives for the performance measures.
    expert_train, expert_val, expert_test : ExpertDemos
        The expert demonstrations for each data split.
    results : PolicyResults
        The evaluation of this trial's learned policy.
    trial_runtime : float
        The runtime to complete the trial, in seconds.
    trial_inputsize : int or None
        The size of the input space. Specifically `mdp.n_states_` (includes y).

    Returns
    -------
    summary : dict<str, numeric>
        The trial's results, ready to be stored in a W&B run summary.
    """
    summary = {}

    expert_mus = (
        ("muE_train", expert_train.mu),
        ("muE_val", expert_val.mu),
        ("muE_test", expert_test.mu),
        ("muE_train_unbiased", expert_train.mu_unbiased),
        ("muE_val_unbiased", expert_val.mu_unbiased),
        ("muE_test_unbiased", expert_test.mu_unbiased),
    )
    for prefix, muE in expert_mus:
        for i, obj in enumerate(feat_obj_set.objectives):
            summary[f"{prefix}_{obj.name}_mean"] = np.mean(muE[:, i])
            summary[f"{prefix}_{obj.name}_std"] = np.std(muE[:, i])

    for i, obj in enumerate(feat_obj_set.objectives):
        summary[f"wL_{obj.name}"] = results.weights[i]

    split_mus = (
        ("train", expert_train.mu, results.muL_train),
        ("val", expert_val.mu, results.muL_val),
        ("test", expert_test.mu, results.muL_test),
    )
    for split, muE, muL in split_mus:
        for i, obj in enumerate(feat_obj_set.objectives):
            summary[f"muL_{split}_{obj.name}"] = muL[i]
            summary[f"muL_{split}_err_{obj.name}"] = muL[i] - np.mean(muE[:, i])

    for key, value in results.subdominance.items():
        summary[key] = value

    summary["muL_train_err_l2"] = results.muL_delta_l2_train
    summary["muL_val_err_l2"] = results.muL_delta_l2_val
    summary["muL_test_err_l2"] = results.muL_delta_l2_test
    summary["mu_delta_abs_l2_train"] = results.muL_delta_abs_l2_train
    summary["mu_delta_abs_l2_val"] = results.muL_delta_abs_l2_val
    summary["mu_delta_abs_l2_test"] = results.muL_delta_abs_l2_test

    # Training errors of the learned policy
    summary["t_train"] = results.t_train
    summary["t_val"] = results.t_val
    summary["t_test"] = results.t_test

    # Performance measures of the expert demos and of the learned policy
    expert_perf_mus = (
        ("muE_perf_train", expert_train.mu_perf),
        ("muE_perf_train_unbiased", expert_train.mu_perf_unbiased),
        ("muE_perf_val", expert_val.mu_perf),
        ("muE_perf_val_unbiased", expert_val.mu_perf_unbiased),
        ("muE_perf_test", expert_test.mu_perf),
        ("muE_perf_test_unbiased", expert_test.mu_perf_unbiased),
    )
    split_perf_mus = (
        ("train", results.muL_perf_train),
        ("val", results.muL_perf_val),
        ("test", results.muL_perf_test),
    )
    for i, obj in enumerate(perf_obj_set.objectives):
        for prefix, muE_perf in expert_perf_mus:
            summary[f"{prefix}_{obj.name}"] = np.mean(muE_perf[:, i])
        for split, muL_perf in split_perf_mus:
            summary[f"muL_perf_{split}_{obj.name}"] = muL_perf[i]

    summary["trial_runtime"] = trial_runtime
    summary["policy_runtime"] = results.runtime
    summary["trial_inputsize"] = trial_inputsize

    return summary


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
    # Since we are using subdominance as an evaluation metric (not a training objective), we don't need to add regularization.
    # final_subdom += (lamda / 2.0) * (np.linalg.norm(alphas) ** 2)
    return final_subdom


# Helper to sample the subdominance groups of a reference (expert) demo set once,
# so that every subsequent compute_iteration_subdominance() call against that
# reference reuses the same groups and the same reference feature losses.
def generate_subdominance_groups(exp_info, raw_demo_ref):
    """Sample the subdominance demo groups for one expert demo set.

    Parameters
    ----------
    exp_info : dict
        Metadata about the experiment.
    raw_demo_ref : pandas.DataFrame
        The expert (raw) demos the learned policy is scored against.

    Returns
    -------
    group_idxs : list<numpy.ndarray>
        The positional indices of each subdominance group. Sampled once here so
        that every learned policy is compared against the expert on the exact
        same groups.
    raw_demos : list<pandas.DataFrame>
        The expert demos of each subdominance group.
    raw_demos_feat_loss : numpy.ndarray
        The feature losses of each expert demo group. Shape is
        (n_subdominance_groups, n_subdominance_metrics).
    """
    raw_demo = raw_demo_ref.copy()
    # OLD IMPLEMENTATION: Split the raw and clf demos into subdominance groups, where there are no repeated demos in each group.
    # raw_demos = np.array_split(raw_demo, exp_info["N_SUBDOMINANCE_GROUPS"])
    # clf_demos = np.array_split(clf_demo, exp_info["N_SUBDOMINANCE_GROUPS"])
    # NEW IMPLEMENTATION: Split the raw and clf demos into subdominance groups, where each group contains half of the demos,
    # sampled randomly without replacement. This way, each group contains a large and diverse set of demos, from which we can
    # more accurately and robustly compute the feature losses. This will help to improve the robustness of the subdominance
    # metric, especially when the number of demos is small.
    n_demos = len(raw_demo)
    group_size = n_demos // 2
    group_idxs = []
    raw_demos = []
    for _ in range(exp_info["N_SUBDOMINANCE_GROUPS"]):
        group_idx = np.random.choice(n_demos, size=group_size, replace=False)
        group_idxs.append(group_idx)
        raw_demos.append(raw_demo.iloc[group_idx])

    raw_demos_feat_loss = np.array(
        [
            compute_relevant_feat_loss(exp_info, raw_demo_group)
            for raw_demo_group in raw_demos
        ]
    )
    return group_idxs, raw_demos, raw_demos_feat_loss


# Helper to compute subdominance for a specific set
def compute_iteration_subdominance(
    exp_info,
    group_idxs,
    raw_demos,
    raw_demos_feat_loss,
    clf_demo_cur,
    subdominance_type="all",
):
    """Compute the subdominance metric of a learned policy's demos.

    `group_idxs`, `raw_demos` and `raw_demos_feat_loss` come from
    `generate_subdominance_groups()`, which is called once per expert demo set in
    `_split_dataset_and_generate_expert_demos()`. Reusing them here keeps the
    subdominance groups (and the expert feature losses computed from them)
    identical across every call, so the subdominance values of different learned
    policies are directly comparable.

    `raw_demos` is not needed to compute the metric itself -- it is carried
    alongside its feature losses so callers hold the groups those losses came
    from.
    """
    # Compute subdominance metric for the learned policy, using the same
    # subdominance groups that the expert feature losses were computed on.
    clf_demo = clf_demo_cur.copy()
    clf_demos = [clf_demo.iloc[group_idx] for group_idx in group_idxs]

    clf_demos_feat_loss = np.array(
        [
            compute_relevant_feat_loss(exp_info, clf_demo_group)
            for clf_demo_group in clf_demos
        ]
    )
    alphas = compute_alphas(raw_demos_feat_loss, clf_demos_feat_loss)

    if subdominance_type == "max_abs" or subdominance_type == "all":
        # Compute max-aggregated absolute subdominance:
        max_abs_subdom = compute_subdominance(
            exp_info,
            alphas,
            raw_demos_feat_loss,
            clf_demos_feat_loss,
            relative=False,
            sum_agg=False,
        )
    if subdominance_type == "sum_abs" or subdominance_type == "all":
        # Compute sum-aggregated absolute subdominance:
        sum_abs_subdom = compute_subdominance(
            exp_info,
            alphas,
            raw_demos_feat_loss,
            clf_demos_feat_loss,
            relative=False,
            sum_agg=True,
        )
    if subdominance_type == "max_rel" or subdominance_type == "all":
        # Compute max-aggregated relative subdominance:
        max_rel_subdom = compute_subdominance(
            exp_info,
            alphas,
            raw_demos_feat_loss,
            clf_demos_feat_loss,
            relative=True,
            sum_agg=False,
        )
    if subdominance_type == "sum_rel" or subdominance_type == "all":
        # Compute sum-aggregated relative subdominance:
        sum_rel_subdom = compute_subdominance(
            exp_info,
            alphas,
            raw_demos_feat_loss,
            clf_demos_feat_loss,
            relative=True,
            sum_agg=True,
        )
    if subdominance_type == "max_abs":
        return max_abs_subdom
    elif subdominance_type == "sum_abs":
        return sum_abs_subdom
    elif subdominance_type == "max_rel":
        return max_rel_subdom
    elif subdominance_type == "sum_rel":
        return sum_rel_subdom
    elif subdominance_type == "all":
        return max_abs_subdom, sum_abs_subdom, max_rel_subdom, sum_rel_subdom
    else:
        raise ValueError(
            f"Invalid subdominance_type: {subdominance_type}. Must be one of ['max_abs', 'sum_abs', 'max_rel', 'sum_rel', 'all']"
        )


def _build_objective_sets(exp_info):
    feat_obj_set = ObjectiveSet(
        [OBJ_LOOKUP_BY_NAME[name]() for name in exp_info["FEAT_EXP_OBJECTIVE_NAMES"]]
    )
    feat_obj_set.reset()

    perf_obj_set = ObjectiveSet(
        [OBJ_LOOKUP_BY_NAME[name]() for name in exp_info["PERF_MEAS_OBJECTIVE_NAMES"]]
    )
    perf_obj_set.reset()

    return feat_obj_set, perf_obj_set


def _load_or_generate_dataset(exp_info, X, y, feature_types):
    if X is None or y is None or feature_types is None:
        return generate_dataset(
            exp_info["DATASET"],
            n_samples=exp_info["N_DATASET_SAMPLES"],
        )
    return X, y, feature_types


def _generate_expert_demonstrations(
    exp_info,
    X,
    y,
    feature_types,
    expert_algo_lookup,
    feat_obj_set,
    perf_obj_set,
    bias_type_list,
):
    """Generate the expert demonstrations of every bias type for one data split.

    All the bias types share one set of unbiased demonstrations, so they only
    differ from each other by the bias that was applied.

    Returns
    -------
    experts : list<ExpertDemos>
        The biased and unbiased demonstrations, feature expectations and
        performance measures for this split, one entry per entry of
        `bias_type_list` and in the same order.
    """
    clf = copy.deepcopy(expert_algo_lookup[exp_info["EXPERT_ALGO"]])
    muE_unbiased, demoE_unbiased, biased = generate_mu_and_demos(
        exp_info,
        X=X,
        y=y,
        feature_types=feature_types,
        clf=clf,
        obj_set=feat_obj_set,
        n_demos=exp_info["N_EXPERT_DEMOS"],
        bias_type_list=bias_type_list,
    )
    muE_perf_unbiased = np.array(
        [perf_obj_set.compute_demo_feature_exp(demoE_unbiased)]
    )
    return [
        ExpertDemos(
            mu=muE,
            demo=demoE,
            mu_unbiased=muE_unbiased,
            demo_unbiased=demoE_unbiased,
            mu_perf=np.array([perf_obj_set.compute_demo_feature_exp(demoE)]),
            mu_perf_unbiased=muE_perf_unbiased,
        )
        for muE, demoE in biased
    ]


def _is_split_acceptable(muE_train, muE_val, muE_test, max_muE_cosine_dist_split):
    for demo_i in range(len(muE_train)):
        if cosine(muE_train[demo_i], muE_val[0]) > max_muE_cosine_dist_split:
            return False
        if cosine(muE_train[demo_i], muE_test[0]) > max_muE_cosine_dist_split:
            return False
    return True


def _split_dataset_and_generate_expert_demos(
    exp_info,
    expert_algo_lookup,
    feat_obj_set,
    perf_obj_set,
    X,
    y,
    feature_types,
    bias_type_list,
):
    """Split the dataset once, then generate expert demonstrations for each
    split and each bias type.

    Every bias type is applied to the same train/validation/test split, so the
    policies learned from them differ only by the bias of the demonstrations
    they were learned from. A split is only accepted once it passes
    `_is_split_acceptable()`, which is checked against the unbiased expert
    demonstrations the bias types share rather than against each bias type's
    own demonstrations, so that how often a split is retried does not depend
    on how many bias types are being run.

    Returns
    -------
    (X_train, X_val, X_test, y_train, y_val, y_test) : pandas objects
        The three data splits, shared by every bias type.
    demos_by_bias_type : list<BiasedExpertDemos>
        The expert demonstrations of each bias type, in the order of
        `bias_type_list`.
    """
    max_muE_cosine_dist_split = 0.002

    while True:
        X_train, X_val_test, y_train, y_val_test = train_test_split(
            X, y, train_size=0.60
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_val_test, y_val_test, test_size=0.50
        )

        logging.info(
            "Generating expert demonstrations and feature expectations for training set..."
        )
        experts_train = _generate_expert_demonstrations(
            exp_info,
            X_train,
            y_train,
            feature_types,
            expert_algo_lookup,
            feat_obj_set,
            perf_obj_set,
            bias_type_list,
        )

        logging.info(
            "Generating expert demonstrations and feature expectations for validation set..."
        )
        experts_val = _generate_expert_demonstrations(
            exp_info,
            X_val,
            y_val,
            feature_types,
            expert_algo_lookup,
            feat_obj_set,
            perf_obj_set,
            bias_type_list,
        )

        logging.info(
            "Generating expert demonstrations and feature expectations for test set..."
        )
        experts_test = _generate_expert_demonstrations(
            exp_info,
            X_test,
            y_test,
            feature_types,
            expert_algo_lookup,
            feat_obj_set,
            perf_obj_set,
            bias_type_list,
        )

        # Every bias type of this split shares one set of unbiased expert
        # demonstrations, so checking the split against those checks it once
        # for all of them.
        if _is_split_acceptable(
            experts_train[0].mu_unbiased,
            experts_val[0].mu_unbiased,
            experts_test[0].mu_unbiased,
            max_muE_cosine_dist_split,
        ):
            break

        logging.info(
            "INFO: Split check failed; retrying train/validation/test split..."
        )

    # Sample the subdominance groups (and compute the expert feature losses on
    # them) once per split and bias type, so that every
    # compute_iteration_subdominance() call of that bias type reuses the same
    # groups.
    logging.info("Generating subdominance groups for the expert demonstrations...")
    demos_by_bias_type = [
        BiasedExpertDemos(
            bias_type=bias_type,
            expert_train=expert_train,
            expert_val=expert_val,
            expert_test=expert_test,
            subdom_groups_train=generate_subdominance_groups(
                exp_info, expert_train.demo
            ),
            subdom_groups_val=generate_subdominance_groups(exp_info, expert_val.demo),
            subdom_groups_test=generate_subdominance_groups(exp_info, expert_test.demo),
        )
        for bias_type, expert_train, expert_val, expert_test in zip(
            bias_type_list, experts_train, experts_val, experts_test
        )
    ]

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ), demos_by_bias_type


def _compute_errors_and_metrics(
    wi,
    muE_train,
    muE_val,
    muE_test,
    muL_train,
    muL_val,
    muL_test,
    exp_info,
):
    """Compute the learned policy's feature expectation error metrics for the
    train/validation/test sets."""
    (
        muL_delta_train,
        muL_delta_l2_train,
        muL_delta_abs_l2_train,
        ti_train,
    ) = policy_error(
        wi,
        muE_train,
        muL_train,
        dot_weights_feat_exp=exp_info["DOT_WEIGHTS_FEAT_EXP"],
    )
    (
        muL_delta_val,
        muL_delta_l2_val,
        muL_delta_abs_l2_val,
        ti_val,
    ) = policy_error(
        wi,
        muE_val,
        muL_val,
        dot_weights_feat_exp=exp_info["DOT_WEIGHTS_FEAT_EXP"],
    )
    (
        muL_delta_test,
        muL_delta_l2_test,
        muL_delta_abs_l2_test,
        ti_test,
    ) = policy_error(
        wi,
        muE_test,
        muL_test,
        dot_weights_feat_exp=exp_info["DOT_WEIGHTS_FEAT_EXP"],
    )

    return (
        muL_delta_train,
        muL_delta_l2_train,
        muL_delta_abs_l2_train,
        ti_train,
        muL_delta_val,
        muL_delta_l2_val,
        muL_delta_abs_l2_val,
        ti_val,
        muL_delta_test,
        muL_delta_l2_test,
        muL_delta_abs_l2_test,
        ti_test,
    )


def _fit_clf_and_demo_df(feature_types, X_train, y_train):
    """Fit a y|x predictor and build the demo DataFrame used by compute_optimal_policy."""
    clf = sklearn_clf_pipeline(
        feature_types=feature_types,
        clf_inst=RandomForestClassifier(),
    )
    clf.fit(X_train, y_train)
    demo_df = pd.DataFrame(X_train)
    demo_df["y"] = y_train
    return clf, demo_df


def _apply_weight_adjustments(
    wi,
    weight_adjust,
    feat_obj_set,
    demo_df,
    clf,
    x_cols,
    exp_info,
    X,
    y,
    can_observe_y,
    subdom_groups,
    run=None,
):
    """Apply a sequence of weight adjustment operations to wi and return the result.

    `run` is the W&B run of the policy these weights are being learned for, or
    `None` to disable W&B logging. Only the adjustments that iterate (i.e.
    `"opt_debias"`) report anything to it.
    """
    if len(weight_adjust) == 0:
        pass
    elif weight_adjust[0] == "mul_negative_weights":
        mul_factor = weight_adjust[1]
        wi[wi < 0] = wi[wi < 0] * mul_factor
    elif weight_adjust[0] == "opt_debias":
        library = weight_adjust[1]
        optimizer = weight_adjust[2]
        n_steps = weight_adjust[3] if 3 < len(weight_adjust) else 400
        wi = iteratively_optimize_weights(
            wi,
            feat_obj_set,
            demo_df,
            clf,
            x_cols,
            exp_info,
            X,
            y,
            can_observe_y,
            subdom_groups,
            library,
            optimizer,
            n_steps,
            run=run,
        )
    wi = normalize(wi.reshape(1, -1), norm="l1").flatten()
    return wi


# The most iterations `iteratively_optimize_weights()` runs before giving up on
# finding a better weight set. This is only a safety net -- the loop is expected
# to stop on its own as soon as an iteration fails to improve.
# TODO: Make this configurable via exp_info
OPT_DEBIAS_MAX_ITERATIONS = 10


def _append_subdominance_groups(exp_info, subdom_groups, clf_demo):
    """Add the demos of a learned policy to a set of subdominance groups.

    `clf_demo` is sampled into `N_SUBDOMINANCE_GROUPS` new groups exactly the
    way `generate_subdominance_groups()` samples an expert demo set, and those
    groups are appended to the ones `subdom_groups` already holds. In other
    words, the learned policy's demos become additional *raw* (reference) demos
    that any later policy is scored against.

    Returns a brand new tuple of brand new containers, so `subdom_groups` --
    which the caller shares with every other weight adjustment of the trial --
    is left completely untouched.

    Parameters
    ----------
    exp_info : dict
        Metadata about the experiment.
    subdom_groups : tuple
        `(group_idxs, raw_demos, raw_demos_feat_loss)`, as returned by
        `generate_subdominance_groups()` or by this function.
    clf_demo : pandas.DataFrame
        The demos of the learned policy to add as reference demos. Must cover
        the same rows as the demos the groups already hold, since the groups
        are positional indices into whichever demo set is being scored.

    Returns
    -------
    subdom_groups : tuple
        The same three elements, each `N_SUBDOMINANCE_GROUPS` entries longer.
    """
    group_idxs, raw_demos, raw_demos_feat_loss = subdom_groups
    (
        new_group_idxs,
        new_raw_demos,
        new_raw_demos_feat_loss,
    ) = generate_subdominance_groups(exp_info, clf_demo)
    return (
        list(group_idxs) + list(new_group_idxs),
        list(raw_demos) + list(new_raw_demos),
        np.concatenate([raw_demos_feat_loss, new_raw_demos_feat_loss]),
    )


def _log_opt_debias_iteration(
    run,
    feat_obj_set,
    iteration,
    wi,
    muL,
    expert_subdom,
    augmented_subdom,
    n_groups,
):
    """Report one `iteratively_optimize_weights()` iteration to the console and W&B."""
    logging.info(
        f"\t\t opt_debias iteration {iteration}: "
        f"sum_abs_subdominance = {expert_subdom:.5f} "
        f"({augmented_subdom:.5f} vs the {n_groups} augmented groups)"
    )
    logging.info(f"\t\t\t weights \t= {str(np.round(wi, 2)).replace('0.', '.')}")
    logging.info(f"\t\t\t muL \t\t= {str(np.round(muL, 2)).replace('0.', '.')}")

    if run is None:
        return

    metrics = {
        "opt_debias/iteration": iteration,
        "opt_debias/sum_abs_subdominance": expert_subdom,
        "opt_debias/sum_abs_subdominance_augmented": augmented_subdom,
        "opt_debias/n_subdominance_groups": n_groups,
    }
    for j, obj in enumerate(feat_obj_set.objectives):
        metrics[f"opt_debias/weight/{obj.name}"] = wi[j]
        metrics[f"opt_debias/muL/{obj.name}"] = muL[j]

    run.log(metrics)


def iteratively_optimize_weights(
    wi,
    feat_obj_set,
    demo_df,
    clf,
    x_cols,
    exp_info,
    X,
    y,
    can_observe_y,
    subdom_groups,
    library,
    optimizer,
    n_steps,
    run=None,
):
    """Optimize the reward weights against a growing set of subdominance groups.

    Each iteration

    1. optimizes the weights with `optimize_weights()`, starting from the
       previous iteration's weights and scoring against every subdominance
       group accumulated so far,
    2. computes the optimal policy of those weights and the demos it produces,
       and
    3. appends those demos -- sampled into `N_SUBDOMINANCE_GROUPS` groups the
       same way the expert demos were -- to the subdominance groups as extra
       raw demos, so that the next iteration is also penalized for behaving
       like this iteration's policy.

    Iterating stops as soon as an iteration's policy fails to improve on the
    best sum-aggregated absolute subdominance seen so far, and the weights of
    the lowest-subdominance iteration are returned.

    Improvement is judged against the *original* `subdom_groups` (the expert
    demos), not against the augmented ones: only the original groups stay fixed
    across iterations, so only subdominance measured against them is comparable
    from one iteration to the next. The subdominance against the augmented
    groups -- the quantity `optimize_weights()` actually minimizes -- is
    reported alongside it, but is not what the stopping rule looks at.

    `subdom_groups` is never mutated. The appended groups only ever go into
    containers created by `_append_subdominance_groups()`, so the caller's
    groups stay usable by the trial's other weight adjustments.

    Parameters
    ----------
    wi : array-like<float>
        The weights to start optimizing from.
    subdom_groups : tuple
        The expert `(group_idxs, raw_demos, raw_demos_feat_loss)` of the split
        being optimized against.
    run : wandb.sdk.wandb_run.Run or None
        The W&B run to report each iteration to. `None` disables W&B logging.

    Returns
    -------
    wi : numpy.ndarray
        The L1 normalized weights of the iteration with the lowest
        subdominance against the expert subdominance groups.
    """
    if run is not None:
        # Plot every `opt_debias/` metric against the iteration it came from
        # rather than against the run's global step.
        run.define_metric("opt_debias/iteration")
        run.define_metric("opt_debias/*", step_metric="opt_debias/iteration")

    expert_subdom_groups = subdom_groups
    cur_subdom_groups = subdom_groups
    cur_wi = normalize(np.asarray(wi, dtype=float).reshape(1, -1), norm="l1").flatten()

    best_wi = cur_wi
    best_subdom = np.inf
    n_iterations = 0

    for iteration in range(OPT_DEBIAS_MAX_ITERATIONS):
        cur_wi = optimize_weights(
            cur_wi,
            feat_obj_set,
            demo_df,
            clf,
            x_cols,
            exp_info,
            X,
            y,
            can_observe_y,
            cur_subdom_groups,
            library,
            optimizer,
            n_steps,
        )

        # The optimal classifier of this iteration's weights, and its demos.
        reward_weights = {
            obj.name: cur_wi[j] for j, obj in enumerate(feat_obj_set.objectives)
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
        demo = generate_demo(clf_pol, X, y, can_observe_y=can_observe_y)

        expert_subdom = compute_iteration_subdominance(
            exp_info,
            *expert_subdom_groups,
            clf_demo_cur=demo,
            subdominance_type="sum_abs",
        )
        if cur_subdom_groups is expert_subdom_groups:
            # First iteration: nothing has been appended yet, so the augmented
            # groups still are the expert groups.
            augmented_subdom = expert_subdom
        else:
            augmented_subdom = compute_iteration_subdominance(
                exp_info,
                *cur_subdom_groups,
                clf_demo_cur=demo,
                subdominance_type="sum_abs",
            )

        n_iterations = iteration + 1
        _log_opt_debias_iteration(
            run,
            feat_obj_set,
            iteration,
            cur_wi,
            feat_obj_set.compute_demo_feature_exp(demo),
            expert_subdom,
            augmented_subdom,
            len(cur_subdom_groups[1]),
        )

        if expert_subdom >= best_subdom:
            logging.info(
                "\t\t opt_debias stopping: this iteration did not improve on "
                f"the best sum_abs_subdominance ({best_subdom:.5f})"
            )
            break

        best_subdom = expert_subdom
        best_wi = cur_wi

        # Treat this iteration's demos as raw demos of their own subdominance
        # groups, so the next iteration has to beat this policy too.
        cur_subdom_groups = _append_subdominance_groups(
            exp_info, cur_subdom_groups, demo
        )
    else:
        logging.info(
            f"\t\t opt_debias stopping: hit the {OPT_DEBIAS_MAX_ITERATIONS} "
            "iteration cap while still improving"
        )

    logging.info(
        f"\t\t opt_debias ran {n_iterations} iteration(s); best "
        f"sum_abs_subdominance = {best_subdom:.5f}"
    )
    if run is not None:
        run.summary["opt_debias/n_iterations"] = n_iterations
        run.summary["opt_debias/best_sum_abs_subdominance"] = best_subdom

    return best_wi


def optimize_weights(
    wi,
    feat_obj_set,
    demo_df,
    clf,
    x_cols,
    exp_info,
    X,
    y,
    can_observe_y,
    subdom_groups,
    library,
    optimizer,
    n_steps,
):
    if library == "optuna":
        objective = partial(
            optuna_objective,
            feat_obj_set=feat_obj_set,
            demo_df=demo_df,
            clf=clf,
            x_cols=x_cols,
            exp_info=exp_info,
            X=X,
            y=y,
            can_observe_y=can_observe_y,
            subdom_groups=subdom_groups,
        )
        n_weights = len(feat_obj_set.objectives)
        x0 = np.clip(np.asarray(wi, dtype=float), -1.0, 1.0)
        if optimizer == "CMA-ES":
            sampler = optuna.samplers.CmaEsSampler(
                x0={f"unnormalized_w{j}": float(x0[j]) for j in range(n_weights)},
                seed=exp_info["RANDOM_SEED"],
            )
            study = optuna.create_study(
                direction="minimize",
                sampler=sampler,
                # optuna.create_study() otherwise generates a random
                # UUID study name (via uuid.uuid4(), independent of
                # any seed), which is a source of non-determinism.
                study_name=f"opt_debias_{exp_info['RANDOM_SEED']}",
            )
            study.optimize(objective, n_trials=n_steps)
        unnormalized_wi = np.array(
            [study.best_params[f"unnormalized_w{j}"] for j in range(n_weights)]
        )
    elif library == "pybobyqa":
        objective = partial(
            pybobyqa_objective,
            feat_obj_set=feat_obj_set,
            demo_df=demo_df,
            clf=clf,
            x_cols=x_cols,
            exp_info=exp_info,
            X=X,
            y=y,
            can_observe_y=can_observe_y,
            subdom_groups=subdom_groups,
        )
        n_weights = len(feat_obj_set.objectives)
        lower_bounds = -1.0 * np.ones(n_weights)
        upper_bounds = 1.0 * np.ones(n_weights)
        x0 = np.clip(np.asarray(wi, dtype=float), lower_bounds, upper_bounds)
        if optimizer == "Multi-Start BOBYQA":
            # Py-BOBYQA has no seed parameter of its own; it draws
            # its multi-start restarts from the global numpy RNG
            soln = pybobyqa.solve(
                objective,
                x0,
                bounds=(lower_bounds, upper_bounds),
                seek_global_minimum=True,
                maxfun=n_steps,
            )
        unnormalized_wi = np.array(soln.x)
        logging.info(f"unnormalized_wi: {unnormalized_wi}")
    elif library == "nevergrad":
        objective = partial(
            nevergrad_objective,
            feat_obj_set=feat_obj_set,
            demo_df=demo_df,
            clf=clf,
            x_cols=x_cols,
            exp_info=exp_info,
            X=X,
            y=y,
            can_observe_y=can_observe_y,
            subdom_groups=subdom_groups,
        )
        n_weights = len(feat_obj_set.objectives)
        # parametrization = ng.p.Array(shape=(n_weights,)).set_bounds(-1.0, 1.0)
        parametrization = ng.p.Array(init=np.asarray(wi, dtype=float)).set_bounds(
            -1.0, 1.0
        )
        parametrization.random_state = np.random.RandomState(exp_info["RANDOM_SEED"])
        if optimizer == "BayesOpt":
            ng_optimizer = ng.optimizers.BO(
                parametrization=parametrization, budget=n_steps
            )
        elif optimizer == "Nelder-Mead":
            ng_optimizer = ng.optimizers.NelderMead(
                parametrization=parametrization, budget=n_steps
            )
        elif optimizer == "Powell":
            ng_optimizer = ng.optimizers.Powell(
                parametrization=parametrization, budget=n_steps
            )
        recommendation = ng_optimizer.minimize(objective)
        unnormalized_wi = np.array(recommendation.value)
    return normalize(unnormalized_wi.reshape(1, -1), norm="l1").flatten()


def optuna_objective(
    trial,
    feat_obj_set,
    demo_df,
    clf,
    x_cols,
    exp_info,
    X,
    y,
    can_observe_y,
    subdom_groups,
):
    """Objective function for Optuna optimization of weights."""
    n_weights = len(feat_obj_set.objectives)
    unnormalized_weights = np.array(
        [trial.suggest_float(f"unnormalized_w{j}", -1.0, 1.0) for j in range(n_weights)]
    )
    # Evaluate the weights using the provided evaluation function
    subdominance_loss = subdominance_of_weights(
        unnormalized_weights,
        feat_obj_set,
        demo_df,
        clf,
        x_cols,
        exp_info,
        X,
        y,
        can_observe_y,
        subdom_groups,
    )
    return subdominance_loss


def pybobyqa_objective(
    unnormalized_weights,
    feat_obj_set,
    demo_df,
    clf,
    x_cols,
    exp_info,
    X,
    y,
    can_observe_y,
    subdom_groups,
):
    """Objective function for Py-BOBYQA optimization of weights."""
    unnormalized_weights = np.array(unnormalized_weights)
    # Evaluate the weights using the provided evaluation function
    subdominance_loss = subdominance_of_weights(
        unnormalized_weights,
        feat_obj_set,
        demo_df,
        clf,
        x_cols,
        exp_info,
        X,
        y,
        can_observe_y,
        subdom_groups,
    )
    return subdominance_loss


def nevergrad_objective(
    unnormalized_weights,
    feat_obj_set,
    demo_df,
    clf,
    x_cols,
    exp_info,
    X,
    y,
    can_observe_y,
    subdom_groups,
):
    """Objective function for Nevergrad optimization of weights."""
    unnormalized_weights = np.array(unnormalized_weights)
    # Evaluate the weights using the provided evaluation function
    subdominance_loss = subdominance_of_weights(
        unnormalized_weights,
        feat_obj_set,
        demo_df,
        clf,
        x_cols,
        exp_info,
        X,
        y,
        can_observe_y,
        subdom_groups,
    )
    return subdominance_loss


def subdominance_of_weights(
    unnormalized_weights,
    feat_obj_set,
    demo_df,
    clf,
    x_cols,
    exp_info,
    X,
    y,
    can_observe_y,
    subdom_groups,
):
    """Computes the subdominance loss for a given set of weights. Uses the weights to
    compute the optimal policy, uses that policy to generate demos, uses the demos to
    compute feature expectations, and then computes the subdominance of the generated
    demos against the expert demos. Returns the subdominance loss metric for that weight set.
    """
    # Start by L1 normalizing the weights to ensure they sum to 1
    weights = normalize(unnormalized_weights.reshape(1, -1), norm="l1").flatten()
    # Compute the optimal policy for the given weights
    reward_weights = {
        obj.name: weights[j] for j, obj in enumerate(feat_obj_set.objectives)
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
    # Generate demos from the optimal policy
    demo = generate_demo(clf_pol, X, y, can_observe_y=can_observe_y)
    # Compute the subdominance of the generated demos against the expert demos
    # lp = LineProfiler()
    # lp.add_function(compute_iteration_subdominance)
    # lp.enable_by_count()
    group_idxs, raw_demos, raw_demos_feat_loss = subdom_groups
    sum_abs_subdom = compute_iteration_subdominance(
        exp_info,
        group_idxs,
        raw_demos,
        raw_demos_feat_loss,
        demo,
        subdominance_type="sum_abs",
    )
    # lp.disable_by_count()
    # lp.print_stats()
    return sum_abs_subdom


def _evaluate_policy(
    clf_pol,
    wi,
    feat_obj_set,
    perf_obj_set,
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    subdom_groups_train,
    subdom_groups_val,
    subdom_groups_test,
    muE_train,
    muE_val,
    muE_test,
    exp_info,
    can_observe_y,
):
    """Generate demos from clf_pol, evaluate errors via _compute_errors_and_metrics,
    and compute subdominance via compute_iteration_subdominance for train/val/test sets.
    """
    logging.debug("\tGenerating learned demostration...")
    demo_train = generate_demo(clf_pol, X_train, y_train, can_observe_y=can_observe_y)
    demo_val = generate_demo(clf_pol, X_val, y_val, can_observe_y=False)
    demo_test = generate_demo(clf_pol, X_test, y_test, can_observe_y=False)

    muL_train = feat_obj_set.compute_demo_feature_exp(demo_train)
    muL_val = feat_obj_set.compute_demo_feature_exp(demo_val)
    muL_test = feat_obj_set.compute_demo_feature_exp(demo_test)
    muL_perf_train = perf_obj_set.compute_demo_feature_exp(demo_train)
    muL_perf_val = perf_obj_set.compute_demo_feature_exp(demo_val)
    muL_perf_test = perf_obj_set.compute_demo_feature_exp(demo_test)

    (
        muL_delta_train,
        muL_delta_l2_train,
        muL_delta_abs_l2_train,
        ti_train,
        muL_delta_val,
        muL_delta_l2_val,
        muL_delta_abs_l2_val,
        ti_val,
        muL_delta_test,
        muL_delta_l2_test,
        muL_delta_abs_l2_test,
        ti_test,
    ) = _compute_errors_and_metrics(
        wi,
        muE_train,
        muE_val,
        muE_test,
        muL_train,
        muL_val,
        muL_test,
        exp_info,
    )

    (
        max_abs_subdominance,
        sum_abs_subdominance,
        max_rel_subdominance,
        sum_rel_subdominance,
    ) = compute_iteration_subdominance(
        exp_info, *subdom_groups_train, clf_demo_cur=demo_train
    )

    (
        max_abs_subdom_v,
        sum_abs_subdom_v,
        max_rel_subdom_v,
        sum_rel_subdom_v,
    ) = compute_iteration_subdominance(
        exp_info, *subdom_groups_val, clf_demo_cur=demo_val
    )

    (
        max_abs_subdom_t,
        sum_abs_subdom_t,
        max_rel_subdom_t,
        sum_rel_subdom_t,
    ) = compute_iteration_subdominance(
        exp_info, *subdom_groups_test, clf_demo_cur=demo_test
    )

    return (
        demo_train,
        demo_val,
        demo_test,
        muL_train,
        muL_val,
        muL_test,
        muL_perf_train,
        muL_perf_val,
        muL_perf_test,
        muL_delta_train,
        muL_delta_l2_train,
        muL_delta_abs_l2_train,
        ti_train,
        muL_delta_val,
        muL_delta_l2_val,
        muL_delta_abs_l2_val,
        ti_val,
        muL_delta_test,
        muL_delta_l2_test,
        muL_delta_abs_l2_test,
        ti_test,
        max_abs_subdominance,
        sum_abs_subdominance,
        max_rel_subdominance,
        sum_rel_subdominance,
        max_abs_subdom_v,
        sum_abs_subdom_v,
        max_rel_subdom_v,
        sum_rel_subdom_v,
        max_abs_subdom_t,
        sum_abs_subdom_t,
        max_rel_subdom_t,
        sum_rel_subdom_t,
    )


def _finalize_trial(
    run,
    results,
    feat_obj_set,
    perf_obj_set,
    expert_train,
    expert_val,
    expert_test,
    clf_pol,
    trial_runtime,
):
    """Book-keeping for one (bias type, weight adjustment) pair of a trial:
    report the learned policy's results to W&B, where they are stored in the
    run summary.
    """
    # Compare the learned policy with the expert demonstrations
    logging.info(
        f"Learned Policy yhat (not real yhat since doesn't factor mu0): {results.demo_train['yhat'].mean():.3f}"
    )
    logging.info(f"weights:\t {np.round(results.weights, 3)}")

    trial_inputsize = None
    if hasattr(clf_pol, "mdp"):
        trial_inputsize = clf_pol.mdp.n_states_

    logging.debug("Experiment Summary")

    summary = _build_trial_summary(
        feat_obj_set,
        perf_obj_set,
        expert_train,
        expert_val,
        expert_test,
        results,
        trial_runtime,
        trial_inputsize,
    )
    run.summary["converged"] = True
    run.summary.update(summary)


def run_experiment_trial(
    exp_info,
    X=None,
    y=None,
    feature_types=None,
    trial_i=0,
    group=None,
    session_id=None,
):
    """
    Runs 1 trial to learn an optimal classifier.

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
    trial_i : int, default 0
        Index of this trial within the experiment. Recorded in W&B.
    group : str, Optional
        W&B group shared by every run of this experiment.
    session_id : str, Optional
        Identifies the execution of the experiment script these runs belong to.
        Defaults to a fresh id, so that calling this directly still produces a
        self-contained session.

    Reports
    -------
    One W&B run per (bias type, weight adjustment) pair: first the unbiased
    demonstrations, then one per entry in exp_info["BIAS_TYPE_LIST"] (which is
    not expected to contain "()" itself), and within each of those first the
    unadjusted weights, then one per entry in exp_info["WEIGHT_ADJUST_LIST"]
    (which is not expected to contain "()" itself either), each derived from
    the unadjusted weights via `_apply_weight_adjustments()`. Every run holds
    this trial's config, the metrics of the policy it learned, and a summary of
    that policy.

    Every bias type is applied to the same train/validation/test split, and
    learns its policies from the same `y|x` predictor, so that the only
    difference between them is the bias of the expert demonstrations.
    """
    trial_start = datetime.datetime.now()

    if session_id is None:
        session_id = new_session_id()

    # The unbiased demonstrations every bias type is derived from always get
    # their own set of runs, so `()` is prepended here and BIAS_TYPE_LIST is
    # not expected to contain it itself. Prepending it also puts the unbiased
    # runs first, ahead of every listed bias type.
    bias_type_list = ((),) + tuple(exp_info["BIAS_TYPE_LIST"])
    weight_adjust_list = ((),) + tuple(exp_info["WEIGHT_ADJUST_LIST"])

    feat_obj_set, perf_obj_set = _build_objective_sets(exp_info)
    can_observe_y = "FO" in exp_info["IRL_METHOD"]
    X, y, feature_types = _load_or_generate_dataset(exp_info, X, y, feature_types)

    expert_algo_lookup = generate_expert_algo_lookup(feature_types)

    (
        (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
        ),
        demos_by_bias_type,
    ) = _split_dataset_and_generate_expert_demos(
        exp_info,
        expert_algo_lookup,
        feat_obj_set,
        perf_obj_set,
        X,
        y,
        feature_types,
        bias_type_list,
    )

    x_cols = (
        feature_types["boolean"]
        + feature_types["categoric"]
        + feature_types["continuous"]
    )
    x_cols.remove("z")
    feat_obj_set_cols = [obj.name for obj in feat_obj_set.objectives]
    perf_obj_set_cols = [obj.name for obj in perf_obj_set.objectives]

    # The `y|x` predictor every learned policy of this trial is computed from.
    logging.debug("Fitting `y|x` predictor for clf policy...")
    clf, demo_df = _fit_clf_and_demo_df(feature_types, X_train, y_train)

    # The expert is an optimal classifier policy whose reward weights are
    # equal and positive across every objective, so those same weights are the
    # unadjusted ones every weight adjustment is derived from.
    n_objs = len(feat_obj_set.objectives)
    unadjusted_weight = np.full(n_objs, 1.0 / n_objs)

    # The unbiased demonstrations, then every bias type of this trial, each
    # against the same data split and the same `y|x` predictor. Within a bias
    # type, one run for the unadjusted weights, then one per weight adjustment.
    # weight_adjust_list is not expected to contain "()" itself; the unadjusted
    # weights always get their own W&B run regardless of its contents.
    for bias_demos in demos_by_bias_type:
        expert_train = bias_demos.expert_train
        expert_val = bias_demos.expert_val
        expert_test = bias_demos.expert_test

        logging.info(f"BIAS TYPE: {bias_demos.bias_type}")
        logging.info(f"muE_train:\n{expert_train.mu}")
        logging.info(f"muE_val:\n{expert_val.mu}")
        logging.info(f"muE_test:\n{expert_test.mu}")
        logging.info(f"muE_perf_train:\n{expert_train.mu_perf}")
        logging.info(f"muE_perf_val:\n{expert_val.mu_perf}")
        logging.info(f"muE_perf_test:\n{expert_test.mu_perf}")

        for weight_adjust in weight_adjust_list:
            with start_wandb_run(
                exp_info,
                bias_demos.bias_type,
                weight_adjust,
                trial_i,
                group,
                session_id,
            ) as run:
                policy_start = datetime.datetime.now()

                results = PolicyResults(feat_obj_set_cols, perf_obj_set_cols)

                wi = _apply_weight_adjustments(
                    unadjusted_weight.copy(),
                    weight_adjust,
                    feat_obj_set,
                    demo_df,
                    clf,
                    x_cols,
                    exp_info,
                    X_train,  # could switch to X_val
                    y_train,  # could switch to y_val
                    can_observe_y,
                    # could switch to bias_demos.subdom_groups_val
                    bias_demos.subdom_groups_train,
                    run=run,
                )

                # Learn a policy that maximizes the reward function.
                results.weights = wi

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
                # Measure and record the error of the learned policy.
                ##

                evaluate_policy_result = _evaluate_policy(
                    clf_pol,
                    wi,
                    feat_obj_set,
                    perf_obj_set,
                    X_train,
                    X_val,
                    X_test,
                    y_train,
                    y_val,
                    y_test,
                    bias_demos.subdom_groups_train,
                    bias_demos.subdom_groups_val,
                    bias_demos.subdom_groups_test,
                    expert_train.mu,
                    expert_val.mu,
                    expert_test.mu,
                    exp_info,
                    can_observe_y,
                )

                results.record(evaluate_policy_result, policy_start, run)

                trial_runtime = (datetime.datetime.now() - trial_start).total_seconds()

                _finalize_trial(
                    run,
                    results,
                    feat_obj_set,
                    perf_obj_set,
                    expert_train,
                    expert_val,
                    expert_test,
                    clf_pol,
                    trial_runtime,
                )


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

    # Now compute the same feature expectations using Fairlearn's metric functions
    # obj_names = (
    #     exp_info["SUBDOMINANCE_PERF_METRICS_LIST"]
    #     + exp_info["SUBDOMINANCE_FAIR_METRICS_LIST"]
    # )
    # demo_feat_exp_fairlearn = np.array(
    #     [FAIRLEARN_OBJ_LOOKUP_BY_NAME[obj_name](demo) for obj_name in obj_names]
    # )
    # These should be the same, but we check to make sure they are. If not, raise an error.
    # assert np.allclose(demo_feat_exp, demo_feat_exp_fairlearn), (
    #     f"Fairlearn-based feature loss {demo_feat_exp_fairlearn} does not match "
    #     f"the original feature loss {demo_feat_exp} for obj_names {obj_names}"
    # )
    return demo_feat_exp


def run_bias_experiment(
    exp_info,
    source_X=None,
    source_y=None,
    source_feature_types=None,
    session_id=None,
):
    """
    Runs experiment for source domain based on the parameters in `exp_info`.

    Parameters
    ----------
    exp_info : dict
        Experiment parameters.
    source_X : pandas.DataFrame, Optional
        The X (including z) columns for the source domain.
    source_y : pandas.Series, Optional
        Just the y column for the source domain.
    source_feature_types : dict<str, array-like>, Optional
        Mapping of column names to their type of feature. Used to when
        constructing sklearn pipelines for the source domain.
    session_id : str, Optional
        Identifies the execution of the experiment script this experiment
        belongs to. One execution calls this function once per experiment, so
        passing the same id to each call is what marks their runs as one set of
        results. Defaults to a fresh id, so calling this on its own still
        produces a self-contained session.

    Reports
    -------
    Every trial is reported to W&B as one run per (bias type, weight
    adjustment) pair, all sharing a `group` unique to this experiment and a
    `SESSION_ID` shared with every other experiment of the same execution. Each
    run's config holds `exp_info`, its history holds the metrics of the policy
    it learned, and its summary holds that policy's results.
    """
    logging.info(f"exp_info: {exp_info}")

    if session_id is None:
        session_id = new_session_id()

    # Shared by every run of this experiment so that its trials, bias types and
    # weight adjustments stay grouped together in the W&B UI. Ending the group with
    # the session id means every group of one execution carries the same
    # visible suffix.
    group = "__".join(
        [
            str(exp_info["EXPERIMENT_NAME"]),
            str(exp_info["EXPERT_ALGO"]),
            session_id,
        ]
    )
    logging.info(f"W&B run group: {group}")

    for trial_i in range(exp_info["N_TRIALS"]):
        # Run trials to learn weights on source domain
        run_experiment_trial(
            exp_info,
            X=source_X,
            y=source_y,
            feature_types=source_feature_types,
            trial_i=trial_i,
            group=group,
            session_id=session_id,
        )
