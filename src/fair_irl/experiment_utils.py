import copy
import datetime
import json
import logging
from functools import partial

import numpy as np
import pandas as pd
import sklearn.base
import optuna
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
from fair_irl.ml.svm import SVM
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
        exp_df_cols.append(f"muE_train_{obj.name}_mean")
        exp_df_cols.append(f"muE_train_{obj.name}_std")

    for obj in feat_obj_set.objectives:
        exp_df_cols.append(f"muE_val_{obj.name}_mean")
        exp_df_cols.append(f"muE_val_{obj.name}_std")

    for obj in feat_obj_set.objectives:
        exp_df_cols.append(f"muE_test_{obj.name}_mean")
        exp_df_cols.append(f"muE_test_{obj.name}_std")

    for obj in feat_obj_set.objectives:
        exp_df_cols.append(f"muE_train_unbiased_{obj.name}_mean")
        exp_df_cols.append(f"muE_train_unbiased_{obj.name}_std")

    for obj in feat_obj_set.objectives:
        exp_df_cols.append(f"muE_val_unbiased_{obj.name}_mean")
        exp_df_cols.append(f"muE_val_unbiased_{obj.name}_std")

    for obj in feat_obj_set.objectives:
        exp_df_cols.append(f"muE_test_unbiased_{obj.name}_mean")
        exp_df_cols.append(f"muE_test_unbiased_{obj.name}_std")

    for obj in feat_obj_set.objectives:
        exp_df_cols.append(f"wL_{obj.name}")

    for obj in feat_obj_set.objectives:
        exp_df_cols.append(f"muL_train_{obj.name}")
        exp_df_cols.append(f"muL_val_{obj.name}")
        exp_df_cols.append(f"muL_test_{obj.name}")

    for obj in feat_obj_set.objectives:
        exp_df_cols.append(f"muL_train_err_{obj.name}")
        exp_df_cols.append(f"muL_val_err_{obj.name}")
        exp_df_cols.append(f"muL_test_err_{obj.name}")

    # Subdominance (Train, Val, Test)
    exp_df_cols.append("max_abs_subdominance_train")
    exp_df_cols.append("sum_abs_subdominance_train")
    exp_df_cols.append("max_rel_subdominance_train")
    exp_df_cols.append("sum_rel_subdominance_train")

    exp_df_cols.append("max_abs_subdominance_val")
    exp_df_cols.append("sum_abs_subdominance_val")
    exp_df_cols.append("max_rel_subdominance_val")
    exp_df_cols.append("sum_rel_subdominance_val")

    exp_df_cols.append("max_abs_subdominance_test")
    exp_df_cols.append("sum_abs_subdominance_test")
    exp_df_cols.append("max_rel_subdominance_test")
    exp_df_cols.append("sum_rel_subdominance_test")

    exp_df_cols.append("svm_margin")
    exp_df_cols.append("muL_train_err_l2")
    exp_df_cols.append("muL_val_err_l2")
    exp_df_cols.append("muL_test_err_l2")
    exp_df_cols.append("mu_delta_abs_l2_train")
    exp_df_cols.append("mu_delta_abs_l2_val")
    exp_df_cols.append("mu_delta_abs_l2_test")

    # Training Errors (Train, Val, Test) for best iteration
    exp_df_cols.append("best_t_train")
    exp_df_cols.append("best_t_val")
    exp_df_cols.append("best_t_test")

    for obj in perf_obj_set.objectives:
        exp_df_cols.append(f"muE_perf_train_{obj.name}")
        exp_df_cols.append(f"muE_perf_train_unbiased_{obj.name}")
        exp_df_cols.append(f"muE_perf_val_{obj.name}")
        exp_df_cols.append(f"muE_perf_val_unbiased_{obj.name}")
        exp_df_cols.append(f"muE_perf_test_{obj.name}")
        exp_df_cols.append(f"muE_perf_test_unbiased_{obj.name}")
        exp_df_cols.append(f"muL_perf_train_{obj.name}")
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
    muE_train,
    muE_train_unbiased,
    muE_val,
    muE_val_unbiased,
    muE_test,
    muE_test_unbiased,
    muE_perf_train,
    muE_perf_train_unbiased,
    muE_perf_val,
    muE_perf_val_unbiased,
    muE_perf_test,
    muE_perf_test_unbiased,
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
    feat_obj_set : fair_irl.irl.fair_irl.ObjectiveSet
        The set of objectives for the feature expectations.
    perf_obj_set : fair_irl.irl.fair_irl.ObjectiveSet
        The set of objectives for the performance measures.
    muE_train : array-like<float>, shape(n_expert_demos, n_objectives)
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

    for i, obj in enumerate(feat_obj_set.objectives):
        muE_mean = np.mean(muE_train[:, i])
        muE_std = np.std(muE_train[:, i])
        result += [muE_mean, muE_std]

    for i, obj in enumerate(feat_obj_set.objectives):
        muE_val_mean = np.mean(muE_val[:, i])
        muE_val_std = np.std(muE_val[:, i])
        result += [muE_val_mean, muE_val_std]

    for i, obj in enumerate(feat_obj_set.objectives):
        muE_test_mean = np.mean(muE_test[:, i])
        muE_test_std = np.std(muE_test[:, i])
        result += [muE_test_mean, muE_test_std]

    # Unbiased feature expectations (train, val, test)
    for i, obj in enumerate(feat_obj_set.objectives):
        muE_unbiased_mean = np.mean(muE_train_unbiased[:, i])
        muE_unbiased_std = np.std(muE_train_unbiased[:, i])
        result += [muE_unbiased_mean, muE_unbiased_std]

    for i, obj in enumerate(feat_obj_set.objectives):
        muE_val_unbiased_mean = np.mean(muE_val_unbiased[:, i])
        muE_val_unbiased_std = np.std(muE_val_unbiased[:, i])
        result += [muE_val_unbiased_mean, muE_val_unbiased_std]

    for i, obj in enumerate(feat_obj_set.objectives):
        muE_test_unbiased_mean = np.mean(muE_test_unbiased[:, i])
        muE_test_unbiased_std = np.std(muE_test_unbiased[:, i])
        result += [muE_test_unbiased_mean, muE_test_unbiased_std]

    best_row = df_irl.loc[best_row_idx]

    for i, obj in enumerate(feat_obj_set.objectives):
        result.append(best_row[f"{obj.name}_weight"])

    for obj in feat_obj_set.objectives:
        result.append(best_row[f"muL_train_{obj.name}"])
        result.append(best_row[f"muL_val_{obj.name}"])
        result.append(best_row[f"muL_test_{obj.name}"])

    for i, obj in enumerate(feat_obj_set.objectives):
        _muL_train_err = best_row[f"muL_train_{obj.name}"] - np.mean(muE_train[:, i])
        _muL_val_err = best_row[f"muL_val_{obj.name}"] - np.mean(muE_val[:, i])
        _muL_test_err = best_row[f"muL_test_{obj.name}"] - np.mean(muE_test[:, i])
        result.append(_muL_train_err)
        result.append(_muL_val_err)
        result.append(_muL_test_err)

    # Subdominance (Train, Val, Test)
    result.append(best_row["max_abs_subdominance_train"])
    result.append(best_row["sum_abs_subdominance_train"])
    result.append(best_row["max_rel_subdominance_train"])
    result.append(best_row["sum_rel_subdominance_train"])

    result.append(best_row["max_abs_subdominance_val"])
    result.append(best_row["sum_abs_subdominance_val"])
    result.append(best_row["max_rel_subdominance_val"])
    result.append(best_row["sum_rel_subdominance_val"])

    result.append(best_row["max_abs_subdominance_test"])
    result.append(best_row["sum_abs_subdominance_test"])
    result.append(best_row["max_rel_subdominance_test"])
    result.append(best_row["sum_rel_subdominance_test"])

    result.append(best_row["svm_margin"])
    result.append(best_row["mu_delta_l2_train"])
    result.append(best_row["mu_delta_l2_val"])
    result.append(best_row["mu_delta_l2_test"])
    result.append(best_row["mu_delta_abs_l2_train"])
    result.append(best_row["mu_delta_abs_l2_val"])
    result.append(best_row["mu_delta_abs_l2_test"])

    # Training Errors (Train, Val, Test)
    result.append(best_row["t_train"])
    result.append(best_row["t_val"])
    result.append(best_row["t_test"])

    # Append performance measures of expert_val, expert_test and best learned policies
    for i, obj in enumerate(perf_obj_set.objectives):
        perf_train_mean = np.mean(muE_perf_train[:, i])
        perf_train_unbiased_mean = np.mean(muE_perf_train_unbiased[:, i])
        perf_val_mean = np.mean(muE_perf_val[:, i])
        perf_val_unbiased_mean = np.mean(muE_perf_val_unbiased[:, i])
        perf_test_mean = np.mean(muE_perf_test[:, i])
        perf_test_unbiased_mean = np.mean(muE_perf_test_unbiased[:, i])
        result.append(perf_train_mean)
        result.append(perf_train_unbiased_mean)
        result.append(perf_val_mean)
        result.append(perf_val_unbiased_mean)
        result.append(perf_test_mean)
        result.append(perf_test_unbiased_mean)
        result.append(best_row[f"muL_perf_train_{obj.name}"])
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
    # Since we are using subdominance as an evaluation metric (not a training objective), we don't need to add regularization.
    # final_subdom += (lamda / 2.0) * (np.linalg.norm(alphas) ** 2)
    return final_subdom


# Helper to compute subdominance for a specific set
def compute_iteration_subdominance(
    exp_info,
    raw_demo_ref,
    clf_demo_cur,
    subdominance_type="all",
):
    # Compute subdominance metric for the learned policy
    raw_demo = raw_demo_ref.copy()
    clf_demo = clf_demo_cur.copy()
    # OLD IMPLEMENTATION: Split the raw and clf demos into subdominance groups, where there are no repeated demos in each group.
    # raw_demos = np.array_split(raw_demo, exp_info["N_SUBDOMINANCE_GROUPS"])
    # clf_demos = np.array_split(clf_demo, exp_info["N_SUBDOMINANCE_GROUPS"])
    # NEW IMPLEMENTATION: Split the raw and clf demos into subdominance groups, where each group contains half of the demos,
    # sampled randomly without replacement. This way, each group contains a large and diverse set of demos, from which we can
    # more accurately and robustly compute the feature losses. This will help to improve the robustness of the subdominance
    # metric, especially when the number of demos is small.
    # ADD NEW IMPLEMENTATION HERE:
    n_demos = len(raw_demo)
    group_size = n_demos // 2
    raw_demos = []
    clf_demos = []
    for _ in range(exp_info["N_SUBDOMINANCE_GROUPS"]):
        group_idx = np.random.choice(n_demos, size=group_size, replace=False)
        raw_demos.append(raw_demo.iloc[group_idx])
        clf_demos.append(clf_demo.iloc[group_idx])

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
    bias_types,
):
    clf = copy.deepcopy(expert_algo_lookup[exp_info["EXPERT_ALGO"]])
    muE, demoE, muE_unbiased, demoE_unbiased = generate_mu_and_demos(
        exp_info,
        X=X,
        y=y,
        feature_types=feature_types,
        clf=clf,
        obj_set=feat_obj_set,
        n_demos=exp_info["N_EXPERT_DEMOS"],
        bias_types=bias_types,
    )
    muE_perf = np.array([perf_obj_set.compute_demo_feature_exp(demoE)])
    muE_perf_unbiased = np.array(
        [perf_obj_set.compute_demo_feature_exp(demoE_unbiased)]
    )
    return (
        muE,
        demoE,
        muE_unbiased,
        demoE_unbiased,
        muE_perf,
        muE_perf_unbiased,
    )


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
    bias_types,
):
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
        (
            muE_train,
            demoE_train,
            muE_train_unbiased,
            demoE_train_unbiased,
            muE_perf_train,
            muE_perf_train_unbiased,
        ) = _generate_expert_demonstrations(
            exp_info,
            X_train,
            y_train,
            feature_types,
            expert_algo_lookup,
            feat_obj_set,
            perf_obj_set,
            bias_types,
        )

        logging.info(
            "Generating expert demonstrations and feature expectations for validation set..."
        )
        (
            muE_val,
            demoE_val,
            muE_val_unbiased,
            demoE_val_unbiased,
            muE_perf_val,
            muE_perf_val_unbiased,
        ) = _generate_expert_demonstrations(
            exp_info,
            X_val,
            y_val,
            feature_types,
            expert_algo_lookup,
            feat_obj_set,
            perf_obj_set,
            bias_types,
        )

        logging.info(
            "Generating expert demonstrations and feature expectations for test set..."
        )
        (
            muE_test,
            demoE_test,
            muE_test_unbiased,
            demoE_test_unbiased,
            muE_perf_test,
            muE_perf_test_unbiased,
        ) = _generate_expert_demonstrations(
            exp_info,
            X_test,
            y_test,
            feature_types,
            expert_algo_lookup,
            feat_obj_set,
            perf_obj_set,
            bias_types,
        )

        if _is_split_acceptable(
            muE_train,
            muE_val,
            muE_test,
            max_muE_cosine_dist_split,
        ):
            break

        logging.info(
            "INFO: Split check failed; retrying train/validation/test split..."
        )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        muE_train,
        demoE_train,
        muE_train_unbiased,
        demoE_train_unbiased,
        muE_perf_train,
        muE_perf_train_unbiased,
        muE_val,
        demoE_val,
        muE_val_unbiased,
        demoE_val_unbiased,
        muE_perf_val,
        muE_perf_val_unbiased,
        muE_test,
        demoE_test,
        muE_test_unbiased,
        demoE_test_unbiased,
        muE_perf_test,
        muE_perf_test_unbiased,
    )


def _generate_initial_negative_examples(
    exp_info,
    X_train,
    y_train,
    feature_types,
    expert_algo_lookup,
    feat_obj_set,
    muE_train,
    bias_types,
):
    muL_train_iters = []
    for non_expert_algo in exp_info["NON_EXPERT_ALGOS"]:
        if non_expert_algo in expert_algo_lookup:
            _muL_train, _, _, _ = generate_mu_and_demos(
                exp_info,
                X=X_train,
                y=y_train,
                feature_types=feature_types,
                clf=copy.deepcopy(expert_algo_lookup[non_expert_algo]),
                obj_set=feat_obj_set,
                n_demos=1,
                bias_types=bias_types,
            )
            muL_train_iters.append(_muL_train[0])
        else:
            muL_train_iters.append(init_muL_from_muE(non_expert_algo, muE_train)[0])

    return np.array(muL_train_iters)


def _build_irl_training_sets(exp_info, muE_train, muL_train_iters, feat_obj_set_cols):
    X_irl_exp = pd.DataFrame(muE_train, columns=feat_obj_set_cols)
    y_irl_exp = pd.Series(np.ones(exp_info["N_EXPERT_DEMOS"]), dtype=int)
    X_irl_learn = pd.DataFrame(muL_train_iters, columns=feat_obj_set_cols)
    y_irl_learn = pd.Series(np.zeros(len(muL_train_iters)), dtype=int)
    return X_irl_exp, y_irl_exp, X_irl_learn, y_irl_learn


def _check_irl_stopping_criteria(
    i,
    exp_info,
    error_variable,
    error_variable_hist,
    weights,
    wi,
):
    """Check if IRL loop should stop. Returns True if done, False to continue."""
    if i >= exp_info["MAX_ITER"] - 1:
        logging.info("\n\t\tReached max iters.")
        return True

    if error_variable < exp_info["EPSILON"] and error_variable <= min(
        error_variable_hist
    ):
        if np.allclose(weights[i][0], 0, atol=1e-5):
            logging.info("\t\tAccuracy weight is zero, continuing")
            return False

        if exp_info["ALLOW_NEG_WEIGHTS"] or np.all(wi > -1e-5):
            logging.info(
                f"\n\t\tError {error_variable:.5f} is less than epsilon {exp_info['EPSILON']:.5f}. Stopping."
            )
            return True

    if len(error_variable_hist) > 1 and error_variable > error_variable_hist[-2]:
        logging.info("\n\t\tError is going back up. Stopping.")
        return True

    if (i - np.argsort(error_variable_hist)[0]) > exp_info[
        "EARLY_STOP_NO_NEW_BEST_ITERS"
    ]:
        logging.info(
            f"\n\t\tNo new best in last {exp_info['EARLY_STOP_NO_NEW_BEST_ITERS']} iterations, so stopping early."
        )
        return True

    if (
        i > 0
        and abs(error_variable_hist[i] - error_variable_hist[i - 1]) < 1e-3
        and np.allclose(weights[i], weights[i - 1], atol=1e-3)
    ):
        logging.info("\t\tStuck in loop")
        logging.info(
            f"WARNING: STUCK IN LOOP DETECTED!!!!!!!!!!!. Error history: {error_variable_hist}"
        )
        return False

    if i > 0 and error_variable_hist[i] == np.inf:
        logging.info("\t\tInfinite error: Treating like stuck in loop")
        return False

    return False


def _compute_irl_errors_and_metrics(
    wi,
    svm,
    muE_train,
    muE_val,
    muE_test,
    muL_train_history,
    muL_val_history,
    muL_test_history,
    exp_info,
):
    """Compute IRL error metrics and subdominance for train/val/test sets."""
    (
        muL_delta_train,
        muL_delta_l2_train,
        muL_delta_abs_l2_train,
        ti_train,
        svm_margin,
    ) = irl_error(
        wi,
        muE_train,
        muL_train_history,
        dot_weights_feat_exp=exp_info["DOT_WEIGHTS_FEAT_EXP"],
        svm_margin=svm.margin(),
    )
    (
        muL_delta_val,
        muL_delta_l2_val,
        muL_delta_abs_l2_val,
        ti_val,
        _,
    ) = irl_error(
        wi,
        muE_val,
        muL_val_history,
        dot_weights_feat_exp=exp_info["DOT_WEIGHTS_FEAT_EXP"],
    )
    (
        muL_delta_test,
        muL_delta_l2_test,
        muL_delta_abs_l2_test,
        ti_test,
        _,
    ) = irl_error(
        wi,
        muE_test,
        muL_test_history,
        dot_weights_feat_exp=exp_info["DOT_WEIGHTS_FEAT_EXP"],
    )

    return (
        muL_delta_train,
        muL_delta_l2_train,
        muL_delta_abs_l2_train,
        ti_train,
        svm_margin,
        muL_delta_val,
        muL_delta_l2_val,
        muL_delta_abs_l2_val,
        ti_val,
        muL_delta_test,
        muL_delta_l2_test,
        muL_delta_abs_l2_test,
        ti_test,
    )


def _irl_find_weights(X_irl_exp, y_irl_exp, X_irl_learn, y_irl_learn, exp_info):
    """Fit SVM on expert/learned feature expectations and return l1-normalized weights."""
    X_irl = pd.concat([X_irl_exp, X_irl_learn], axis=0).reset_index(drop=True)
    y_irl = pd.concat([y_irl_exp, y_irl_learn], axis=0).reset_index(drop=True)
    try:
        allow_pos_weights = not exp_info["ALLOW_NEG_WEIGHTS"]
        svm = SVM(positive_weights_only=allow_pos_weights).fit(X_irl, y_irl)
    except ValueError as e:
        if e.args[0] != "No support vectors found.":
            raise e
        logging.info("\t\tAllowing negative weights.")
        svm = SVM(positive_weights_only=False).fit(X_irl, y_irl)
    wi = svm.weights(norm="l1")
    return wi, svm


def _irl_fit_clf_and_demo_df(feature_types, X_train, y_train):
    """Fit a y|x predictor and build the demo DataFrame used by compute_optimal_policy."""
    clf = sklearn_clf_pipeline(
        feature_types=feature_types,
        clf_inst=RandomForestClassifier(),
    )
    clf.fit(X_train, y_train)
    demo_df = pd.DataFrame(X_train)
    demo_df["y"] = y_train
    return clf, demo_df


def _irl_apply_weight_adjustments(
    wi,
    weight_adjusts,
    feat_obj_set,
    demo_df,
    clf,
    x_cols,
    exp_info,
    X,
    y,
    can_observe_y,
    demoE,
):
    """Apply a sequence of weight adjustment operations to wi and return the result."""
    for weight_adjust in weight_adjusts:
        if weight_adjust[0] == "mul_negative_weights":
            mul_factor = weight_adjust[1]
            wi[wi < 0] = wi[wi < 0] * mul_factor
        elif weight_adjust[0] == "opt_debias":
            library = weight_adjust[1]
            optimizer = weight_adjust[2]
            # TODO: Make n_steps configurable via exp_info
            n_steps = 5
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
                    demoE=demoE,
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
                unnormalized_wi = np.array([study.best_params[f"unnormalized_w{j}"] for j in range(n_weights)])
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
                    demoE=demoE,
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
                        maxfun=n_steps
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
                    demoE=demoE,
                )
                n_weights = len(feat_obj_set.objectives)
                parametrization = ng.p.Array(shape=(n_weights,)).set_bounds(-1.0, 1.0)
                parametrization.random_state = np.random.RandomState(exp_info["RANDOM_SEED"])
                if optimizer == "BayesOpt":
                    ng_optimizer = ng.optimizers.BO(parametrization=parametrization, budget=n_steps)
                recommendation = ng_optimizer.minimize(objective)
                unnormalized_wi = np.array(recommendation.value)
            wi = normalize(unnormalized_wi.reshape(1, -1), norm="l1").flatten()
        wi = normalize(wi.reshape(1, -1), norm="l1").flatten()
    return wi


def optuna_objective(
    trial, feat_obj_set, demo_df, clf, x_cols, exp_info, X, y, can_observe_y, demoE
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
        demoE,
    )
    return subdominance_loss


def pybobyqa_objective(
    unnormalized_weights, feat_obj_set, demo_df, clf, x_cols, exp_info, X, y, can_observe_y, demoE
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
        demoE,
    )
    return subdominance_loss


def nevergrad_objective(
    unnormalized_weights, feat_obj_set, demo_df, clf, x_cols, exp_info, X, y, can_observe_y, demoE
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
        demoE,
    )
    return subdominance_loss


def subdominance_of_weights(
    unnormalized_weights, feat_obj_set, demo_df, clf, x_cols, exp_info, X, y, can_observe_y, demoE
):
    """Computes the subdominance loss for a given set of weights. Uses the weights to
    compute the optimal policy, uses that policy to generate demos, uses the demos to
    compute feature expectations, and then computes the subdominance of the generated
    demos against the expert demos. Returns the subdominance loss metric for that weight set.
    """
    # Start by L1 normalizing the weights to ensure they sum to 1
    weights = normalize(unnormalized_weights.reshape(1, -1), norm="l1").flatten()
    # Compute the optimal policy for the given weights
    clf_pol = _irl_compute_optimal_policy(
        weights, feat_obj_set, demo_df, clf, x_cols, exp_info
    )
    # Generate demos from the optimal policy
    demo = generate_demo(clf_pol, X, y, can_observe_y=can_observe_y)
    # Compute the subdominance of the generated demos against the expert demos
    # lp = LineProfiler()
    # lp.add_function(compute_iteration_subdominance)
    # lp.enable_by_count()
    sum_abs_subdom = compute_iteration_subdominance(
        exp_info, demoE, demo, subdominance_type="sum_abs"
    )
    # lp.disable_by_count()
    # lp.print_stats()
    return sum_abs_subdom


def _irl_compute_optimal_policy(wi, feat_obj_set, demo_df, clf, x_cols, exp_info):
    """Build reward weights from wi and return the corresponding optimal classifier policy."""
    reward_weights = {obj.name: wi[j] for j, obj in enumerate(feat_obj_set.objectives)}
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
    return clf_pol


def _irl_evaluate_policy(
    clf_pol,
    wi,
    svm,
    feat_obj_set,
    perf_obj_set,
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    demoE_train,
    demoE_val,
    demoE_test,
    muE_train,
    muE_val,
    muE_test,
    muL_train_history,
    muL_val_history,
    muL_test_history,
    exp_info,
    can_observe_y,
):
    """Generate demos from clf_pol, evaluate errors via _compute_irl_errors_and_metrics,
    and compute subdominance via compute_iteration_subdominance for train/val/test sets.

    muL_train_history, muL_val_history, muL_test_history should NOT yet contain the
    current iteration's muL values; this function appends them internally before calling
    _compute_irl_errors_and_metrics without mutating the caller's lists.
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
        svm_margin,
        muL_delta_val,
        muL_delta_l2_val,
        muL_delta_abs_l2_val,
        ti_val,
        muL_delta_test,
        muL_delta_l2_test,
        muL_delta_abs_l2_test,
        ti_test,
    ) = _compute_irl_errors_and_metrics(
        wi,
        svm,
        muE_train,
        muE_val,
        muE_test,
        muL_train_history + [muL_train],
        muL_val_history + [muL_val],
        muL_test_history + [muL_test],
        exp_info,
    )

    (
        max_abs_subdominance,
        sum_abs_subdominance,
        max_rel_subdominance,
        sum_rel_subdominance,
    ) = compute_iteration_subdominance(exp_info, demoE_train, demo_train)

    (
        max_abs_subdom_v,
        sum_abs_subdom_v,
        max_rel_subdom_v,
        sum_rel_subdom_v,
    ) = compute_iteration_subdominance(exp_info, demoE_val, demo_val)

    (
        max_abs_subdom_t,
        sum_abs_subdom_t,
        max_rel_subdom_t,
        sum_rel_subdom_t,
    ) = compute_iteration_subdominance(exp_info, demoE_test, demo_test)

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
        svm_margin,
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


def _build_df_irl(
    X_irl_exp,
    y_irl_exp,
    X_irl_learn,
    y_irl_learn,
    feat_obj_set_cols,
    perf_obj_set_cols,
    n_expert_demos,
    n_init_policies,
    muL_val_history,
    muL_test_history,
    muL_perf_train_history,
    muL_perf_val_history,
    muL_perf_test_history,
    weights,
    t_train,
    t_val,
    t_test,
    svm_margin_hist,
    subdom_train_hists,
    subdom_val_hists,
    subdom_test_hists,
    muL_delta_l2_train_hist,
    muL_delta_l2_val_hist,
    muL_delta_l2_test_hist,
    muL_delta_abs_l2_train_hist,
    muL_delta_abs_l2_val_hist,
    muL_delta_abs_l2_test_hist,
):
    """Build the df_irl results DataFrame from IRL loop history.

    subdom_train/val/test_hists are each a 4-tuple of
    (max_abs, sum_abs, max_rel, sum_rel) subdominance history lists.
    """
    n_prefix = n_expert_demos + n_init_policies
    n_iter = len(t_train)

    X_irl = pd.concat([X_irl_exp, X_irl_learn], axis=0).reset_index(drop=True)
    y_irl = pd.concat([y_irl_exp, y_irl_learn], axis=0).reset_index(drop=True)
    df_irl = X_irl.copy()
    df_irl["is_expert"] = y_irl.copy()

    for i, col in enumerate(feat_obj_set_cols):
        df_irl[f"muL_val_{col}"] = [0.0] * n_prefix + np.array(muL_val_history)[
            :, i
        ].tolist()
        df_irl[f"muL_test_{col}"] = [0.0] * n_prefix + np.array(muL_test_history)[
            :, i
        ].tolist()

    new_columns = df_irl.columns.tolist()
    for i, col in enumerate(feat_obj_set_cols):
        new_columns[i] = f"muL_train_{col}"
    df_irl.columns = new_columns

    for i, col in enumerate(perf_obj_set_cols):
        df_irl[f"muL_perf_train_{col}"] = [0.0] * n_prefix + np.array(
            muL_perf_train_history
        )[:, i].tolist()
        df_irl[f"muL_perf_val_{col}"] = [0.0] * n_prefix + np.array(
            muL_perf_val_history
        )[:, i].tolist()
        df_irl[f"muL_perf_test_{col}"] = [0.0] * n_prefix + np.array(
            muL_perf_test_history
        )[:, i].tolist()

    df_irl["is_init_policy"] = (
        [0.0] * n_expert_demos + [1.0] * n_init_policies + [0.0] * n_iter
    )
    df_irl["learn_idx"] = [-1.0] * n_expert_demos + list(
        np.arange(n_init_policies + n_iter)
    )

    for i, col in enumerate(feat_obj_set_cols):
        df_irl[f"{col}_weight"] = [0.0] * n_prefix + [w[i] for w in weights]

    (
        max_abs_subdom_hist,
        sum_abs_subdom_hist,
        max_rel_subdom_hist,
        sum_rel_subdom_hist,
    ) = subdom_train_hists
    (
        max_abs_subdom_val_hist,
        sum_abs_subdom_val_hist,
        max_rel_subdom_val_hist,
        sum_rel_subdom_val_hist,
    ) = subdom_val_hists
    (
        max_abs_subdom_test_hist,
        sum_abs_subdom_test_hist,
        max_rel_subdom_test_hist,
        sum_rel_subdom_test_hist,
    ) = subdom_test_hists

    df_irl["max_abs_subdominance_train"] = [np.inf] * n_prefix + max_abs_subdom_hist
    df_irl["sum_abs_subdominance_train"] = [np.inf] * n_prefix + sum_abs_subdom_hist
    df_irl["max_rel_subdominance_train"] = [np.inf] * n_prefix + max_rel_subdom_hist
    df_irl["sum_rel_subdominance_train"] = [np.inf] * n_prefix + sum_rel_subdom_hist

    df_irl["max_abs_subdominance_val"] = [np.inf] * n_prefix + max_abs_subdom_val_hist
    df_irl["sum_abs_subdominance_val"] = [np.inf] * n_prefix + sum_abs_subdom_val_hist
    df_irl["max_rel_subdominance_val"] = [np.inf] * n_prefix + max_rel_subdom_val_hist
    df_irl["sum_rel_subdominance_val"] = [np.inf] * n_prefix + sum_rel_subdom_val_hist

    df_irl["max_abs_subdominance_test"] = [np.inf] * n_prefix + max_abs_subdom_test_hist
    df_irl["sum_abs_subdominance_test"] = [np.inf] * n_prefix + sum_abs_subdom_test_hist
    df_irl["max_rel_subdominance_test"] = [np.inf] * n_prefix + max_rel_subdom_test_hist
    df_irl["sum_rel_subdominance_test"] = [np.inf] * n_prefix + sum_rel_subdom_test_hist

    df_irl["svm_margin"] = [np.inf] * n_prefix + svm_margin_hist
    df_irl["t_train"] = [np.inf] * n_prefix + t_train
    df_irl["t_val"] = [np.inf] * n_prefix + t_val
    df_irl["t_test"] = [np.inf] * n_prefix + t_test
    df_irl["mu_delta_l2_train"] = [np.inf] * n_prefix + muL_delta_l2_train_hist
    df_irl["mu_delta_l2_val"] = [np.inf] * n_prefix + muL_delta_l2_val_hist
    df_irl["mu_delta_l2_test"] = [np.inf] * n_prefix + muL_delta_l2_test_hist
    df_irl["mu_delta_abs_l2_train"] = [np.inf] * n_prefix + muL_delta_abs_l2_train_hist
    df_irl["mu_delta_abs_l2_val"] = [np.inf] * n_prefix + muL_delta_abs_l2_val_hist
    df_irl["mu_delta_abs_l2_test"] = [np.inf] * n_prefix + muL_delta_abs_l2_test_hist

    return df_irl


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

    Returns
    -------
    return_vals : list of tuple
        One tuple per entry in exp_info["WEIGHT_ADJUSTS_LIST"]. Each tuple
        contains (best_row_idx, muE_train, muE_train_unbiased, muE_val,
        muE_val_unbiased, muE_test, muE_test_unbiased, muE_perf_train,
        muE_perf_train_unbiased, muE_perf_val, muE_perf_val_unbiased,
        muE_perf_test, muE_perf_test_unbiased, df_irl, clf_pol,
        avg_runtime_per_irl_loop). All values are None if the IRL loop did
        not converge within exp_info["IGNORE_RESULTS_EPSILON"].

    """
    weight_adjusts_list = exp_info["WEIGHT_ADJUSTS_LIST"]
    n_init_policies = len(exp_info["NON_EXPERT_ALGOS"])

    feat_obj_set, perf_obj_set = _build_objective_sets(exp_info)
    can_observe_y = "FO" in exp_info["IRL_METHOD"]
    X, y, feature_types = _load_or_generate_dataset(exp_info, X, y, feature_types)
    bias_types = exp_info["BIAS_TYPES"]

    expert_algo_lookup = generate_expert_algo_lookup(feature_types)

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        muE_train,
        demoE_train,
        muE_train_unbiased,
        demoE_train_unbiased,
        muE_perf_train,
        muE_perf_train_unbiased,
        muE_val,
        demoE_val,
        muE_val_unbiased,
        demoE_val_unbiased,
        muE_perf_val,
        muE_perf_val_unbiased,
        muE_test,
        demoE_test,
        muE_test_unbiased,
        demoE_test_unbiased,
        muE_perf_test,
        muE_perf_test_unbiased,
    ) = _split_dataset_and_generate_expert_demos(
        exp_info,
        expert_algo_lookup,
        feat_obj_set,
        perf_obj_set,
        X,
        y,
        feature_types,
        bias_types,
    )

    logging.info(f"muE_train:\n{muE_train}")
    logging.info(f"muE_val:\n{muE_val}")
    logging.info(f"muE_test:\n{muE_test}")
    logging.info(f"muE_perf_train:\n{muE_perf_train}")
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
        feature_types["boolean"]
        + feature_types["categoric"]
        + feature_types["continuous"]
    )
    x_cols.remove("z")
    feat_obj_set_cols = [obj.name for obj in feat_obj_set.objectives]
    perf_obj_set_cols = [obj.name for obj in perf_obj_set.objectives]

    # Set aggregator variables for the main IRL while loop
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

    svm_margin_hist = []
    t_train = []  # Errors for each iteration
    t_val = []  # Errors on validation set for each iteration
    t_test = []  # Errors on test set for each iteration
    muL_delta_l2_train_hist = []
    muL_delta_l2_val_hist = []
    muL_delta_l2_test_hist = []
    muL_delta_abs_l2_train_hist = []
    muL_delta_abs_l2_val_hist = []
    muL_delta_abs_l2_test_hist = []
    weights = []
    i = 0
    demo_train_history = []
    demo_val_history = []
    demo_test_history = []
    muL_train_history = []
    muL_val_history = []
    muL_test_history = []
    muL_perf_train_history = []
    muL_perf_val_history = []
    muL_perf_test_history = []

    muL_train_iters = _generate_initial_negative_examples(
        exp_info,
        X_train,
        y_train,
        feature_types,
        expert_algo_lookup,
        feat_obj_set,
        muE_train,
        bias_types,
    )

    logging.info(f"muL_train:\n{muL_train_iters}")
    X_irl_exp, y_irl_exp, X_irl_learn, y_irl_learn = _build_irl_training_sets(
        exp_info, muE_train, muL_train_iters, feat_obj_set_cols
    )

    logging.debug("")
    logging.debug("Starting IRL Loop ...")

    done = False  # When true, breaks IRL loop
    num_loops = 0
    return_vals = []
    unadjusted_best_weight = None

    # Context passed to _irl_apply_weight_adjustments so that methods such as
    # "ml_debias" have access to the dataset and objective sets.
    _irl_context = {
        "exp_info": exp_info,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "feature_types": feature_types,
        "feat_obj_set": feat_obj_set,
        "perf_obj_set": perf_obj_set,
        "demoE_train": demoE_train,
    }

    all_irl_loop_start = datetime.datetime.now()
    for weight_adjusts in weight_adjusts_list:
        while not done:
            this_irl_loop_start = datetime.datetime.now()
            num_loops += 1
            if weight_adjusts == ():
                logging.info(f"\tIRL Loop iteration {i+1}/{exp_info['MAX_ITER']} ...")

                logging.debug("\tFitting SVM classifier...")
                wi, svm = _irl_find_weights(
                    X_irl_exp, y_irl_exp, X_irl_learn, y_irl_learn, exp_info
                )

                ##
                # Learn a policy (clf_pol) from the reward (SVM) weights.
                ##

                logging.debug("\tFitting `y|x` predictor for clf policy...")
                clf, demo_df = _irl_fit_clf_and_demo_df(feature_types, X_train, y_train)
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

                svm_margin_hist = []
                t_train = []  # Errors for each iteration
                t_val = []  # Errors on validation set for each iteration
                t_test = []  # Errors on test set for each iteration
                muL_delta_l2_train_hist = []
                muL_delta_l2_val_hist = []
                muL_delta_l2_test_hist = []
                muL_delta_abs_l2_train_hist = []
                muL_delta_abs_l2_val_hist = []
                muL_delta_abs_l2_test_hist = []
                weights = []
                i = 0
                demo_train_history = []
                demo_val_history = []
                demo_test_history = []
                muL_train_history = []
                muL_val_history = []
                muL_test_history = []
                muL_perf_train_history = []
                muL_perf_val_history = []
                muL_perf_test_history = []
                X_irl_exp = pd.DataFrame(muE_train, columns=feat_obj_set_cols)
                y_irl_exp = pd.Series(np.ones(exp_info["N_EXPERT_DEMOS"]), dtype=int)
                X_irl_learn = pd.DataFrame(muL_train_iters, columns=feat_obj_set_cols)
                y_irl_learn = pd.Series(np.zeros(len(muL_train_iters)), dtype=int)
                wi = _irl_apply_weight_adjustments(
                    unadjusted_best_weight.copy(),
                    weight_adjusts,
                    feat_obj_set,
                    demo_df,
                    clf,
                    x_cols,
                    exp_info,
                    X_train,  # could switch to X_val
                    y_train,  # could switch to y_val
                    can_observe_y,
                    demoE_train,  # could switch to demoE_val
                )
                done = True
            # Learn a policy that maximizes the reward function.
            weights.append(wi)

            clf_pol = _irl_compute_optimal_policy(
                wi, feat_obj_set, demo_df, clf, x_cols, exp_info
            )

            ##
            # Measure and record the error of the learned policy, and keep it as
            # a negative training example for next IRL Loop iteration.
            ##

            (
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
                svm_margin,
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
            ) = _irl_evaluate_policy(
                clf_pol,
                wi,
                svm,
                feat_obj_set,
                perf_obj_set,
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
                demoE_train,
                demoE_val,
                demoE_test,
                muE_train,
                muE_val,
                muE_test,
                muL_train_history,
                muL_val_history,
                muL_test_history,
                exp_info,
                can_observe_y,
            )

            demo_train_history.append(demo_train)
            demo_val_history.append(demo_val)
            demo_test_history.append(demo_test)
            muL_train_history.append(muL_train)
            muL_val_history.append(muL_val)
            muL_test_history.append(muL_test)
            muL_perf_train_history.append(muL_perf_train)
            muL_perf_val_history.append(muL_perf_val)
            muL_perf_test_history.append(muL_perf_test)

            logging.info(
                f"\t\t muL_train[{i}] \t\t= {str(np.round(muL_train, 2)).replace('0.', '.')}"
            )
            logging.debug(f"\t\t muL_val[{i}] = {np.round(muL_val, 2)}")
            logging.debug(f"\t\t muL_test[{i}] = {np.round(muL_test, 2)}")

            # Append policy's feature expectations to irl clf dataset
            X_irl_learn_i = pd.DataFrame(
                np.array([muL_train]), columns=feat_obj_set_cols
            )
            y_irl_learn_i = pd.Series(np.zeros(1), dtype=int)
            X_irl_learn = pd.concat([X_irl_learn, X_irl_learn_i], axis=0)
            y_irl_learn = pd.concat([y_irl_learn, y_irl_learn_i], axis=0)

            max_abs_subdom_hist.append(max_abs_subdominance)
            sum_abs_subdom_hist.append(sum_abs_subdominance)
            max_rel_subdom_hist.append(max_rel_subdominance)
            sum_rel_subdom_hist.append(sum_rel_subdominance)

            max_abs_subdom_val_hist.append(max_abs_subdom_v)
            sum_abs_subdom_val_hist.append(sum_abs_subdom_v)
            max_rel_subdom_val_hist.append(max_rel_subdom_v)
            sum_rel_subdom_val_hist.append(sum_rel_subdom_v)

            max_abs_subdom_test_hist.append(max_abs_subdom_t)
            sum_abs_subdom_test_hist.append(sum_abs_subdom_t)
            max_rel_subdom_test_hist.append(max_rel_subdom_t)
            sum_rel_subdom_test_hist.append(sum_rel_subdom_t)

            svm_margin_hist.append(svm_margin)
            t_train.append(ti_train)
            t_val.append(ti_val)
            t_test.append(ti_test)
            muL_delta_l2_train_hist.append(muL_delta_l2_train)
            muL_delta_l2_val_hist.append(muL_delta_l2_val)
            muL_delta_l2_test_hist.append(muL_delta_l2_test)
            muL_delta_abs_l2_train_hist.append(muL_delta_abs_l2_train)
            muL_delta_abs_l2_val_hist.append(muL_delta_abs_l2_val)
            muL_delta_abs_l2_test_hist.append(muL_delta_abs_l2_test)

            # CHOOSE YOUR ERROR VARIABLE HERE:
            error_variable_hist = svm_margin_hist

            error_variable = error_variable_hist[-1]

            logging.info(
                f"\t\t svm_margin[{i}] (innacurate for weight adjust iters) \t= {svm_margin:.10f}"
            )
            logging.info(f"\t\t t_train[{i}] \t\t= {t_train[i]:.5f}")
            logging.info(f"\t\t muL_delta_l2_train[{i}] \t= {muL_delta_l2_train:.5f}")
            logging.info(
                f"\t\t muL_delta_abs_l2_train[{i}] \t= {muL_delta_abs_l2_train:.5f}"
            )
            logging.info(
                f"\t\t muL_delta_train[{i}] \t= {str(np.round(muL_delta_train, 2)).replace('0.', '.')}"
            )
            logging.info(
                f"\t\t weights[{i}] \t= {str(np.round(weights[i], 2)).replace('0.', '.')}"
            )

            runtime_this_irl_loop = (
                datetime.datetime.now() - this_irl_loop_start
            ).total_seconds()
            logging.info(f"\t\t Runtime for this IRL loop: {runtime_this_irl_loop}")

            if weight_adjusts == ():
                should_stop = _check_irl_stopping_criteria(
                    i, exp_info, error_variable, error_variable_hist, weights, wi
                )
                if should_stop:
                    done = True
                    break
                else:
                    i += 1
        done = False

        avg_runtime_per_irl_loop = (
            datetime.datetime.now() - all_irl_loop_start
        ).total_seconds() / num_loops

        if weight_adjusts == ():
            logging.info(
                "IRL Loop finished. Now proceeding with weight adjustment iterations (if any) ..."
            )

        # If solution not sufficient for use, exit early
        if min(muL_delta_l2_train_hist) > exp_info["IGNORE_RESULTS_EPSILON"]:
            logging.info(
                f"IGNORING RESULTS BECAUSE BEST ERROR {min(muL_delta_l2_train_hist):.3f} > {exp_info['IGNORE_RESULTS_EPSILON']:.3f}"
            )
            return_val = (
                None,  # best_row_idx,
                None,  # muE_train,
                None,  # muE_train_unbiased,
                None,  # muE_val,
                None,  # muE_val_unbiased,
                None,  # muE_test,
                None,  # muE_test_unbiased,
                None,  # muE_perf_train,
                None,  # muE_perf_train_unbiased,
                None,  # muE_perf_val,
                None,  # muE_perf_val_unbiased,
                None,  # muE_perf_test,
                None,  # muE_perf_test_unbiased,
                None,  # df_irl,
                None,  # clf_pol,
                None,  # avg_runtime_per_irl_loop,
            )
            return_vals.append(return_val)
            return return_vals

        best_iter = np.argsort(error_variable_hist)[0]

        if not exp_info["ALLOW_NEG_WEIGHTS"]:
            raise ValueError(
                "In order to disable negative weights, you need to understand/fix this part of the code"
            )

        ##
        # Book keeping stuff for the trial.
        ##

        # Compare the best learned policy with the expert demonstrations
        best_demo = demo_train_history[best_iter]
        best_weight = weights[best_iter]
        if weight_adjusts == ():
            unadjusted_best_weight = weights[best_iter].copy()
        logging.debug("Best iteration: " + str(best_iter))
        logging.info(
            f"Best Learned Policy yhat (not real yhat since doesn't factor mu0): {best_demo['yhat'].mean():.3f}"
        )
        logging.info(f"best weight:\t {np.round(best_weight, 3)}")

        best_row_idx = exp_info["N_EXPERT_DEMOS"] + n_init_policies + best_iter

        df_irl = _build_df_irl(
            X_irl_exp,
            y_irl_exp,
            X_irl_learn,
            y_irl_learn,
            feat_obj_set_cols,
            perf_obj_set_cols,
            exp_info["N_EXPERT_DEMOS"],
            n_init_policies,
            muL_val_history,
            muL_test_history,
            muL_perf_train_history,
            muL_perf_val_history,
            muL_perf_test_history,
            weights,
            t_train,
            t_val,
            t_test,
            svm_margin_hist,
            (
                max_abs_subdom_hist,
                sum_abs_subdom_hist,
                max_rel_subdom_hist,
                sum_rel_subdom_hist,
            ),
            (
                max_abs_subdom_val_hist,
                sum_abs_subdom_val_hist,
                max_rel_subdom_val_hist,
                sum_rel_subdom_val_hist,
            ),
            (
                max_abs_subdom_test_hist,
                sum_abs_subdom_test_hist,
                max_rel_subdom_test_hist,
                sum_rel_subdom_test_hist,
            ),
            muL_delta_l2_train_hist,
            muL_delta_l2_val_hist,
            muL_delta_l2_test_hist,
            muL_delta_abs_l2_train_hist,
            muL_delta_abs_l2_val_hist,
            muL_delta_abs_l2_test_hist,
        )

        logging.debug("Experiment Summary")

        # End regular IRL trial
        return_val = (
            best_row_idx,
            muE_train,
            muE_train_unbiased,
            muE_val,
            muE_val_unbiased,
            muE_test,
            muE_test_unbiased,
            muE_perf_train,
            muE_perf_train_unbiased,
            muE_perf_val,
            muE_perf_val_unbiased,
            muE_perf_test,
            muE_perf_test_unbiased,
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
    source_clf_pol : fair_irl.rl.clf_mdp.ClassificationMDPPolicy
        The classification MDP optimal policy for the source domain. This is
        returned only for debugging or inpsection purposes.
    target_clf_pol : fair_irl.rl.clf_mdp.ClassificationMDPPolicy
        The classification MDP optimal policy for the target domain. This is
        returned only for debugging or inpsection purposes.

    Persists
    --------
    exp_df : pandas.DataFrame
        Saves experiment results as a CSV where each row in the CSV represents
        the relevant results of one trial. The file is stored as
        "./experiment_output/fair_irl/exp_results/{timestamp}.csv"
    exp_info : dict
        Saves the experiment parameters and metadata metadata as a JSON file
        "./experiment_output/fair_irl/exp_info/{timestamp}.json"
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
                muE_train,
                muE_train_unbiased,
                muE_val,
                muE_val_unbiased,
                muE_test,
                muE_test_unbiased,
                muE_perf_train,
                muE_perf_train_unbiased,
                muE_perf_val,
                muE_perf_val_unbiased,
                muE_perf_test,
                muE_perf_test_unbiased,
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
                muE_train,
                muE_train_unbiased,
                muE_val,
                muE_val_unbiased,
                muE_test,
                muE_test_unbiased,
                muE_perf_train,
                muE_perf_train_unbiased,
                muE_perf_val,
                muE_perf_val_unbiased,
                muE_perf_test,
                muE_perf_test_unbiased,
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
                f"./experiment_output/fair_irl/exp_conv_details/{timestamp}_{weight_adjust_names[i]}_trial{trial_i}.csv",
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
            f"./experiment_output/fair_irl/exp_results/{timestamp}_{weight_adjust_names[i]}.csv",
            index=None,
        )

        # Persist trial info
        exp_info["timestamp"] = timestamp
        exp_info["WEIGHT_ADJUSTS"] = weight_adjusts
        fp = f"./experiment_output/fair_irl/exp_info/{timestamp}_{weight_adjust_names[i]}.json"
        json.dump(exp_info, open(fp, "w"))
