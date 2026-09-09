import copy
import datetime
import hashlib
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
from .sh.superhuman_fairness import (
    SH_DEFAULT_ITERS,
    SH_DEFAULT_LAMDA,
    SH_DEFAULT_LOGI_PARAMS,
    SH_DEFAULT_LR_THETA,
    SH_DEFAULT_NUM_OF_DEMOS,
    SuperhumanFairness,
    build_expert_demo_list,
    build_pp_demo_list,
    compute_alphas,
    make_feature_loss_fn,
    objective_feature_losses,
)

# `OBJ_LOOKUP_BY_NAME` now lives in `fair_irl.rl.objectives`, next to the
# objectives it names, so that modules which cannot import this one (e.g.
# `fair_irl.sh.superhuman_fairness`) can still resolve objective names. It is
# star-imported above and so is still available as
# `experiment_utils.OBJ_LOOKUP_BY_NAME`.


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

# The techniques `run_experiment_trial()` can train and evaluate. Which of them
# a run of the experiment script actually covers is set per experiment, as
# `exp_info["ALGORITHMS"]`.
ALGORITHM_FAIRIRL = "FairIRL Bias Reduction"
ALGORITHM_SUPERHUMAN = "Superhuman Fairness"
ALGORITHMS = (ALGORITHM_FAIRIRL, ALGORITHM_SUPERHUMAN)

# Defaults for the Superhuman Fairness baseline's own configuration. Each is
# overridable through the correspondingly named `exp_info` key; see
# `_superhuman_config()`.
SH_DEFAULTS = {
    # "expert_demos": imitate this project's expert, on the same demonstration
    #   groups the subdominance metric scores every policy against.
    # "pp_baseline": imitate the original paper's own demonstrator, a logistic
    #   regression post-processed by a fairlearn ThresholdOptimizer.
    "SH_DEMO_SOURCE": "expert_demos",
    # The performance/fairness measures the baseline optimizes, as objective
    # names (e.g. "Acc", "DemPar") or the original paper's metric names (e.g.
    # "inacc", "dp"). `None` uses this experiment's subdominance metrics, so
    # that the baseline optimizes exactly what both techniques are scored on.
    "SH_FEATURES": None,
    # How many demonstrations to imitate. `None` uses every group the
    # subdominance metric already sampled ("expert_demos"), or the original's
    # 50 ("pp_baseline").
    "SH_NUM_DEMOS": None,
    "SH_ITERS": SH_DEFAULT_ITERS,
    "SH_LR_THETA": SH_DEFAULT_LR_THETA,
    "SH_LAMDA": SH_DEFAULT_LAMDA,
    # The fairness constraint the "pp_baseline" demonstrator satisfies. Unused
    # by the "expert_demos" source.
    "SH_DEMO_CONSTRAINTS": "demographic_parity",
}


def algorithm_slug(algorithm):
    """
    Build the short, filename-safe label of an algorithm.

    Used as part of the W&B run name and as a run tag, so that the runs of the
    two techniques can be told apart at a glance.
    """
    return str(algorithm).lower().replace(" ", "_")


def selected_algorithms(exp_info):
    """
    The algorithms one experiment covers, validated and in a fixed order.

    Defaults to just the FairIRL Bias Reduction technique, so that an
    `exp_info` written before this baseline existed behaves exactly as it did
    then.

    Returns
    -------
    algorithms : list<str>
        The selected entries of `ALGORITHMS`, always in `ALGORITHMS` order so
        that the FairIRL technique runs first regardless of how the config
        lists them.
    """
    selected = exp_info.get("ALGORITHMS") or [ALGORITHM_FAIRIRL]

    unknown = [name for name in selected if name not in ALGORITHMS]
    if unknown:
        raise ValueError(
            f"Unrecognized entries in exp_info['ALGORITHMS']: {unknown}."
            f" Valid entries are {list(ALGORITHMS)}."
        )

    return [name for name in ALGORITHMS if name in selected]


def _superhuman_config(exp_info):
    """`SH_DEFAULTS`, overridden by whatever `exp_info` sets."""
    return {
        key: exp_info.get(key, default) if exp_info.get(key) is not None else default
        for key, default in SH_DEFAULTS.items()
    }


def _superhuman_rng(exp_info, dataset_bias_type, trial_i):
    """
    Build the random generator the Superhuman Fairness baseline draws from.

    The baseline gets a dedicated generator, seeded deterministically from the
    experiment's own seed and from what makes this run unique, rather than
    drawing from the global numpy random state. That way enabling the baseline
    cannot shift a single random draw the FairIRL technique makes, so a session
    that runs both produces exactly the FairIRL results a session that runs
    only FairIRL would.
    """
    key = "|".join(
        [
            str(exp_info.get("RANDOM_SEED")),
            str(exp_info.get("EXPERIMENT_NAME")),
            str(exp_info.get("DATASET")),
            str(exp_info.get("EXPERT_ALGO")),
            dataset_bias_type_name(dataset_bias_type),
            str(trial_i),
        ]
    )
    seed = int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "big")
    return np.random.default_rng(seed)


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


def dataset_bias_type_name(dataset_bias_type):
    """
    Build a short, human readable label for a bias type configuration.

    Used as part of the W&B run name and as a run tag, so that the runs of the
    bias types one trial covers can be told apart from each other.

    Parameters
    ----------
    dataset_bias_type : tuple
        One entry of `exp_info["DATASET_BIAS_TYPE_LIST"]`, or `()` for the unbiased
        demonstrations.

    Returns
    -------
    name : str
        E.g. `"unbiased"` or `"balanced_redlining_0.2"`.
    """
    if not dataset_bias_type:
        return "unbiased"

    return "_".join(str(component) for component in dataset_bias_type)


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


def start_wandb_run(
    exp_info,
    algorithm,
    dataset_bias_type,
    weight_adjust,
    trial_i,
    group,
    session_id,
):
    """
    Start the W&B run that records one trial of one algorithm, one bias type
    and one weight adjustment.

    Every algorithm reports through this same function, so that the runs of the
    FairIRL Bias Reduction technique and of the Superhuman Fairness baseline
    carry the same config keys, the same metric names and the same summary
    keys, and differ only by their `ALGORITHM`.

    Parameters
    ----------
    exp_info : dict
        Experiment parameters. Logged in full as the run's config, replacing
        the exp_info JSON files this pipeline used to write. It holds the
        listed bias types of the trial as `DATASET_BIAS_TYPE_LIST`; the single one this
        run covers -- which may be the unbiased `()` that is always run and so
        is not listed there -- is recorded separately as `DATASET_BIAS_TYPE`.
    algorithm : str
        The technique this run covers, i.e. one entry of
        `exp_info["ALGORITHMS"]`. Recorded as `ALGORITHM` so that the two
        techniques' runs can be told apart and compared.
    dataset_bias_type : tuple
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
    bias_name = dataset_bias_type_name(dataset_bias_type)
    adjust_name = weight_adjusts_name(weight_adjust)
    algorithm_name = algorithm_slug(algorithm)

    config = {key: _json_safe(value) for key, value in exp_info.items()}
    config["ALGORITHM"] = algorithm
    config["ALGORITHM_NAME"] = algorithm_name
    config["DATASET_BIAS_TYPE"] = _json_safe(dataset_bias_type)
    config["DATASET_BIAS_TYPE_NAME"] = bias_name
    config["WEIGHT_ADJUST"] = _json_safe(weight_adjust)
    config["WEIGHT_ADJUST_NAME"] = adjust_name
    config["TRIAL"] = trial_i
    config["SESSION_ID"] = session_id

    return wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        group=group,
        job_type=adjust_name,
        name=f"{group}__{algorithm_name}__{bias_name}__{adjust_name}__trial{trial_i}",
        config=config,
        tags=[
            str(exp_info["DATASET"]),
            str(exp_info["EXPERT_ALGO"]),
            str(exp_info["IRL_METHOD"]),
            algorithm_name,
            bias_name,
            adjust_name,
        ],
    )


@dataclass
class ExpertDemos:
    """
    The expert demonstrations and feature expectations for one data split.

    Groups everything `_generate_expert_demonstrations()` produces for a split
    so that the train/validation/test expert data can be passed around as three
    objects instead of nine separate arrays. Any bias of the bias type these
    demonstrations belong to is already carried by the labels of the dataset
    they were generated from, so there are no separate "unbiased" counterparts
    here.

    Attributes
    ----------
    mu : array-like<float>, shape(n_expert_demos, n_objectives)
        Feature expectations of the expert demonstrations.
    demo : pandas.DataFrame
        The expert demonstrations themselves.
    mu_perf : array-like<float>, shape(1, n_perf_objectives)
        Performance measures of the expert demonstrations.
    """

    mu: np.ndarray
    demo: pd.DataFrame
    mu_perf: np.ndarray


@dataclass
class BiasedDatasetDemos:
    """
    The biased dataset, data split and expert demonstrations of one bias type.

    `run_experiment_trial()` runs the unbiased dataset and every bias type of
    `exp_info["DATASET_BIAS_TYPE_LIST"]`, each of which is one biased copy of the same
    dataset's labels, split along the same randomized indices. This groups
    everything that is specific to a single one of those bias types, so that
    the trial can iterate over one list instead of threading a dozen parallel
    lists through its loop.

    Attributes
    ----------
    dataset_bias_type : tuple
        The bias type this dataset's labels were biased with; `()` for the
        unbiased dataset.
    X_train, X_val, X_test : pandas.DataFrame
        The input columns of each data split. Identical across bias types,
        since only the labels are biased.
    y_train, y_val, y_test : pandas.Series
        The biased labels of each data split.
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

    dataset_bias_type: tuple
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
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
    bias_demos,
    unbiased_demos,
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
    bias_demos : BiasedDatasetDemos
        The dataset, expert demonstrations and bias type this trial's policy
        was learned from.
    unbiased_demos : BiasedDatasetDemos
        The unbiased dataset and its expert demonstrations, reported alongside
        `bias_demos` so that a biased run's expert can be compared against the
        unbiased expert of the same data split.
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

    expert_train = bias_demos.expert_train
    expert_val = bias_demos.expert_val
    expert_test = bias_demos.expert_test

    expert_mus = (
        ("muE_train", expert_train.mu),
        ("muE_val", expert_val.mu),
        ("muE_test", expert_test.mu),
        ("muE_train_unbiased", unbiased_demos.expert_train.mu),
        ("muE_val_unbiased", unbiased_demos.expert_val.mu),
        ("muE_test_unbiased", unbiased_demos.expert_test.mu),
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
        ("muE_perf_train_unbiased", unbiased_demos.expert_train.mu_perf),
        ("muE_perf_val", expert_val.mu_perf),
        ("muE_perf_val_unbiased", unbiased_demos.expert_val.mu_perf),
        ("muE_perf_test", expert_test.mu_perf),
        ("muE_perf_test_unbiased", unbiased_demos.expert_test.mu_perf),
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
    expert_algo_lookup,
    feat_obj_set,
    perf_obj_set,
):
    """Generate the expert demonstrations for one data split.

    Whatever bias the bias type being run applies is already carried by `y`
    (see `_split_dataset_and_generate_expert_demos()`), so the expert is fit on
    biased labels and its demonstrations are scored against them; no bias is
    added to the demonstrations themselves.

    Returns
    -------
    expert : ExpertDemos
        The demonstrations, feature expectations and performance measures of
        this split.
    """
    clf = copy.deepcopy(expert_algo_lookup[exp_info["EXPERT_ALGO"]])
    muE, demoE = generate_mu_and_demos(
        exp_info,
        X=X,
        y=y,
        clf=clf,
        obj_set=feat_obj_set,
        n_demos=exp_info["N_EXPERT_DEMOS"],
    )
    return ExpertDemos(
        mu=muE,
        demo=demoE,
        mu_perf=np.array([perf_obj_set.compute_demo_feature_exp(demoE)]),
    )


def _is_split_acceptable(muE_train, muE_val, muE_test, max_muE_cosine_dist_split):
    for demo_i in range(len(muE_train)):
        if cosine(muE_train[demo_i], muE_val[0]) > max_muE_cosine_dist_split:
            return False
        if cosine(muE_train[demo_i], muE_test[0]) > max_muE_cosine_dist_split:
            return False
    return True


def _bias_dataset_labels(exp_info, X, y, feature_types, dataset_bias_type):
    """Apply one bias type to a dataset's labels.

    Returns
    -------
    y_biased : pandas.Series
        A copy of `y` with `dataset_bias_type` applied, indexed and named like `y`.
        `()` (the unbiased "bias type") returns the labels unchanged.
    """
    y_biased = add_corruption_bias(
        X, y, feature_types, dataset_bias_type=dataset_bias_type
    )
    y_biased = add_redlining_bias(
        X, y_biased, dataset_bias_type=dataset_bias_type, dataset=exp_info["DATASET"]
    )

    pct_unchanged = (np.asarray(y) == np.asarray(y_biased)).mean() * 100.0
    logging.info(f"Bias type added: {dataset_bias_type}")
    logging.info(f"Percent of y unchanged with added bias: {pct_unchanged}%")

    return y_biased


def _split_dataset_and_generate_expert_demos(
    exp_info,
    expert_algo_lookup,
    feat_obj_set,
    perf_obj_set,
    X,
    y,
    feature_types,
    dataset_bias_type_list,
):
    """Bias the dataset's labels once per bias type, then split each biased
    dataset and generate expert demonstrations from its splits.

    The bias of a bias type is applied to the dataset's `y` values before the
    data is split, so the expert of that bias type is fit on biased labels and
    every metric of that bias type -- the expert's own feature expectations,
    its subdominance groups, and the learned policies scored against them -- is
    computed against the same biased labels.

    Every biased dataset is split along the same randomized indices, so the
    bias types differ from each other only by their labels and their results
    stay comparable.

    Returns
    -------
    demos_by_dataset_bias_type : list<BiasedDatasetDemos>
        The biased data split, expert demonstrations and subdominance groups of
        each bias type, in the order of `dataset_bias_type_list`. Since
        `dataset_bias_type_list` starts with `()`, the first entry always holds the
        unbiased dataset and its demonstrations.
    """
    # One biased copy of the dataset's labels per bias type. `()` leaves the
    # labels untouched, so its entry is the unbiased dataset.
    logging.info("Applying each bias type to the dataset's labels...")
    biased_ys = [
        _bias_dataset_labels(exp_info, X, y, feature_types, dataset_bias_type)
        for dataset_bias_type in dataset_bias_type_list
    ]

    # One randomized split, shared by every bias type: the biased datasets
    # differ from each other only by their labels, so splitting them all along
    # the same indices keeps their results comparable.
    idxs = np.arange(len(X))
    train_idxs, val_test_idxs = train_test_split(idxs, train_size=0.60)
    val_idxs, test_idxs = train_test_split(val_test_idxs, test_size=0.50)

    X_train = X.iloc[train_idxs]
    X_val = X.iloc[val_idxs]
    X_test = X.iloc[test_idxs]

    demos_by_dataset_bias_type = []
    for dataset_bias_type, y_biased in zip(dataset_bias_type_list, biased_ys):
        logging.info(
            f"Generating expert demonstrations for bias type: {dataset_bias_type}"
        )

        y_train = y_biased.iloc[train_idxs]
        y_val = y_biased.iloc[val_idxs]
        y_test = y_biased.iloc[test_idxs]

        logging.info(
            "Generating expert demonstrations and feature expectations for training set..."
        )
        expert_train = _generate_expert_demonstrations(
            exp_info,
            X_train,
            y_train,
            expert_algo_lookup,
            feat_obj_set,
            perf_obj_set,
        )

        logging.info(
            "Generating expert demonstrations and feature expectations for validation set..."
        )
        expert_val = _generate_expert_demonstrations(
            exp_info,
            X_val,
            y_val,
            expert_algo_lookup,
            feat_obj_set,
            perf_obj_set,
        )

        logging.info(
            "Generating expert demonstrations and feature expectations for test set..."
        )
        expert_test = _generate_expert_demonstrations(
            exp_info,
            X_test,
            y_test,
            expert_algo_lookup,
            feat_obj_set,
            perf_obj_set,
        )

        # TODO: Reimplement the split check. It used to retry the split until
        # the feature expectations of the three splits were close enough to
        # each other, which no longer works as-is now that the split is shared
        # by every bias type (and is what the biased datasets are compared
        # along).
        # max_muE_cosine_dist_split = 0.002
        # if not _is_split_acceptable(
        #     expert_train.mu,
        #     expert_val.mu,
        #     expert_test.mu,
        #     max_muE_cosine_dist_split,
        # ):
        #     logging.info(
        #         "INFO: Split check failed; retrying train/validation/test split..."
        #     )

        # Sample the subdominance groups (and compute the expert feature losses
        # on them) once per split and bias type, so that every
        # compute_iteration_subdominance() call of that bias type reuses the
        # same groups.
        logging.info("Generating subdominance groups for the expert demonstrations...")
        demos_by_dataset_bias_type.append(
            BiasedDatasetDemos(
                dataset_bias_type=dataset_bias_type,
                X_train=X_train,
                X_val=X_val,
                X_test=X_test,
                y_train=y_train,
                y_val=y_val,
                y_test=y_test,
                expert_train=expert_train,
                expert_val=expert_val,
                expert_test=expert_test,
                subdom_groups_train=generate_subdominance_groups(
                    exp_info, expert_train.demo
                ),
                subdom_groups_val=generate_subdominance_groups(
                    exp_info, expert_val.demo
                ),
                subdom_groups_test=generate_subdominance_groups(
                    exp_info, expert_test.demo
                ),
            )
        )

    return demos_by_dataset_bias_type


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
    exp_info : dict
        Metadata about the experiment. `OPT_DEBIAS_MAX_ITERATIONS` caps how
        many iterations run before giving up on finding a better weight set.
        That cap is only a safety net -- the loop is expected to stop on its
        own as soon as an iteration fails to improve.
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

    max_iterations = exp_info["OPT_DEBIAS_MAX_ITERATIONS"]

    for iteration in range(max_iterations):
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
            f"\t\t opt_debias stopping: hit the {max_iterations} "
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
    bias_demos,
    unbiased_demos,
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
        bias_demos,
        unbiased_demos,
        results,
        trial_runtime,
        trial_inputsize,
    )
    run.summary["converged"] = True
    run.summary.update(summary)


def _build_superhuman_demo_list(
    exp_info, sh_config, bias_demos, feature_types, loss_fn, rng
):
    """
    Build the reference decisions the Superhuman Fairness baseline imitates,
    together with the training pool they index into.

    Which source is used is set by `exp_info["SH_DEMO_SOURCE"]`:

    `"expert_demos"`
        This project's own expert demonstrations for the training split,
        grouped exactly the way `generate_subdominance_groups()` grouped them
        for the subdominance metric. The baseline then imitates the same
        expert, on the same rows, that the FairIRL technique learns from, which
        is what makes the two directly comparable.
    `"pp_baseline"`
        The original paper's own demonstrator (a logistic regression
        post-processed by a fairlearn `ThresholdOptimizer`), rebuilt on this
        project's training split.

    Parameters
    ----------
    exp_info : dict
        Experiment parameters.
    sh_config : dict
        The resolved baseline configuration, from `_superhuman_config()`.
    bias_demos : BiasedDatasetDemos
        The biased data split, expert demonstrations and subdominance groups of
        the bias type being run.
    feature_types : dict<str, list>
        Mapping of column names to their type of feature. Only the
        `"pp_baseline"` source uses it, to build its demonstrator's pipeline.
    loss_fn : callable(y_true, y_pred, z) -> numpy.ndarray
        The feature losses a demonstration is scored with, from
        `make_feature_loss_fn()`.
    rng : numpy.random.Generator
        The baseline's dedicated source of randomness.

    Returns
    -------
    pool_X : pandas.DataFrame
        The training pool the demonstrations' indices are positions into.
    pool_y : pandas.Series
        That pool's labels.
    demo_list : list<SuperhumanDemo>
    """
    demo_source = sh_config["SH_DEMO_SOURCE"]
    num_demos = sh_config["SH_NUM_DEMOS"]

    if demo_source == "expert_demos":
        # The expert's demonstrations cover every row of the training split
        # exactly once, in their own order, and `subdom_groups_train` holds
        # positional indices into that order.
        expert_demo_df = bias_demos.expert_train.demo
        group_idxs = bias_demos.subdom_groups_train[0]
        if num_demos is not None:
            group_idxs = group_idxs[:num_demos]

        pool_X = expert_demo_df[bias_demos.X_train.columns]
        pool_y = expert_demo_df["y"]
        demo_list = build_expert_demo_list(expert_demo_df, group_idxs, loss_fn)

    elif demo_source == "pp_baseline":
        pool_X = bias_demos.X_train
        pool_y = bias_demos.y_train
        demo_list = build_pp_demo_list(
            pool_X,
            pool_y,
            feature_types=feature_types,
            loss_fn=loss_fn,
            num_of_demos=(
                num_demos if num_demos is not None else SH_DEFAULT_NUM_OF_DEMOS
            ),
            constraints=sh_config["SH_DEMO_CONSTRAINTS"],
            rng=rng,
        )

    else:
        raise ValueError(
            f"Unrecognized exp_info['SH_DEMO_SOURCE']: {demo_source!r}."
            " Valid values are 'expert_demos' and 'pp_baseline'."
        )

    return pool_X, pool_y, demo_list


def _log_superhuman_training(run, sh_model):
    """Report the baseline's per-iteration training curve to its W&B run."""
    if run is None:
        return

    history = sh_model.history_
    for i in range(history.n_iterations):
        metrics = {
            "superhuman/iteration": i,
            "superhuman/subdom_sum": history.subdom_sum[i],
            "superhuman/gamma_superhuman_mean": float(
                np.mean(history.gamma_superhuman[i])
            ),
        }
        for j, name in enumerate(sh_model.feature_names):
            metrics[f"superhuman/loss/{name}"] = history.feature_loss[i][j]
            metrics[f"superhuman/gamma_superhuman/{name}"] = history.gamma_superhuman[
                i
            ][j]
            metrics[f"superhuman/alpha/{name}"] = history.alphas[i][j]
        run.log(metrics)


def _run_superhuman_trial(
    exp_info,
    bias_demos,
    unbiased_demos,
    feat_obj_set,
    perf_obj_set,
    feature_types,
    weights,
    trial_i,
    group,
    session_id,
    trial_start,
):
    """
    Train and evaluate the Superhuman Fairness baseline for one bias type.

    Reports one W&B run, built by the same `start_wandb_run()` and finished by
    the same `_evaluate_policy()`/`PolicyResults`/`_finalize_trial()` the
    FairIRL technique goes through, so that both techniques' runs hold the same
    metrics, computed the same way, on the same train/validation/test splits of
    the same biased dataset.

    Parameters
    ----------
    weights : numpy.ndarray
        The reward weights every metric that needs one is computed with. The
        baseline does not learn reward weights, so it is scored with the same
        equal, positive weights the FairIRL technique starts from; that keeps
        `t_train`/`t_val`/`t_test` and `wL_*` comparable between the two.
    """
    sh_config = _superhuman_config(exp_info)
    feature_names = sh_config["SH_FEATURES"] or subdominance_metric_names(exp_info)
    rng = _superhuman_rng(exp_info, bias_demos.dataset_bias_type, trial_i)

    with start_wandb_run(
        exp_info,
        ALGORITHM_SUPERHUMAN,
        bias_demos.dataset_bias_type,
        (),
        trial_i,
        group,
        session_id,
    ) as run:
        policy_start = datetime.datetime.now()

        run.config.update(
            {f"{key}_RESOLVED": _json_safe(value) for key, value in sh_config.items()}
            | {"SH_FEATURES_RESOLVED": _json_safe(list(feature_names))},
            allow_val_change=True,
        )

        results = PolicyResults(
            [obj.name for obj in feat_obj_set.objectives],
            [obj.name for obj in perf_obj_set.objectives],
        )
        results.weights = weights

        loss_fn = make_feature_loss_fn(feature_names)

        logging.info("Building Superhuman Fairness demonstrations...")
        pool_X, pool_y, demo_list = _build_superhuman_demo_list(
            exp_info, sh_config, bias_demos, feature_types, loss_fn, rng
        )
        logging.info(
            f"\t {len(demo_list)} demonstrations over {len(pool_X)} training rows,"
            f" features {list(feature_names)}"
        )

        logging.info("Training Superhuman Fairness...")
        sh_model = SuperhumanFairness(
            feature_types=feature_types,
            feature_names=feature_names,
            loss_fn=loss_fn,
            lr_theta=sh_config["SH_LR_THETA"],
            iters=sh_config["SH_ITERS"],
            lamda=sh_config["SH_LAMDA"],
            rng=rng,
        )
        sh_model.fit(pool_X, pool_y, demo_list)

        _log_superhuman_training(run, sh_model)

        ##
        # Measure and record the error of the learned classifier. This is the
        # exact same evaluation the FairIRL policies go through.
        ##

        evaluate_policy_result = _evaluate_policy(
            sh_model,
            weights,
            feat_obj_set,
            perf_obj_set,
            bias_demos.X_train,
            bias_demos.X_val,
            bias_demos.X_test,
            bias_demos.y_train,
            bias_demos.y_val,
            bias_demos.y_test,
            bias_demos.subdom_groups_train,
            bias_demos.subdom_groups_val,
            bias_demos.subdom_groups_test,
            bias_demos.expert_train.mu,
            bias_demos.expert_val.mu,
            bias_demos.expert_test.mu,
            exp_info,
            # The baseline always predicts `y` from `X`; it never observes it.
            can_observe_y=False,
        )

        results.record(evaluate_policy_result, policy_start, run)

        run.summary["superhuman_iterations"] = sh_model.history_.n_iterations
        run.summary["superhuman_num_demos"] = len(demo_list)
        for j, name in enumerate(feature_names):
            run.summary[f"superhuman_gamma_{name}"] = (
                sh_model.history_.gamma_superhuman[-1][j]
            )

        trial_runtime = (datetime.datetime.now() - trial_start).total_seconds()

        _finalize_trial(
            run,
            results,
            feat_obj_set,
            perf_obj_set,
            bias_demos,
            unbiased_demos,
            sh_model,
            trial_runtime,
        )


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
    One W&B run per (algorithm, bias type, weight adjustment) combination:
    first the unbiased dataset, then one per entry in
    exp_info["DATASET_BIAS_TYPE_LIST"] (which is not expected to contain "()"
    itself), and within each of those, every algorithm in
    exp_info["ALGORITHMS"].

    The FairIRL Bias Reduction technique reports one run for the unadjusted
    weights, then one per entry in exp_info["WEIGHT_ADJUST_LIST"] (which is not
    expected to contain "()" itself either), each derived from the unadjusted
    weights via `_apply_weight_adjustments()`. The Superhuman Fairness baseline
    has no reward weights to adjust and so reports a single run per bias type.

    Every run holds this trial's config, the metrics of the model it learned,
    and a summary of that model, all produced by the same evaluation code, so
    that the two techniques' numbers are directly comparable.

    Every bias type is applied to the dataset's labels before the data is
    split, and the biased datasets are then all split along the same indices,
    so that the only difference between them is the bias of their labels. Each
    bias type fits its own `y|x` predictor on those labels and learns its
    policies in the MDP built from it, so a policy is computed in, demonstrated
    against, and scored against the same biased labels its expert was.
    """
    trial_start = datetime.datetime.now()

    if session_id is None:
        session_id = new_session_id()

    # The unbiased dataset every bias type is derived from always gets its own
    # set of runs, so `()` is prepended here and DATASET_BIAS_TYPE_LIST is not expected
    # to contain it itself. Prepending it also puts the unbiased runs first,
    # ahead of every listed bias type, and makes the first entry of
    # `demos_by_dataset_bias_type` the unbiased one.
    dataset_bias_type_list = ((),) + tuple(exp_info["DATASET_BIAS_TYPE_LIST"])
    weight_adjust_list = ((),) + tuple(exp_info["WEIGHT_ADJUST_LIST"])

    # Which techniques this trial trains and evaluates. Always in ALGORITHMS
    # order, so the FairIRL technique runs first and its random draws are
    # unaffected by whether the baseline is enabled.
    algorithms = selected_algorithms(exp_info)
    logging.info(f"ALGORITHMS: {algorithms}")

    feat_obj_set, perf_obj_set = _build_objective_sets(exp_info)
    can_observe_y = "FO" in exp_info["IRL_METHOD"]
    X, y, feature_types = _load_or_generate_dataset(exp_info, X, y, feature_types)

    expert_algo_lookup = generate_expert_algo_lookup(feature_types)

    demos_by_dataset_bias_type = _split_dataset_and_generate_expert_demos(
        exp_info,
        expert_algo_lookup,
        feat_obj_set,
        perf_obj_set,
        X,
        y,
        feature_types,
        dataset_bias_type_list,
    )

    # `dataset_bias_type_list` starts with `()`, so the first entry is the unbiased
    # dataset and its demonstrations. It is what everything that needs
    # unbiased data uses -- currently just the `*_unbiased` trial summary
    # metrics, which report every bias type's expert alongside the unbiased
    # one.
    unbiased_demos = demos_by_dataset_bias_type[0]

    x_cols = (
        feature_types["boolean"]
        + feature_types["categoric"]
        + feature_types["continuous"]
    )
    x_cols.remove("z")

    # The expert is an optimal classifier policy whose reward weights are
    # equal and positive across every objective, so those same weights are the
    # unadjusted ones every weight adjustment is derived from.
    n_objs = len(feat_obj_set.objectives)
    unadjusted_weight = np.full(n_objs, 1.0 / n_objs)

    # The unbiased dataset, then every bias type of this trial, each against
    # the same data split, differing only by its labels. Within a bias type,
    # every selected algorithm is trained and evaluated on that same data: the
    # FairIRL technique gets one run for the unadjusted weights and then one
    # per weight adjustment (weight_adjust_list is not expected to contain
    # "()" itself; the unadjusted weights always get their own W&B run
    # regardless of its contents), and the Superhuman Fairness baseline, which
    # has no reward weights to adjust, gets a single run.
    for bias_demos in demos_by_dataset_bias_type:
        expert_train = bias_demos.expert_train
        expert_val = bias_demos.expert_val
        expert_test = bias_demos.expert_test

        logging.info(f"BIAS TYPE: {bias_demos.dataset_bias_type}")

        logging.info(f"muE_train:\n{expert_train.mu}")
        logging.info(f"muE_val:\n{expert_val.mu}")
        logging.info(f"muE_test:\n{expert_test.mu}")
        logging.info(f"muE_perf_train:\n{expert_train.mu_perf}")
        logging.info(f"muE_perf_val:\n{expert_val.mu_perf}")
        logging.info(f"muE_perf_test:\n{expert_test.mu_perf}")

        if ALGORITHM_FAIRIRL in algorithms:
            _run_fairirl_trials(
                exp_info,
                bias_demos,
                unbiased_demos,
                feat_obj_set,
                perf_obj_set,
                feature_types,
                x_cols,
                weight_adjust_list,
                unadjusted_weight,
                can_observe_y,
                trial_i,
                group,
                session_id,
                trial_start,
            )

        if ALGORITHM_SUPERHUMAN in algorithms:
            logging.info(f"ALGORITHM: {ALGORITHM_SUPERHUMAN}")
            _run_superhuman_trial(
                exp_info,
                bias_demos,
                unbiased_demos,
                feat_obj_set,
                perf_obj_set,
                feature_types,
                unadjusted_weight.copy(),
                trial_i,
                group,
                session_id,
                trial_start,
            )


def _run_fairirl_trials(
    exp_info,
    bias_demos,
    unbiased_demos,
    feat_obj_set,
    perf_obj_set,
    feature_types,
    x_cols,
    weight_adjust_list,
    unadjusted_weight,
    can_observe_y,
    trial_i,
    group,
    session_id,
    trial_start,
):
    """
    Train and evaluate the FairIRL Bias Reduction technique for one bias type.

    One W&B run per entry of `weight_adjust_list`. Lifted verbatim out of
    `run_experiment_trial()` when the Superhuman Fairness baseline was added,
    so that each technique's trial reads as one block; what it does is
    unchanged.
    """
    expert_train = bias_demos.expert_train
    expert_val = bias_demos.expert_val
    expert_test = bias_demos.expert_test

    feat_obj_set_cols = [obj.name for obj in feat_obj_set.objectives]
    perf_obj_set_cols = [obj.name for obj in perf_obj_set.objectives]

    # The `y|x` predictor -- and the MDP built from it -- of every policy
    # this bias type learns. It is fit on this bias type's own biased
    # labels, so the MDP its policies are computed in is the world its
    # expert demonstrated in.
    logging.debug("Fitting `y|x` predictor for clf policy...")
    clf, demo_df = _fit_clf_and_demo_df(
        feature_types, bias_demos.X_train, bias_demos.y_train
    )

    for weight_adjust in weight_adjust_list:
        with start_wandb_run(
            exp_info,
            ALGORITHM_FAIRIRL,
            bias_demos.dataset_bias_type,
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
                bias_demos.X_train,  # could switch to bias_demos.X_val
                bias_demos.y_train,  # could switch to bias_demos.y_val
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
                bias_demos.X_train,
                bias_demos.X_val,
                bias_demos.X_test,
                bias_demos.y_train,
                bias_demos.y_val,
                bias_demos.y_test,
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
                bias_demos,
                unbiased_demos,
                clf_pol,
                trial_runtime,
            )


def subdominance_metric_names(exp_info):
    """The objective names the subdominance metric is computed over, in order."""
    return list(exp_info["SUBDOMINANCE_PERF_METRICS_LIST"]) + list(
        exp_info["SUBDOMINANCE_FAIR_METRICS_LIST"]
    )


def compute_relevant_feat_loss(exp_info, demo):
    # Generate feature expectations of the demo, inverted to be loss, where
    # lower is better. `objective_feature_losses()` is the shared definition of
    # that inversion, so that the Superhuman Fairness baseline optimizes and is
    # scored on exactly the quantities used here.
    demo_feat_exp = objective_feature_losses(demo, subdominance_metric_names(exp_info))

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
