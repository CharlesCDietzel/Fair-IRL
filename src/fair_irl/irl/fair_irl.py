import copy
import json
import logging
import numpy as np
import os
import tempfile
import pandas as pd
from fair_irl.rl.clf_mdp import *
from fair_irl.rl.clf_mdp_policy import *
from fair_irl.rl.objectives import *
from fair_irl.utils import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


def compute_optimal_policy(
    clf_df,
    clf,
    x_cols,
    obj_set,
    reward_weights,
    skip_error_terms=True,
    method="highs",
    gamma=1e-9,
    min_freq_fill_pct=0,
    restrict_y=True,
):
    """
    Learns the optimal policies from the provided reward weights.

    The `clf_df` is used to "fit" the ClfMDP parameters (e.g. b_eq, A_eq, etc)
    as well as the classifier that predicts `y` from `X`.

    The other classification parameters (`feature_types`, and `clf` are
    used only on the classifier that predicts the `y` value from the `X`
    values.

    Parameters
    ----------
    clf_df : pandas.DataFrame
        Classification dataset. Required columns:
            'z' : int. Binary protected attribute.
            'y' : int. Binary target variable.
    clf : sklearn.pipeline.Pipeline
        Sklearn classifier instance.
    x_cols : list<str>
        The columns that are used in the state (along with `z` and `y`).
    obj_set : ObjectiveSet
        The objective set.
    reward_weights : dict<str, float>
        Keys are objective identifiers. Values are their respective reward
        weights.
    skip_error_terms : bool, default False
        If true, doesn't try and find all solutions and instead just invokes
        the scipy solver on the input terms.
    method : str, default 'highs'
        The scipy solver method to use. Options are 'highs' (default),
        'highs-ds', 'highs-ipm'.
    gamma : float, range, [0, 1)
        The MDP discount factor. Should be close to zero since this is a
        classification mdp, but cannot be zero exactly otherwise the linear
        program won't converge.
    min_freq_fill_pct, float, range[0, 1), default 0
        Minimum frequency for each input variable to not get replaced by a
        default value.
    restrict_y : bool, deafult True
        If True, policy must have same action for any x,z combo, regardless
        of y.

    Returns
    -------
    opt_pol : fair_irl.rl.clf_mdp.ClassificationMDPPolicy
        The optimal policy. If there are multiple, it randomly selects one.
    """
    clf_mdp = ClassificationMDP(
        gamma=gamma,
        obj_set=obj_set,
        x_cols=x_cols,
    )

    # Construct the mdp, including optimization problems
    clf_mdp.fit(
        reward_weights=reward_weights,
        clf_df=clf_df,
        min_freq_fill_pct=min_freq_fill_pct,
        restrict_y=restrict_y,
    )

    # Compute the optimal policy(s). This does NOT fit the classifier that
    # predicts Y from Z, X. That occurs on the generate_demo() call. This
    # `compute_optimal_policies` computes the optimal policy, assuming the
    # state is known.
    optimal_policies = clf_mdp.compute_optimal_policies(
        skip_error_terms=skip_error_terms,
        method=method,
    )

    # Pick from one of the policies (if there are multiple).
    logging.info(f"\n\t\tFound {len(optimal_policies)} optimal policies.")
    sampled_policy = optimal_policies[np.random.choice(len(optimal_policies))]

    clf_pol = ClassificationMDPPolicy(
        mdp=clf_mdp,
        pi=sampled_policy,
        clf=clf,
    )

    return clf_pol


class OptClfMDPPolicyExpert:
    """
    Configuration holder for the "OptClfMDPPol" expert algo (see
    `experiment_utils.generate_expert_algo_lookup()`).

    Unlike the other entries in `expert_algo_lookup`, a `ClassificationMDPPolicy`
    can't be built until the feature-expectation objective set (`obj_set`) and
    a concrete training fold (`X_train`, `y_train`) are known, neither of which
    is available yet when `expert_algo_lookup` is constructed. This class just
    holds `feature_types` and defers the actual `compute_optimal_policy()` call
    to `build_policy()`, which `generate_non_overfit_demos()` invokes once per
    fold instead of the usual `clf.fit()`/`clf.predict()`.

    Parameters
    ----------
    feature_types : dict<str, list>
        Mapping of column names to their type of feature.
    """

    def __init__(self, feature_types):
        self.feature_types = feature_types

    def build_policy(self, X_train, y_train, obj_set, exp_info):
        """
        Fits the `y|x` predictor on `(X_train, y_train)` and computes the
        optimal classifier policy for that data, using equal, positive reward
        weights (summing to 1) across every objective in `obj_set`.

        Returns
        -------
        clf_pol : ClassificationMDPPolicy
        """
        x_cols = (
            self.feature_types["boolean"]
            + self.feature_types["categoric"]
            + self.feature_types["continuous"]
        )
        x_cols.remove("z")

        inner_clf = sklearn_clf_pipeline(
            feature_types=self.feature_types,
            clf_inst=RandomForestClassifier(),
        )
        inner_clf.fit(X_train, y_train)

        clf_df = pd.DataFrame(X_train).copy()
        clf_df["y"] = y_train

        n_objs = len(obj_set.objectives)
        reward_weights = {obj.name: 1.0 / n_objs for obj in obj_set.objectives}

        return compute_optimal_policy(
            clf_df=clf_df,
            clf=inner_clf,
            x_cols=x_cols,
            obj_set=obj_set,
            reward_weights=reward_weights,
            skip_error_terms=True,
            method=exp_info["METHOD"],
            min_freq_fill_pct=exp_info["MIN_FREQ_FILL_PCT"],
            restrict_y=exp_info["RESTRICT_Y_ACTION"],
        )


def generate_demo(clf, X_test, y_test, can_observe_y=False):
    """
    Create demonstration dataframe (columns are '**X', 'yhat', 'y') from a
    fitted classifer `clf`.

    Parameters
    ----------
    clf : fitted sklearn classifier
    X_test : pandas.DataFrame
    y_test : pandas.Series
    can_observe_y : bool, default False
        Whether the policy can "see" y or if it needs to predict it from X.

    Returns
    -------
    demo : pandas.DataFrame
        Demonstrations. Each demonstration represents an iteration of a
        trained classifier and its predictions on a hold-out set. Columns are
            **`X` columns : all input columns (i.e. `X`)
            yhat : predictions
            y : ground truth targets
    """
    demo = pd.DataFrame(X_test).copy()

    if can_observe_y:
        # yhat = clf.predict(X_test, y_test)
        if "y" in X_test.columns:
            demo["yhat"] = demo["y"].copy()
        else:
            demo["yhat"] = y_test
    else:
        demo["yhat"] = clf.predict(X_test).astype(np.int64)

    demo["y"] = y_test.copy()

    return demo


def add_demo_bias(demo, bias_types=(), dataset=None):
    # Z = 0 is discriminated against, Y = 0 is "bad" outcome (except for COMPAS where Y=1 is "bad" outcome)
    # rz - redlined Z value
    # ry - redline Y outcome
    # nrz - non-redlined Z value
    # nry - non-redline Y outcome
    if dataset == "Adult":
        rz = 0
        ry = 0
        nrz = 1
        nry = 1
    elif dataset == "Boston":
        rz = 0
        ry = 0
        nrz = 1
        nry = 1
    elif "ACSIncome__" in dataset:  # All ACSIncome datasets have same strtucture
        rz = 0
        ry = 0
        nrz = 1
        nry = 1
    elif dataset == "COMPAS":  # COMPAS Y redlining is reversed
        rz = 0
        ry = 1
        nrz = 1
        nry = 0

    for bias_type in bias_types:
        percent = 0.2
        bias_type_name = bias_type[0]
        if len(bias_type) > 1:
            percent = bias_type[1]
        demo.reset_index(inplace=True)
        match bias_type_name:
            case "unbalanced_redlining":
                # wherever Z==rz, set yhat = ry with 20% probability. Otherwise, keep yhat the same
                # Count how many rows have z == rz
                rz_count = (demo["z"] == rz).sum()
                # Multiply that by 20% to get the number of rows to redline
                n = int(rz_count * percent)
                # Step 3: Get n indices where z == rz
                rz_indices = demo[demo["z"] == rz].sample(n=n).index
                # Step 4: Set yhat to ry for sampled rows
                demo.loc[rz_indices, "yhat"] = ry
            case "balanced_redlining":
                # wherever Z==rz, set yhat = ry with 20% probability. Otherwise, keep yhat the same. Also, whereever Z==nrz, randomly set an equal number of yhat = nry
                # Count how many rows have z == rz
                rz_count = (demo["z"] == rz).sum()
                # Multiply that by 20% to get the number of rows to redline
                n = int(rz_count * percent)
                # Edge case check: If there are fewer z == nrz than n, set n = nrz_count instead
                nrz_count = (demo["z"] == nrz).sum()
                n = min(nrz_count, n)
                # Get the n indices where z == rz
                rz_indices = demo[demo["z"] == rz].sample(n=n).index
                # Set yhat to ry for sampled rows
                demo.loc[rz_indices, "yhat"] = ry
                # Get the n indices where z == nrz
                nrz_indices = demo[demo["z"] == nrz].sample(n=n).index
                # Set yhat to nry for sampled rows
                demo.loc[nrz_indices, "yhat"] = nry
            case "perfectly_balanced_redlining":
                # wherever Z==rz and yhat==nry, set yhat = ry with 20% probability. Also, whereever Z==nrz and yhat==ry, randomly set an equal number of yhat = nry
                # Count how many rows have z == rz AND yhat == nry
                rz_count = ((demo["z"] == rz) & (demo["yhat"] == nry)).sum()
                # Multiply that by 20% to get the number of rows to flip
                n = int(rz_count * percent)
                # Edge case check: If there are fewer z == nrz and yhat == ry than n, set n = nrz_count instead
                nrz_count = ((demo["z"] == nrz) & (demo["yhat"] == ry)).sum()
                n = min(nrz_count, n)
                # Get the n indices where z == rz AND yhat == nry
                rz_indices = (
                    demo[(demo["z"] == rz) & (demo["yhat"] == nry)].sample(n=n).index
                )
                # Get the n indices where z == nrz AND yhat == ry
                nrz_indices = (
                    demo[(demo["z"] == nrz) & (demo["yhat"] == ry)].sample(n=n).index
                )
                # Set yhat to ry for sampled rows
                demo.loc[rz_indices, "yhat"] = ry
                # Set yhat to nry for sampled rows
                demo.loc[nrz_indices, "yhat"] = nry
        demo.set_index("index", inplace=True)
        demo.index.name = None
    return demo


def _make_corruption_clf(clf_type, feature_types):
    """Return a fresh, unfitted sklearn pipeline for the configured classifier type."""
    if clf_type == "MLP":
        clf_inst = MLPClassifier(
            hidden_layer_sizes=(100, 50), max_iter=500, random_state=42
        )
    elif clf_type == "CatBoost":
        clf_inst = CatBoostClassifier(allow_writing_files=False, logging_level="Silent")
    elif clf_type == "XGBoost":
        clf_inst = XGBClassifier(eval_metric="logloss", verbosity=0)
    elif clf_type == "SVM":
        clf_inst = SVC(kernel="rbf")
    else:
        raise ValueError(f"Unknown ML_WEIGHT_ADJUST_CLF_TYPE: {clf_type}")
    return sklearn_clf_pipeline(feature_types, clf_inst)


def _ml_xgb_get_leaf_values(clf_inner):
    """Extract XGBoost leaf node values as a flat numpy array via JSON serialisation."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        temp_path = f.name
    try:
        clf_inner.get_booster().save_model(temp_path)
        with open(temp_path) as f:
            model_dict = json.load(f)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    leaves = []
    for tree in (
        model_dict.get("learner", {})
        .get("gradient_booster", {})
        .get("model", {})
        .get("trees", [])
    ):
        left_children = tree.get("left_children", [])
        base_weights = tree.get("base_weights", [])
        for j, lc in enumerate(left_children):
            # -1 or very large positive value signals a leaf (no left child)
            if (lc == -1 or lc > 1_000_000_000) and j < len(base_weights):
                leaves.append(float(base_weights[j]))
    return np.array(leaves)


def _ml_xgb_set_leaf_values(clf_inner, new_values):
    """Overwrite XGBoost leaf values in-place via JSON save/load."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        temp_path = f.name
    try:
        clf_inner.get_booster().save_model(temp_path)
        with open(temp_path) as f:
            model_dict = json.load(f)

        leaf_idx = 0
        for tree in (
            model_dict.get("learner", {})
            .get("gradient_booster", {})
            .get("model", {})
            .get("trees", [])
        ):
            left_children = tree.get("left_children", [])
            for j, lc in enumerate(left_children):
                if (lc == -1 or lc > 1_000_000_000) and leaf_idx < len(new_values):
                    lv = float(new_values[leaf_idx])
                    if j < len(tree.get("base_weights", [])):
                        tree["base_weights"][j] = lv
                    if j < len(tree.get("split_conditions", [])):
                        tree["split_conditions"][j] = lv
                    leaf_idx += 1

        with open(temp_path, "w") as f:
            json.dump(model_dict, f)
        clf_inner.get_booster().load_model(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def _get_flat_clf_params(clf_pipeline, clf_type):
    """Return the learnable parameters of the inner classifier as a flat numpy array."""
    inner = clf_pipeline.named_steps["classifier"]
    if clf_type == "MLP":
        return np.concatenate(
            [c.flatten() for c in inner.coefs_]
            + [b.flatten() for b in inner.intercepts_]
        )
    elif clf_type == "CatBoost":
        return inner.get_leaf_values().copy()
    elif clf_type == "XGBoost":
        return _ml_xgb_get_leaf_values(inner)
    elif clf_type == "SVM":
        return np.concatenate([inner.dual_coef_.flatten(), inner.intercept_.flatten()])
    raise ValueError(f"Unknown clf_type: {clf_type}")


def _set_flat_clf_params(clf_pipeline, clf_type, biased_params):
    """Return a deep copy of clf_pipeline with biased_params installed."""
    new_clf = copy.deepcopy(clf_pipeline)
    orig_inner = clf_pipeline.named_steps["classifier"]
    new_inner = new_clf.named_steps["classifier"]

    if clf_type == "MLP":
        idx = 0
        for coef in new_inner.coefs_:
            n = coef.size
            coef[:] = biased_params[idx : idx + n].reshape(coef.shape)
            idx += n
        for intercept in new_inner.intercepts_:
            n = intercept.size
            intercept[:] = biased_params[idx : idx + n].reshape(intercept.shape)
            idx += n
    elif clf_type == "CatBoost":
        # copy.deepcopy() leaves the model in a non-"solid" internal state that
        # CatBoost refuses to modify in-place, so round-trip through disk first.
        with tempfile.NamedTemporaryFile(suffix=".cbm", delete=False) as f:
            temp_path = f.name
        try:
            new_inner.save_model(temp_path)
            new_inner.load_model(temp_path)
            new_inner.set_leaf_values(biased_params)
        finally:
            os.unlink(temp_path)
    elif clf_type == "XGBoost":
        _ml_xgb_set_leaf_values(new_inner, biased_params)
    elif clf_type == "SVM":
        n_dual = orig_inner.dual_coef_.size
        new_inner.dual_coef_[:] = biased_params[:n_dual].reshape(
            orig_inner.dual_coef_.shape
        )
        new_inner.intercept_[:] = biased_params[n_dual:].reshape(
            orig_inner.intercept_.shape
        )
    return new_clf


def _apply_corruption(params, percent, noise_type, magnitude):
    """Return a biased copy of the flat parameter array according to the selected process."""
    params = params.copy()
    if len(params) == 0:
        return params
    if noise_type == "uniform":
        n_select = max(1, int(len(params) * percent))
        idx = np.random.choice(len(params), size=n_select, replace=False)
        params[idx] += np.random.uniform(-magnitude, magnitude, size=n_select)
        return params
    elif noise_type == "gaussian":
        n_select = max(1, int(len(params) * percent))
        idx = np.random.choice(len(params), size=n_select, replace=False)
        params[idx] += np.random.normal(0, magnitude, size=n_select)
        return params
    raise ValueError(f"Unknown noise_type: {noise_type}")


def add_corruption_bias(X, y, feature_types, bias_types=()):
    yhat = None
    for bias_type in bias_types:
        bias_type_name = bias_type[0]
        if len(bias_type) > 1:
            parameters = bias_type[1:]
        match bias_type_name:
            case "corruption_bias":
                clf_type, percent, noise_type, magnitude = parameters
                clf = _make_corruption_clf(clf_type, feature_types)
                clf.fit(X, y)
                flat_params = _get_flat_clf_params(clf, clf_type)
                biased_params = _apply_corruption(
                    flat_params, percent, noise_type, magnitude
                )
                biased_clf = _set_flat_clf_params(clf, clf_type, biased_params)
                demo = generate_demo(
                    biased_clf,
                    X,
                    y,
                    can_observe_y=False,
                )
                yhat = demo["yhat"]
    return yhat


def generate_non_overfit_demos(
    exp_info,
    X,
    y,
    feature_types,
    clf,
    n_demos=2,
    yhat=None,
    obj_set=None,
):
    demos_folds = []  # list of demo dataframes with no bias
    demos_folds_biased = []  # list of demo dataframes with bias

    k_fold = KFold(n_demos)
    for train, test in k_fold.split(X, y):
        X_train, y_train = (
            X.iloc[train],
            y.iloc[train],
        )
        X_test, y_test = (
            X.iloc[test],
            y.iloc[test],
        )
        if yhat is not None:
            yhat_train = yhat.iloc[train]

        # Generate demo with bias
        if isinstance(clf, OptClfMDPPolicyExpert):
            # A ClassificationMDPPolicy can't be fit like a regular
            # classifier: it must be rebuilt (via compute_optimal_policy())
            # from this fold's own training data instead.
            clf_fold = clf.build_policy(X_train, y_train, obj_set, exp_info)
        else:
            clf_fold = copy.deepcopy(clf)
            clf_fold.fit(X_train, y_train)
        demos_fold = generate_demo(clf_fold, X_test, y_test)
        if yhat is not None:
            if isinstance(clf, OptClfMDPPolicyExpert):
                clf_fold_biased = clf.build_policy(
                    X_train, yhat_train, obj_set, exp_info
                )
            else:
                clf_fold_biased = copy.deepcopy(clf)
                clf_fold_biased.fit(X_train, yhat_train)
            demos_fold_biased = generate_demo(clf_fold_biased, X_test, y_test)
            demos_folds_biased.append(demos_fold_biased)

        demos_folds.append(demos_fold)

    # Combine all the biased demos into one dataframe and all the unbiased demos into another dataframe
    # Preserves the original data sample order from the dataset.
    if yhat is None:
        return pd.concat(demos_folds), None
    else:
        return pd.concat(demos_folds), pd.concat(demos_folds_biased)


def generate_mu_and_demos(
    exp_info,
    X,
    y,
    feature_types,
    clf,
    obj_set,
    n_demos=2,
    bias_types=(),
):
    """
    Improved implementation of `generate_demos_k_folds` that uses k-folds to generate
    demonstrations for all data samples without overlap and also without overfitting.
    Returns demos as the combined demonstrations from all folds, and mu as the feature
    expectations of all demos. Additionally, this implementation returns versions of mu and demos
    with no bias added, which can be used to sanity-check that the bias is behaving as expected.

    Parameters
    ----------
    X : numpy.ndarray
        The X training data reserved for generating the demonstrations.
    y : array-like
        The y training data reserved for generating the demonstrations.
    clf : sklearn.pipeline.Pipeline, ClassifierMixin
        Sklearn classifier pipeline.
    obj_set : ObjectiveSet
        The objective set.
    n_demos : int, default 2
        The number of demonstrations to generate (also the k in k folds).

    Returns
    -------
    mu : numpy.ndarray, shape(1, len(obj_set.objectives))
        The feature expectations across all demonstrations with bias added.
    demos : pandas.DataFrame
        A dataframe of all the demonstrations across the K-folds with bias added.
    mu_unbiased : numpy.ndarray, shape(1, len(obj_set.objectives))
        The feature expectations across all demonstrations with no bias added.
    demos_unbiased : pandas.DataFrame
        A dataframe of all the demonstrations across the K-folds with no bias added.
    """
    mu_biased = np.zeros((1, len(obj_set.objectives)))  # demo feat exp with bias

    mu_unbiased = np.zeros((1, len(obj_set.objectives)))  # demo feat exp no bias

    if n_demos < 2:
        n_demos = 2  # KFold requires at least 2 splits

    yhat_biased = add_corruption_bias(X, y, feature_types, bias_types=bias_types)
    unbiased_demos, biased_demos = generate_non_overfit_demos(
        exp_info,
        X,
        y,
        feature_types,
        clf,
        n_demos=n_demos,
        yhat=yhat_biased,
        obj_set=obj_set,
    )
    if biased_demos is None:
        biased_demos = unbiased_demos.copy()
    biased_demos = add_demo_bias(
        biased_demos, bias_types=bias_types, dataset=exp_info["DATASET"]
    )

    logging.info(f"Bias types added: {bias_types}")
    logging.info(
        f"Percent of yhat unchanged with added bias: {((unbiased_demos["yhat"] == biased_demos["yhat"]).mean() * 100.0).item()}%"
    )

    # Compute mu for the combined demos and combinded unbiased demos
    mu_unbiased[0] = obj_set.compute_demo_feature_exp(unbiased_demos)
    mu_biased[0] = obj_set.compute_demo_feature_exp(biased_demos)

    logging.info(f"mu_unbiased[0] = {mu_unbiased[0]}")
    logging.info(f"mu_biased[0] = {mu_biased[0]}")
            
    return mu_biased, biased_demos, mu_unbiased, unbiased_demos


def init_muL_from_muE(non_expert_algo, muE):
    """
    Initializes the learned feature expectations list (muL) with a single element
    that is equal to the learned expert feature expectations (muE), with some
    "degrading factor" applied to represent a worse initial policy than the expert.
    The main advantage of initializing muL this way as opposed to training a
    separate non-expert classifier is that this ensures that the initial muL value
    contains the same pattern of error as muE, which should reduce the noise in
    the learned feature weights and improve fairness preference matching between
    muE and the final muL.

    Parameters    ----------
    non_expert_algo : str
        The name of the non-expert algorithm. This is used to determine the degrading
        factor applied to muE to get the initial muL.
    muE : numpy.ndarray, shape (n_expert_demos, len(len(objective_set.objectives)))
        Array of all expert feature expectations.
    Returns
    -------
    muL : array-like<array-like<float>>
        A numpy array containing a single element which is the initialized learned feature expectations.
    """

    absolute_degrading_factor = (
        0.05  # reduce the magnitude of each feature expectation by 0.05.
    )
    relative_degrading_factor = 0.9333333333333

    muE_avg = muE.mean(axis=0)
    # # Calculate relative_degrading_factor such that the final magnitude of degredation is the same as
    # # the absolute_degrading_factor.
    # pre_muL = np.array([muE_avg - absolute_degrading_factor])
    # muE_sum = np.sum(muE_avg)
    # pre_muL_sum = np.sum(pre_muL)
    # relative_degrading_factor = pre_muL_sum / muE_sum

    if non_expert_algo == "DegradeAbsolute":
        muL = np.array([muE_avg - absolute_degrading_factor])
    elif non_expert_algo == "DegradeRelative":
        muL = np.array([muE_avg * relative_degrading_factor])
    elif non_expert_algo == "DegradeNoisy":
        noise = np.random.uniform(
            -absolute_degrading_factor, absolute_degrading_factor, size=muE_avg.shape
        )
        muL = np.array([muE_avg + noise])
    elif non_expert_algo == "DegradeNoisyAbsolute":
        noise = np.random.uniform(-absolute_degrading_factor * 2, 0, size=muE_avg.shape)
        muL = np.array([muE_avg + noise])
    elif non_expert_algo == "DegradeNoisyRelative":
        noise = np.random.uniform(
            2 * relative_degrading_factor - 1, 1, size=muE_avg.shape
        )
        muL = np.array([muE_avg * noise])
    return muL


def irl_error(
    w,
    muE,
    muL,
    dot_weights_feat_exp=True,
    svm_margin=None,
):
    """
    Computes t[i] = argmax_{mu[j] for j in muL} wT(muE-mu[j])

    Parameters
    ----------
    w : np.array<float>
        Weights.
    muE : array-like<np.array<float>>, shape (n_expert_demos, len(len(objective_set.objectives)))
        Array of all expert feature expectations.
    muL : array-like<array-like<float>>
        List of all learned feature expectations.
    dot_weights_feat_exp : bool, default True
        If True, error = l2_norm( (muE - muL).dot(w) )
        If False, error = l2_norm(muE - muL) * l2_norm(w)
        The default value is the typical IRL one. Use the other if all feat
        exp should be non-zero weights. Helps avoid issue of IRL nulling out
        feat exp componets that it has trouble matching.
    allow_neg_weights : bool, default False
        If True, allows positive feature expectation errors to be nulled out
        even if weights are negative.
    svm_margin : float, optional
        If provided, use this for error.

    Returns
    -------
    mu_deltas : array<float>
        The array of differences between muE and muL.
    l2_mu_delta : float
        The l2 norm of the muE and muL deltas.
    abs_l2_mu_delta : float
        The l2 norm of the muE and muL deltas, without normalizing by muE.
    err : float
        The IRL error of the current policy.
    svm_margin : float
        The margin of the SVM separating the expert and learned feature expectations.
    """
    # Trying this out. Make mu delta errors RELATIVE to their magnitude. So
    # adding the muE.mean(axis=0) as a denominator
    mu_deltas = (muE.mean(axis=0) - muL[-1]) / muE.mean(axis=0)

    abs_mu_deltas = muE.mean(axis=0) - muL[-1]

    if dot_weights_feat_exp:
        err = np.linalg.norm(w * mu_deltas, ord=2)
    else:
        err = np.linalg.norm(mu_deltas) * np.linalg.norm(w, ord=2)

    l2_mu_delta = np.linalg.norm(mu_deltas, ord=2)
    abs_l2_mu_delta = np.linalg.norm(abs_mu_deltas, ord=2)

    return mu_deltas, l2_mu_delta, abs_l2_mu_delta, err, svm_margin
