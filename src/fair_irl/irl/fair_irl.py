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


def add_redlining_bias(X, y, bias_type=(), dataset=None):
    """
    Apply a redlining bias to a dataset's labels.

    The bias is applied to the dataset itself, before it is split and before
    any expert is fit on it, so that every downstream use of the labels -- the
    expert's training targets, the `y` column of its demonstrations, and every
    metric computed against them -- sees the same biased labels.

    Parameters
    ----------
    X : pandas.DataFrame
        The dataset's input columns. Only the protected attribute `z` is used.
    y : pandas.Series
        The dataset's labels.
    bias_type : tuple, default ()
        The bias to apply, `()` (or any non-redlining bias type) leaving the
        labels untouched. The first element names the bias; the second, if
        given, is the percent of the redlined group whose label is flipped.
    dataset : str
        Name of the dataset, which decides which `z` group is redlined and
        which `y` outcome counts as the "bad" one.

    Returns
    -------
    y_biased : pandas.Series
        A copy of `y` with the bias applied, indexed and named like `y`.
    """
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

    percent = 0.2
    if len(bias_type) > 0:
        bias_type_name = bias_type[0]
    else:
        bias_type_name = None
    if len(bias_type) > 1:
        percent = bias_type[1]

    # Work on a positionally indexed frame of just the columns the bias needs,
    # so that the dataset's own index (which needn't be unique or ordered)
    # can't interfere with the sampling below.
    df = pd.DataFrame({"z": np.asarray(X["z"]), "y": np.asarray(y)})

    match bias_type_name:
        case "unbalanced_redlining":
            # wherever Z==rz, set y = ry with 20% probability. Otherwise, keep y the same
            # Count how many rows have z == rz
            rz_count = (df["z"] == rz).sum()
            # Multiply that by 20% to get the number of rows to redline
            n = int(rz_count * percent)
            # Step 3: Get n indices where z == rz
            rz_indices = df[df["z"] == rz].sample(n=n).index
            # Step 4: Set y to ry for sampled rows
            df.loc[rz_indices, "y"] = ry
        case "balanced_redlining":
            # wherever Z==rz, set y = ry with 20% probability. Otherwise, keep y the same. Also, whereever Z==nrz, randomly set an equal number of y = nry
            # Count how many rows have z == rz
            rz_count = (df["z"] == rz).sum()
            # Multiply that by 20% to get the number of rows to redline
            n = int(rz_count * percent)
            # Edge case check: If there are fewer z == nrz than n, set n = nrz_count instead
            nrz_count = (df["z"] == nrz).sum()
            n = min(nrz_count, n)
            # Get the n indices where z == rz
            rz_indices = df[df["z"] == rz].sample(n=n).index
            # Set y to ry for sampled rows
            df.loc[rz_indices, "y"] = ry
            # Get the n indices where z == nrz
            nrz_indices = df[df["z"] == nrz].sample(n=n).index
            # Set y to nry for sampled rows
            df.loc[nrz_indices, "y"] = nry
        case "perfectly_balanced_redlining":
            # wherever Z==rz and y==nry, set y = ry with 20% probability. Also, whereever Z==nrz and y==ry, randomly set an equal number of y = nry
            # Count how many rows have z == rz AND y == nry
            rz_count = ((df["z"] == rz) & (df["y"] == nry)).sum()
            # Multiply that by 20% to get the number of rows to flip
            n = int(rz_count * percent)
            # Edge case check: If there are fewer z == nrz and y == ry than n, set n = nrz_count instead
            nrz_count = ((df["z"] == nrz) & (df["y"] == ry)).sum()
            n = min(nrz_count, n)
            # Get the n indices where z == rz AND y == nry
            rz_indices = df[(df["z"] == rz) & (df["y"] == nry)].sample(n=n).index
            # Get the n indices where z == nrz AND y == ry
            nrz_indices = df[(df["z"] == nrz) & (df["y"] == ry)].sample(n=n).index
            # Set y to ry for sampled rows
            df.loc[rz_indices, "y"] = ry
            # Set y to nry for sampled rows
            df.loc[nrz_indices, "y"] = nry

    return pd.Series(df["y"].to_numpy(), index=y.index, name=y.name)


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


def add_corruption_bias(X, y, feature_types, bias_type=()):
    """
    Apply a corruption bias to a dataset's labels.

    A classifier is fit on the dataset, its learned parameters are corrupted,
    and its predictions on the dataset become the biased labels. As with
    `add_redlining_bias()`, the bias lands on the dataset itself rather than
    on an expert's predictions, so everything downstream is fit on and scored
    against the biased labels.

    Returns
    -------
    y_biased : pandas.Series or None
        A copy of `y` with the bias applied, indexed and named like `y`, or
        `None` if `bias_type` is not a corruption bias.
    """
    y_biased = None
    if len(bias_type) > 0:
        bias_type_name = bias_type[0]
    else:
        bias_type_name = None
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
            y_biased = pd.Series(demo["yhat"].to_numpy(), index=y.index, name=y.name)
    return y_biased


def generate_non_overfit_demos(
    exp_info,
    X,
    y,
    clf,
    n_demos=2,
    obj_set=None,
):
    """
    Generate the demonstrations of every fold.

    Returns
    -------
    demos : pandas.DataFrame
        The demonstrations of all the folds combined.
    """
    demos_folds = []

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

        if isinstance(clf, OptClfMDPPolicyExpert):
            # A ClassificationMDPPolicy can't be fit like a regular
            # classifier: it must be rebuilt (via compute_optimal_policy())
            # from this fold's own training data instead.
            clf_fold = clf.build_policy(X_train, y_train, obj_set, exp_info)
        else:
            clf_fold = copy.deepcopy(clf)
            clf_fold.fit(X_train, y_train)
        demos_folds.append(generate_demo(clf_fold, X_test, y_test))

    # Combine all the demos into one dataframe.
    # Preserves the original data sample order from the dataset.
    return pd.concat(demos_folds)


def generate_mu_and_demos(
    exp_info,
    X,
    y,
    clf,
    obj_set,
    n_demos=2,
):
    """
    Improved implementation of `generate_demos_k_folds` that uses k-folds to generate
    demonstrations for all data samples without overlap and also without overfitting.
    Returns demos as the combined demonstrations from all folds, and mu as the feature
    expectations of all demos.

    Any bias has already been applied to `y` by the caller (see
    `add_corruption_bias()` and `add_redlining_bias()`), so the demonstrations
    generated here carry the bias of the dataset they were generated from and
    no bias is added to them afterwards.

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
        The feature expectations across all demonstrations.
    demos : pandas.DataFrame
        A dataframe of all the demonstrations across the K-folds.
    """
    mu = np.zeros((1, len(obj_set.objectives)))  # demo feat exp

    if n_demos < 2:
        n_demos = 2  # KFold requires at least 2 splits

    demos = generate_non_overfit_demos(
        exp_info,
        X,
        y,
        clf,
        n_demos=n_demos,
        obj_set=obj_set,
    )

    # Compute mu for the combined demos
    mu[0] = obj_set.compute_demo_feature_exp(demos)
    logging.info(f"mu[0] = {mu[0]}")

    return mu, demos


def policy_error(
    w,
    muE,
    muL,
    dot_weights_feat_exp=True,
):
    """
    Computes how far a learned policy's feature expectations are from the
    expert's, i.e. wT(muE-muL).

    Parameters
    ----------
    w : np.array<float>
        Weights.
    muE : array-like<np.array<float>>, shape (n_expert_demos, len(len(objective_set.objectives)))
        Array of all expert feature expectations.
    muL : array-like<float>
        The learned policy's feature expectations.
    dot_weights_feat_exp : bool, default True
        If True, error = l2_norm( (muE - muL).dot(w) )
        If False, error = l2_norm(muE - muL) * l2_norm(w)
        Use the latter if all feat exp should be non-zero weights. Helps avoid
        the issue of nulling out feat exp componets that are hard to match.

    Returns
    -------
    mu_deltas : array<float>
        The array of differences between muE and muL.
    l2_mu_delta : float
        The l2 norm of the muE and muL deltas.
    abs_l2_mu_delta : float
        The l2 norm of the muE and muL deltas, without normalizing by muE.
    err : float
        The error of the policy.
    """
    # Trying this out. Make mu delta errors RELATIVE to their magnitude. So
    # adding the muE.mean(axis=0) as a denominator
    mu_deltas = (muE.mean(axis=0) - muL) / muE.mean(axis=0)

    abs_mu_deltas = muE.mean(axis=0) - muL

    if dot_weights_feat_exp:
        err = np.linalg.norm(w * mu_deltas, ord=2)
    else:
        err = np.linalg.norm(mu_deltas) * np.linalg.norm(w, ord=2)

    l2_mu_delta = np.linalg.norm(mu_deltas, ord=2)
    abs_l2_mu_delta = np.linalg.norm(abs_mu_deltas, ord=2)

    return mu_deltas, l2_mu_delta, abs_l2_mu_delta, err
