import copy
import logging
import numpy as np
import os
import pandas as pd
from research.rl.env.clf_mdp import *
from research.rl.env.clf_mdp_policy import *
from research.rl.env.objectives import *
from research.utils import *
from sklearn.model_selection import train_test_split, KFold


def compute_optimal_policy(
    clf_df,
    clf,
    x_cols,
    obj_set,
    reward_weights,
    skip_error_terms=False,
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
    opt_pol : research.rl.env.clf_mdp.ClassificationMDPPolicy
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
            # case "broken_redlining":
            #     # This code is bugged, don't use it. It's only still here for replicating old results.
            #     minority_z = demo["z"].value_counts(ascending=True).index[0]
            #     minority_yhat = demo["yhat"].value_counts(ascending=True).index[0]
            #     z_mask = demo["z"] == minority_z
            #     random_mask = np.zeros(len(demo), dtype=bool)
            #     random_mask[z_mask] = np.random.random(z_mask.sum()) < percent
            #     demo.loc[random_mask, "yhat"] = minority_yhat
        demo.set_index("index", inplace=True)
        demo.index.name = None
    return demo


def add_classifier_bias(clf, bias_types=[]):
    # Swap thresholds for expert classifiers (invert fairness objective, sort of)
    # Since this doesn't work for non postprocessing classifiers, probably shouldnt use this
    for bias_type in bias_types:
        match bias_type:
            case "threshold_swapping":
                # try: # Commenting out the try except because I want it to fail if it doesn't work, since this is only meant to be used with postprocessing classifiers where it should work
                thresholds = clf.clf.interpolated_thresholder_.interpolation_dict
                thresholds[0], thresholds[1] = thresholds[1], thresholds[0]
                clf.clf.interpolated_thresholder_.interpolation_dict = thresholds
                # except AttributeError:
                #     pass
    # Access clf
    # clf.clf.interpolated_thresholder_.interpolation_dict[0]["operation0"]
    return clf


def generate_demos_k_folds(exp_info, X, y, clf, obj_set, n_demos=3, bias_types=()):
    """
    Generates the expert demonstrations which will be used as the positive
    training samples in the IRL loop, or generates the initial policy
    demonstrations that serve as the initial positive examples in the IRL loop.
    Each demonstration is a vector of feature expectations. Each demonstration
    fits a classifier on a fold of the provided dataset and then predicting on
    the holdout part of the fold.

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
    n_demos : int, default 3
        The number of demonstrations to generate (also the k in k folds).

    Returns
    -------
    mu : numpy.ndarray, shape(m, len(obj_set.objectives))
        The demonstrations feature expectations.
    demos : list<pandas.DataFrame>
        A list of all the demonstration dataframes.
    """
    mu = np.zeros((n_demos, len(obj_set.objectives)))  # demo feat exp
    demos = []

    # Generate demonstrations (populate mu)
    def _generate_demo(clf, mu, demos, k, X_train, X_test, y_train, y_test):
        logging.debug(f"\tStaring iteration {k+1}/{n_demos}")

        # Fit the classifier
        clf.fit(X_train, y_train)
        clf = add_classifier_bias(clf, bias_types)

        logging.debug("\t\tGenerating demo...")
        demo = generate_demo(clf, X_test, y_test)
        demo = add_demo_bias(demo, bias_types=bias_types, dataset=exp_info["DATASET"])
        logging.debug(
            df_to_log(demo.groupby(["z", "y"])[["yhat"]].agg(["count", "mean"]))
        )

        logging.debug("\t\tComputing feature expectations...")
        mu[k] = obj_set.compute_demo_feature_exp(demo)
        demos.append(demo)
        logging.debug(f"\t\tmu[{k}]: {mu[k]}")

        return mu, demos

    logging.debug("")
    logging.debug("Generating expert demonstrations...")

    if n_demos > 1:
        k_fold = KFold(n_demos)
        for k, (train, test) in enumerate(k_fold.split(X, y)):
            _X_train, _y_train = X.iloc[train], y.iloc[train]
            _X_test, _y_test = X.iloc[test], y.iloc[test]
            mu, demos = _generate_demo(
                clf,
                mu,
                demos,
                k,
                _X_train,
                _X_test,
                _y_train,
                _y_test,
            )
    else:
        _X_train, _X_test, _y_train, _y_test = train_test_split(
            X,
            y,
            test_size=0.33,
        )
        mu, demos = _generate_demo(
            clf,
            mu,
            demos,
            0,
            _X_train,
            _X_test,
            _y_train,
            _y_test,
        )

    return mu, demos


def generate_mu_and_demos(
    exp_info,
    X,
    y,
    clf,
    obj_set,
    n_demos=3,
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
    n_demos : int, default 3
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
    mu = np.zeros((1, len(obj_set.objectives)))  # demo feat exp with bias
    demos = []  # list of demo dataframes with bias

    mu_unbiased = np.zeros((1, len(obj_set.objectives)))  # demo feat exp no bias
    demos_unbiased = []  # list of demo dataframes with no bias

    if n_demos < 2:
        n_demos = 2  # KFold requires at least 2 splits

    k_fold = KFold(n_demos)
    for k, (train, test) in enumerate(k_fold.split(X, y)):
        _X_train, _y_train = X.iloc[train], y.iloc[train]
        _X_test, _y_test = X.iloc[test], y.iloc[test]

        # Generate demo with bias
        clf_new = copy.deepcopy(clf)
        clf_new.fit(_X_train, _y_train)
        clf_biased = add_classifier_bias(clf_new, bias_types)
        demo_biased = generate_demo(clf_biased, _X_test, _y_test)
        demo_biased = add_demo_bias(
            demo_biased, bias_types=bias_types, dataset=exp_info["DATASET"]
        )
        # mu += obj_set.compute_demo_feature_exp(demo_biased)
        demos.append(demo_biased)

        # Generate demo without bias
        clf_new = copy.deepcopy(clf)
        clf_new.fit(_X_train, _y_train)
        demo_unbiased = generate_demo(clf_new, _X_test, _y_test)
        # mu_unbiased += obj_set.compute_demo_feature_exp(demo_unbiased)
        demos_unbiased.append(demo_unbiased)

    # Combine all the biased demos into one dataframe and all the unbiased demos into another dataframe
    # Preserves the original data sample order from the dataset.
    demos_combined = pd.concat(demos)
    demos_unbiased_combined = pd.concat(demos_unbiased)

    # Compute mu for the combined demos and combinded unbiased demos
    mu[0] = obj_set.compute_demo_feature_exp(demos_combined)
    mu_unbiased[0] = obj_set.compute_demo_feature_exp(demos_unbiased_combined)

    return mu, demos_combined, mu_unbiased, demos_unbiased_combined


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
