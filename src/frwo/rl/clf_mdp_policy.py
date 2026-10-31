import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


class ClassificationMDPPolicy(BaseEstimator, ClassifierMixin):
    """
    A Scikit-Learn compatible wrapper for a Classification MDP policy.

    Parameters
    ----------
    mdp : ClassificationMDP
        The classification mdp instance.
    pi : numpy.array<int>
        The optimal policy (optimal action once `y` is known).
    clf : sklearn.BaseEstimator
        Binary classifier used for predicting `y` from `X`. It needs to be
        fitted already, and it needs to contain the full preprocessing of
        inputs.
    default_action : int
        The default action to use if the state lookup fails.

    Attributes
    ----------
    reward_weights : list<float>
        Weights for each objective component.
    pi_df_ : pandas.DataFrame
        Policy represented as a dataframe.
    """

    def __init__(self, mdp, pi, clf, default_action=0):
        self.mdp = mdp
        self.pi = pi
        self.clf = clf
        self.default_action = default_action
        self._pi_df = None

    def fit(self, X, y):
        """
        Pass through. Doesn't do anything.

        Parameters
        ----------
        X : pandas.DataFrame
            Classification input.
        y : pandas.Series<int>
            Binary target variable.

        Returns
        -------
        None
        """
        pass

    def predict(self, X, y=None):
        """
        If `y` is None, then predicts `y` from `X`, then returns the optimal
        action for that value of `(X, y)`. Sort of like a two-step POMDP.

        If `y` is provided, then does not predict `y`.

        I.e. here is some crude pseudocode reprsenting what's actually
        happening:
            ```
            if y is None:
                y = predict(X)

            a = pi(X, y)
            return a
            ```
        Parameters
        ---------
        X : pandas.DataFrame
            Input data.
        y : pandas.Series
            label data.

        Returns
        -------
        actions : numpy.array<int>, len(len(X))
            The "predictions", actually the actions from the Clf MDP.
        """
        return self.actions_for_states(self.lookup_states(X, y=y))

    def lookup_states(self, X, y=None):
        """
        Looks up the MDP state of each row of `X`, predicting `y` from `X` if
        it isn't provided (see `predict()`).

        The states only depend on the MDP and `clf`, not on the policy `pi`, so
        they can be computed once and reused with `actions_for_states()` for
        any number of policies of the same MDP.

        Parameters
        ---------
        X : pandas.DataFrame
            Input data.
        y : pandas.Series
            label data.

        Returns
        -------
        states : numpy.array<int>, len(len(X))
            The MDP state index of each row, or -1 where the state lookup
            failed.
        """
        X = X.copy()
        df = pd.DataFrame(X)
        # By using `predict_proba` and inserting randomness, we ensure that the
        # assumed y values are not always the majority.
        # df['y'] = (
        #     self.clf.predict_proba(X)[:,0] >= np.random.rand(len(X))
        # ).astype(int)
        if y is None:
            df["y"] = self.clf.predict(df)
        else:
            df["y"] = y

        # Get rid of any unused columns otherwise the state lookup breaks.
        df = df[self.mdp.x_cols + ["z", "y"]]

        # Transform state input using fitted state_reducer_
        for x in self.mdp.state_reducer_.keys():
            for x_val in self.mdp.state_reducer_[x]:
                default_val = self.mdp.state_reducer_[x][x_val]
                mask = df[x] == x_val
                try:
                    df.loc[mask, x] = default_val
                except TypeError:
                    # Casting to object matches pandas <3 behavior, which
                    # silently upcast columns (e.g. bool, str) to object
                    # when assigning an incompatible value.
                    df[x] = df[x].astype(object)
                    df.loc[mask, x] = default_val

        # Build each row's state key column-wise rather than with iterrows(),
        # which is orders of magnitude slower. The keys hash and compare equal
        # to the `tuple(row)` keys iterrows() produces, so every row resolves
        # to the same state.
        state_keys = list(zip(*(df[col].tolist() for col in df.columns)))
        lookup = self.mdp.reduced_state_lookup_
        states = np.fromiter(
            (lookup.get(key, -1) for key in state_keys),
            dtype=np.int64,
            count=len(state_keys),
        )
        not_found = states < 0
        n_state_lookup_errors = int(np.count_nonzero(not_found))

        if n_state_lookup_errors and logging.getLogger().isEnabledFor(logging.DEBUG):
            for i in np.flatnonzero(not_found):
                logging.debug("\tState Lookup Error: " + str(state_keys[i]))
                logging.debug(f"\tUsing default action: {self.default_action}")
        log_msg = f"""
            \t\tThere were {n_state_lookup_errors} state lookup errors when trying
            \t\tto set the optimal action for the input dataset. There are
            \t\t{len(df)} total input rows. So {n_state_lookup_errors}/{len(df)}
            \t\tforced to use the default action ({self.default_action}).
            """
        logging.debug(log_msg)
        return states

    def actions_for_states(self, states):
        """
        The policy's action for each of `states` (as returned by
        `lookup_states()`), using the default action where the state lookup
        failed.

        Parameters
        ----------
        states : numpy.array<int>
            MDP state indices, -1 for a failed state lookup.

        Returns
        -------
        actions : numpy.array<float>, len(len(states))
        """
        actions = np.zeros(len(states))
        found = states >= 0
        actions[found] = np.asarray(self.pi)[states[found]]
        actions[~found] = self.default_action
        return actions

    @property
    def pi_df_(self):
        """
        The policy represented as a dataframe: the self.mdp.ldf_ dataframe
        with the policy actions added as a column.

        Built on first access rather than in __init__, since constructing it
        runs a full prediction that most policies never need.
        """
        if self._pi_df is None:
            self._construct_pi_df()
        return self._pi_df

    def _construct_pi_df(self):
        """
        Adds the policy actions as a column to the self.mdp.ldf_ dataframe.

        Sets pi_df_
        ----------
        pi_df_ : pandas.DataFrame
            Policy represented as a dataframe.
        """
        pi_df = self.mdp.ldf_.iloc[:, :-2].drop_duplicates().copy()
        pi_df["a"] = self.predict(pi_df)
        self._pi_df = pi_df
        return None
