import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from ...interface import SignalBranch # type: ignore

class RSITree(SignalBranch):
    """Decision tree on RSI to predict next-day direction."""
    name = "tech_indicators_rsi"

    def __init__(self, max_depth: int = 3, min_samples_leaf: int = 50, random_state: int = 42):
        self.clf = DecisionTreeClassifier(
            max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=random_state
        )
        self._is_fit = False

    def fit(self, df: pd.DataFrame) -> None:
        # Expect columns: ['rsi','ret'] with chronological index
        feats = ["rsi"]
        X = df[feats].values[:-1]                       # features at t
        y = (df["ret"].shift(-1).values[:-1] > 0).astype(int)  # label: direction at t+1
        self.clf.fit(X, y)
        self._is_fit = True

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self._is_fit, "Call fit() before predict()."
        X = df[["rsi"]].values
        proba_up = self.clf.predict_proba(X)[:, 1]      # in [0,1]
        score = proba_up * 2 - 1                        # map -> [-1, +1]
        signal = np.sign(score)                         # -1, 0, +1
        confidence = np.abs(score).clip(0, 1)
        return pd.DataFrame({"signal": signal, "confidence": confidence}, index=df.index)
