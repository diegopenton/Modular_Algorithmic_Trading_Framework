import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.interface import SignalBranch

class BollingerMeanReversion(SignalBranch):
    """
    Mean-reversion branch using Bollinger z-score.
    Predicts next-day direction with Logistic Regression on zscore.
    """
    name = "bollinger_reversion"

    def __init__(self, C: float = 1.0, class_weight="balanced", random_state: int = 42):
        self.clf = LogisticRegression(
            C=C, class_weight=class_weight, random_state=random_state, max_iter=1000
        )
        self._is_fit = False

    def fit(self, df: pd.DataFrame) -> None:
        # Expect columns: ['zscore','ret']
        X = df[["zscore"]].values[:-1]                        # features at t
        y = (df["ret"].shift(-1).values[:-1] > 0).astype(int) # label at t+1
        self.clf.fit(X, y)
        self._is_fit = True

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self._is_fit, "Call fit() before predict()."
        proba_up = self.clf.predict_proba(df[["zscore"]].values)[:, 1]  # [0..1]
        score = proba_up * 2 - 1                                        # [-1..+1]
        signal = np.sign(score)                                         # -1,0,+1
        confidence = np.abs(score).clip(0, 1)
        return pd.DataFrame({"signal": signal, "confidence": confidence}, index=df.index)
