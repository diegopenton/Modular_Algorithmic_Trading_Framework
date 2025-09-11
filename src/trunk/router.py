# src/trunk/router.py
from typing import Dict, List
import pandas as pd

def weighted_ensemble(outputs: Dict[str, pd.DataFrame], weights: Dict[str, float]) -> pd.DataFrame:
    """
    outputs: {branch_name: DataFrame(signal, confidence)}
    weights: {branch_name: weight}  (missing names default to 0)
    returns: DataFrame(score, confidence)
    """
    # ensure same index across branches
    idx = next(iter(outputs.values())).index
    for df in outputs.values():
        if not df.index.equals(idx):
            raise ValueError("All branch outputs must share the same index")

    # weighted sum of signals; confidence as weighted avg of |weights|
    score = pd.Series(0.0, index=idx)
    conf  = pd.Series(0.0, index=idx)

    for name, df in outputs.items():
        w = float(weights.get(name, 0.0))
        score = score.add(df["signal"] * w, fill_value=0.0)
        conf  = conf.add(df["confidence"].abs() * abs(w), fill_value=0.0)

    # normalize confidence to [0,1] if sum|w| > 1
    denom = max(1.0, sum(abs(w) for w in weights.values()))
    conf = (conf / denom).clip(0, 1)

    return pd.DataFrame({"score": score, "confidence": conf}, index=idx)