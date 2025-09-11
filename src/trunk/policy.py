# src/trunk/policy.py
import pandas as pd

def apply_policy(ensemble: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """
    thresholds keys:
      score_min, conf_min, base_size, max_size
    returns: DataFrame(action, size)
    """
    score_min = thresholds.get("score_min", 0.25)
    conf_min  = thresholds.get("conf_min", 0.25)
    base_size = thresholds.get("base_size", 1.0)     # 1.0 = 100% of unit
    max_size  = thresholds.get("max_size", 1.0)

    out = []
    for _, row in ensemble.iterrows():
        s, c = float(row["score"]), float(row["confidence"])
        if abs(s) < score_min or c < conf_min:
            out.append(("HOLD", 0.0))
            continue
        side = "BUY" if s > 0 else "SELL"
        size = min(max_size, base_size * c)
        out.append((side, size))

    return pd.DataFrame(out, index=ensemble.index, columns=["action", "size"])