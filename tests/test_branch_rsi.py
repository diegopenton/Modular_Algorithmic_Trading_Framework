import numpy as np
import pandas as pd
from src.branches.tech_indicators.branch_rsi import RSITree

def _make_synth_df(n=300, seed=42):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.01, size=n)
    close = 100 * np.exp(np.cumsum(rets))
    df = pd.DataFrame({"close": close})
    # RSI(14)
    delta = df["close"].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    rs = up / (down + 1e-12)
    df["rsi"] = 100 - 100/(1 + rs)
    df["ret"] = np.log(df["close"]).diff()
    return df.dropna()

def test_rsi_branch_shapes_and_bounds():
    df = _make_synth_df()
    b = RSITree(max_depth=3, min_samples_leaf=10)
    b.fit(df)
    out = b.predict(df.tail(50))
    assert set(out.columns) == {"signal", "confidence"}
    assert len(out) == 50
    assert out["confidence"].between(0, 1).all()
    assert out["signal"].isin([-1, 0, 1]).all()
