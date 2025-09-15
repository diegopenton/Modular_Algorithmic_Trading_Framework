import numpy as np
import pandas as pd
from src.branches.bollinger_reversion.branch_bollinger import BollingerMeanReversion

def _make_synth_df(n=300, seed=7):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0003, 0.01, size=n)
    close = 100 * np.exp(np.cumsum(rets))
    df = pd.DataFrame({"close": close})
    sma = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std(ddof=0)
    df["zscore"] = (df["close"] - sma) / (std.replace(0, np.nan))
    df["ret"] = np.log(df["close"]).diff()
    return df.dropna()

def test_bollinger_branch_shapes_and_bounds():
    df = _make_synth_df()
    b = BollingerMeanReversion()
    b.fit(df)
    out = b.predict(df.tail(50))
    assert set(out.columns) == {"signal", "confidence"}
    assert len(out) == 50
    assert out["confidence"].between(0, 1).all()
    assert out["signal"].isin([-1, 0, 1]).all()