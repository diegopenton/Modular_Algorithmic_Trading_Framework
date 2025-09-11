# src/backtest/walkforward.py
from dataclasses import dataclass
from typing import Iterable, Tuple, List
import numpy as np
import pandas as pd

from src.features.loaders import load_prices, add_rsi
from src.branches.tech_indicators.branch_rsi import RSITree
from src.trunk.router import weighted_ensemble
from src.trunk.policy import apply_policy
from src.backtest.metrics import sharpe, max_drawdown

@dataclass
class WindowCfg:
    train_days: int = 252      # ~1y trading days
    test_days: int  = 22       # ~1 month

def rolling_windows(idx: Iterable[pd.Timestamp], cfg: WindowCfg) -> List[Tuple[int,int,int,int]]:
    """Yield (train_start, train_end, test_start, test_end) as integer indices."""
    n = len(idx)
    out = []
    i0 = 0
    while True:
        tr_start = i0
        tr_end   = tr_start + cfg.train_days
        te_start = tr_end
        te_end   = te_start + cfg.test_days
        if te_end > n: break
        out.append((tr_start, tr_end, te_start, te_end))
        i0 += cfg.test_days
    return out

def wfv_rsi(ticker: str, start: str, end: str, cfg=WindowCfg()) -> pd.DataFrame:
    df = load_prices(ticker, start, end)
    df = add_rsi(df, 14)
    idx = df.index.to_list()

    rows = []
    for (tr_s, tr_e, te_s, te_e) in rolling_windows(idx, cfg):
        df_tr = df.iloc[tr_s:tr_e]
        df_te = df.iloc[te_s:te_e]

        # --- train branch
        rsi = RSITree(max_depth=3, min_samples_leaf=50)
        rsi.fit(df_tr)

        # --- predict on test
        preds = rsi.predict(df_te)

        # --- combine (only RSI for now) + policy
        ens  = weighted_ensemble({"rsi": preds}, {"rsi": 1.0})
        dec  = apply_policy(ens, dict(score_min=0.2, conf_min=0.2, base_size=1.0, max_size=1.0))

        # --- backtest on test window (daily, no costs for now)
        rets = df["ret"].iloc[te_s:te_e].reindex(dec.index)
        equity = [1.0]; pos = 0.0
        for i in range(1, len(rets)):
            equity.append(equity[-1] * (1.0 + pos * rets.iloc[i]))
            act, size = dec["action"].iloc[i], float(dec["size"].iloc[i])
            if   act == "BUY":  pos = +size
            elif act == "SELL": pos = -size
            else:               pos = pos

        eq = pd.Series(equity, index=rets.index[:len(equity)])
        sr = sharpe(eq.pct_change().fillna(0))
        dd = max_drawdown(eq)
        rows.append({
            "train_start": df.index[tr_s], "train_end": df.index[tr_e-1],
            "test_start": df.index[te_s],  "test_end":  df.index[te_e-1],
            "sharpe": sr, "maxdd": dd
        })

    return pd.DataFrame(rows)

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default="AAPL")
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end",   default="2024-01-01")
    p.add_argument("--train_days", type=int, default=252)
    p.add_argument("--test_days",  type=int, default=22)
    args = p.parse_args()

    res = wfv_rsi(args.ticker, args.start, args.end, WindowCfg(args.train_days, args.test_days))
    print(res[["train_end","test_start","test_end","sharpe","maxdd"]])
    print("\nAggregated:")
    print(f"Median Sharpe: {res['sharpe'].median():.2f}  |  Windows: {len(res)}")
    print(f"Sharpe>0 windows: {(res['sharpe']>0).mean():.1%}")

if __name__ == "__main__":
    main()