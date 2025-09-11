# src/backtest/engine.py
import argparse
import numpy as np
import pandas as pd

from src.features.loaders import load_prices, add_rsi
from src.branches.tech_indicators.branch_rsi import RSITree
from src.trunk.router import weighted_ensemble
from src.trunk.policy import apply_policy
from src.backtest.metrics import sharpe, max_drawdown

def run_backtest(ticker: str, start: str, end: str, costs_bps: float = 0.0):
    df = load_prices(ticker, start, end)
    df = add_rsi(df, 14)

    # --- branch
    rsi = RSITree(max_depth=3, min_samples_leaf=50)
    rsi.fit(df)
    preds = rsi.predict(df)

    # --- trunk combine (only RSI for now)
    ens = weighted_ensemble({"tech_indicators_rsi": preds}, {"tech_indicators_rsi": 1.0})
    decisions = apply_policy(ens, dict(score_min=0.2, conf_min=0.2, base_size=1.0, max_size=1.0))

    # --- backtest loop (daily close-to-close, position on next day)
    equity = [1.0]     # start at 1.0 = 100%
    pos = 0.0
    rets = df["ret"].reindex(decisions.index)  # align

    # transaction cost per trade (round turn) in bps of notional
    cost = costs_bps / 1e4

    for i in range(1, len(rets)):
        # Apply return to current position
        equity.append(equity[-1] * (1.0 + pos * rets.iloc[i]))

        # New decision for next bar
        act, size = decisions["action"].iloc[i], float(decisions["size"].iloc[i])
        new_pos = pos
        if act == "BUY":  new_pos = +size
        elif act == "SELL": new_pos = -size
        # charge cost if position changes
        if new_pos != pos:
            equity[-1] *= (1.0 - cost)
        pos = new_pos

    equity = pd.Series(equity, index=rets.index[:len(equity)])
    strat_rets = equity.pct_change().fillna(0.0)

    print(f"[{ticker}] {start} → {end}")
    print(f"Sharpe: {sharpe(strat_rets):.2f}   MaxDD: {max_drawdown(equity):.2%}")
    return equity, decisions

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default="AAPL")
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end",   default="2023-01-01")
    p.add_argument("--costs_bps", type=float, default=0.0)
    args = p.parse_args()
    run_backtest(args.ticker, args.start, args.end, args.costs_bps)

if __name__ == "__main__":
    main()