# src/backtest/metrics.py
import numpy as np

def sharpe(returns, freq=252, eps=1e-12):
    r = np.array(returns)
    if r.size == 0: return 0.0
    mu, sd = r.mean(), r.std(ddof=1)
    return (mu / (sd + eps)) * np.sqrt(freq)

def max_drawdown(equity):
    eq = np.array(equity)
    peaks = np.maximum.accumulate(eq)
    dd = (eq - peaks) / peaks
    return dd.min() if dd.size else 0.0