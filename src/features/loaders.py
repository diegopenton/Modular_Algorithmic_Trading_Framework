import pandas as pd
import numpy as np
import yfinance as yf

def load_prices(ticker: str, start: str, end: str, interval="1d") -> pd.DataFrame:
    """Download OHLCV data and compute log returns."""
    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True)
    df = df.rename(columns=str.lower).dropna()
    df["ret"] = np.log(df["close"]).diff()
    return df.dropna()

def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Add RSI(window) to df; returns a DataFrame without NaNs from rolling calc."""
    delta = df["close"].diff()
    up = delta.clip(lower=0).rolling(window).mean()
    down = (-delta.clip(upper=0)).rolling(window).mean()
    rs = up / (down + 1e-12)
    df["rsi"] = 100 - 100 / (1 + rs)
    return df.dropna()

def add_bollinger_zscore(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Adds 'zscore' = (close - SMA(window)) / rolling_std(window).
    Returns a DataFrame without NaNs from rolling calculations.
    """
    sma = df["close"].rolling(window).mean()
    std = df["close"].rolling(window).std(ddof=0)
    df["zscore"] = (df["close"] - sma) / (std.replace(0, np.nan))
    return df.dropna()