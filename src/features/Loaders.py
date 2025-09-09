import pandas as pd
import numpy as np
import yfinance as yf

def load_prices(ticker: str, start: str, end: str, interval="1d") -> pd.DataFrame:
    """Download OHLCV data and compute log returns."""
    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True)
    df = df.rename(columns=str.lower).dropna()
    df["ret"] = np.log(df["close"]).diff()
    return df.dropna()
