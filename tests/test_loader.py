import pandas as pd
from src.features.loaders import load_prices

def test_loader_returns_dataframe():
    df = load_prices("AAPL", "2022-01-01", "2022-02-01")
    assert isinstance(df, pd.DataFrame)
    assert "close" in df.columns
    assert "ret" in df.columns
    assert not df.empty
