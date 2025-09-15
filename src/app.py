from src.features.loaders import load_prices, add_rsi, add_bollinger_zscore
from src.branches.tech_indicators.branch_rsi import RSITree
from src.branches.bollinger_reversion.branch_bollinger import BollingerMeanReversion

def main():
    df = load_prices("AAPL", "2020-01-01", "2024-01-01")

    # RSI branch
    df_rsi = add_rsi(df.copy(), 14)
    rsi = RSITree(max_depth=3, min_samples_leaf=50)
    rsi.fit(df_rsi)
    out_rsi = rsi.predict(df_rsi.tail(10))

    # Bollinger branch
    df_boll = add_bollinger_zscore(df.copy(), 20)
    boll = BollingerMeanReversion()
    boll.fit(df_boll)
    out_boll = boll.predict(df_boll.tail(10))

    print("✅ RSI (last 5):")
    print(out_rsi.tail())
    print("\n✅ Bollinger Mean-Reversion (last 5):")
    print(out_boll.tail())

if __name__ == "__main__":
    main()
