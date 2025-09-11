# src/app.py
from .features.loaders import load_prices, add_rsi
from .branches.tech_indicators.branch_rsi import RSITree

def main():
    df = load_prices("AAPL", "2022-01-01", "2023-01-01")
    df = add_rsi(df, 14)

    branch = RSITree(max_depth=3, min_samples_leaf=50)
    branch.fit(df)
    out = branch.predict(df.tail(30))  # demo on last 30 rows

    print("✅ RSI branch output (last 5 rows):")
    print(out.tail())

if __name__ == "__main__":
    main()