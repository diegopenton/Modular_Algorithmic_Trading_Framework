# src/app.py
from features.loaders import load_prices  # 👈 import your new loader function

def main():
    # call your loader for Apple stock (AAPL) just as a test
    df = load_prices("AAPL", "2022-01-01", "2022-06-01")
    print("✅ Loaded data sample:")
    print(df.head())  # show the first few rows

if __name__ == "__main__":
    main()
