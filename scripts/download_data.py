import os, sys
import pandas as pd
import yfinance as yf

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

TICKERS = ["DBC", "EEM", "GLD", "HYG", "QQQ", "SPY", "TLT", "USO", "UUP", "VNQ"]

def ensure_real_market_checkpoints():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(project_root, "data", "real_market_checkpoints")
    os.makedirs(target_dir, exist_ok=True)
    
    train_path = os.path.join(target_dir, "train_prices.csv")
    test_path = os.path.join(target_dir, "test_prices.csv")
    
    if not os.path.exists(train_path):
        print("Downloading train_prices.csv (2010-2019)...", flush=True)
        df_train = yf.download(TICKERS, start="2010-01-01", end="2019-12-31", progress=False, auto_adjust=True)
        if isinstance(df_train.columns, pd.MultiIndex):
            df_train = df_train['Close'] if 'Close' in df_train.columns.get_level_values(0) else df_train.iloc[:, :len(TICKERS)]
        df_train = df_train[TICKERS].dropna()
        df_train.to_csv(train_path, encoding='utf-8')
        print(f"✓ Saved {train_path} ({len(df_train)} rows)", flush=True)
        
    if not os.path.exists(test_path):
        print("Downloading test_prices.csv (2020-2024)...", flush=True)
        df_test = yf.download(TICKERS, start="2020-01-01", end="2024-05-31", progress=False, auto_adjust=True)
        if isinstance(df_test.columns, pd.MultiIndex):
            df_test = df_test['Close'] if 'Close' in df_test.columns.get_level_values(0) else df_test.iloc[:, :len(TICKERS)]
        df_test = df_test[TICKERS].dropna()
        df_test.to_csv(test_path, encoding='utf-8')
        print(f"✓ Saved {test_path} ({len(df_test)} rows)", flush=True)

if __name__ == "__main__":
    ensure_real_market_checkpoints()
