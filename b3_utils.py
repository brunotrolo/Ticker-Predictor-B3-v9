
import pandas as pd

def load_b3_tickers():
    df = pd.read_csv("data/b3_tickers.csv")
    df["ticker"] = df["ticker"].str.upper()
    return df

def ensure_sa_suffix(ticker: str) -> str:
    if not ticker: return ""
    t = ticker.strip().upper()
    return t if t.endswith(".SA") else f"{t}.SA"

def is_known_b3_ticker(ticker: str) -> bool:
    df = load_b3_tickers()
    return ensure_sa_suffix(ticker) in set(df["ticker"])

def search_b3(query: str, limit: int=20):
    df = load_b3_tickers()
    if not query:
        return df.head(limit)
    q = query.lower()
    mask = df["ticker"].str.lower().str.contains(q) | df["name"].str.lower().str.contains(q)
    return df[mask].head(limit)
