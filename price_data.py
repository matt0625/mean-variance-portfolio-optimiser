import yfinance as yf
import pandas as pd
from pathlib import Path


CACHE = Path("cache.csv")
START = "2015-01-01"
END = "2025-12-31"
TICKERS = [
    "AAPL", "MSFT", "NVDA",   # tech
    "JPM", "BAC",             # banks
    "XOM", "CVX",             # energy
    "JNJ", "PFE", "UNH",      # healthcare
    "PG", "KO", "WMT",        # consumer staples
    "AMZN", "MCD",            # consumer discretionary
    "NEE", "DUK",             # utilities
    "CAT",                    # industrials
    "TLT",                    # long-term US Treasury bond ETF
    "GLD",                    # gold ETF
]


def load_prices(tickers=TICKERS, start=START, end=END, cache=CACHE):
    if cache.exists():
        data = pd.read_csv(cache, index_col=0, parse_dates=True)

    else:
        data = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
        data = data.dropna()
        data.to_csv(cache)

    return data[tickers]


def compute_returns(prices):
    return prices.pct_change().dropna()




