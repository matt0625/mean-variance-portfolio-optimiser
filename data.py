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


# Step 1: use yfinance to get the price data for 2015-2025, then use pandas to compute the returns
# optimisation: price data is cached in csv so we can directly access it after the first call
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




