import numpy as np
import matplotlib.pyplot as plt
from data import load_prices, compute_returns, TICKERS

RF = 0.02 # risk free rate assumption since real one varies over time but aim of project is to learn the machinery
SPLIT = "2020-12-31"

def main():
    prices = load_prices()
    returns = compute_returns(prices)

    #step 2
    train = returns.loc[:SPLIT]
    test = returns.loc[SPLIT:]

    # scale linearly while deviation scales at root 252
    mew = train.mean() * 252
    sigma = train.cov() * 252

    corr = train.corr()

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=98)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)

    fig.colorbar(im, ax=ax, label="correlation")
    ax.set_title("Correlation of daily returns (training period)")
    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()