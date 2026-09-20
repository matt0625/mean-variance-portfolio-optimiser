import numpy as np
import matplotlib.pyplot as plt
from data import load_prices, compute_returns, TICKERS
from portfolio import port_vol, port_return, sharpe

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
    mew_v = mew.to_numpy()
    sigma = train.cov() * 252
    sigma_m = sigma.to_numpy()
    tickers = list(mew.index)


    '''
    # testing a stack of single asset portfolios; we want both cases to be true,
    # as the matrix multiplication should just pick out mew[i] and sigma[i, i] when only asset i has weight
    I = np.eye(len(tickers))

    print(np.allclose(port_return(I, mew_v), mew_v))
    print(np.allclose(port_vol(I, sigma_m), np.sqrt(np.diag(sigma_m))))
    
    # testing the hard way: calculating return and volatility without summary statistics
    train_v = train.to_numpy()
    w = np.ones(20) / 20
    daily = train_v @ w

    print(daily.mean() * 252, port_return(w, mew_v))
    print(daily.std(ddof=1) * np.sqrt(252), port_vol(w, sigma_m))
    '''

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