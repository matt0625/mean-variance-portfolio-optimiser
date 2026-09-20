import numpy as np
import matplotlib.pyplot as plt
from data import load_prices, compute_returns, TICKERS
from portfolio import port_vol, port_return, sharpe, min_variance, tangency, constrained_frontier_point, sweep_frontier
from plotting import plot_frontier

RF = 0.02 # risk free rate assumption since real one varies over time but aim of project is to learn the machinery
SPLIT = "2020-12-31"

def main():
    prices = load_prices()
    returns = compute_returns(prices)
    N = len(TICKERS)

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

    w_mv = min_variance(sigma_m)
    w_tan = tangency(mew_v, sigma_m, RF)


    a = np.linspace(-3, 2, 500)[:, None]
    W_front = (a * w_mv) + (1 - a) * w_tan


    rng = np.random.default_rng(0)
    long_uniform = rng.dirichlet(np.ones(N), size=5000)
    long_conc = rng.dirichlet(np.ones(N) * 0.2, size=5000)

    W_short = rng.normal(size=(5000, N))
    s = W_short.sum(axis=1)
    keep = np.abs(s) > 0.1  # drop rows whose sum is near zero
    W_short = W_short[keep] / s[keep][:, None]

    clouds = {"long-only (uniform)": long_uniform,
              "long-only (concentrated)": long_conc,
              "unconstrained (shorting)": W_short}


    '''
    # testing to check that no random portfolio sits above the frontier using interpolation
    frontier_returns = port_return(W_front, mew_v)
    frontier_vol = port_vol(W_front, sigma_m)
    
    upper = frontier_returns >= port_return(w_mv, mew_v)
    order = np.argsort(frontier_vol[upper])
    xp = frontier_vol[upper][order]
    fp = frontier_returns[upper][order]

    def check_below_frontier(W, label, tol=1e-4):
        v = port_vol(W, sigma_m)
        r = port_return(W, mew_v)
        # only test where frontier is defined
        inside = (v >= xp.min()) & (v <= xp.max())
        gap = r[inside] - np.interp(v[inside], xp, fp)
        print(label, "worst excess over frontier: ", gap.max(), "->", gap.max() <= tol)

    for label, W in clouds.items():
        check_below_frontier(W, label)

    '''

    # adding constraints to portfolios i.e. no shorting, restrictions on diversification
    w_lo_mv, W_lo = sweep_frontier(sigma_m, mew_v)
    w_cap_mv, W_cap = sweep_frontier(sigma_m, mew_v, cap=0.2)

    extra = {"long-only frontier": (W_lo, "tab:green"),
             "long-only, cap 20%": (W_cap, "tab:purple")}
    plot_frontier(tickers, mew_v, sigma_m, RF, w_mv, w_tan, W_front, clouds, extra=extra)

    s_lo = sharpe(W_lo, mew_v, sigma_m, RF)
    w_lo_sharpe = W_lo[np.argmax(s_lo)]

    s_cap = sharpe(W_cap, mew_v, sigma_m, RF)
    w_cap_sharpe = W_cap[np.argmax(s_cap)]
    plt.show()



if __name__ == "__main__":
    main()