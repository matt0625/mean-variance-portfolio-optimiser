import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from portfolio import port_return, port_vol, sharpe

def plot_correlation(returns, title="Correlation of daily returns"):
    corr = returns.corr()

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)

    fig.colorbar(im, ax=ax, label="correlation")
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax

def plot_frontier(tickers, mew_v, sigma_m, rf, w_mv, w_tan, W_front, clouds, extra=None):
    N = len(tickers)
    stock_vol = np.sqrt(np.diag(sigma_m))
    fig, ax = plt.subplots(figsize=(11, 8))

    # random portfolio clouds (faint, behind everything)
    for label, W in clouds.items():
        ax.scatter(port_vol(W, sigma_m), port_return(W, mew_v),
                   s=4, alpha=0.15, label=label, zorder=1)

    # frontier: upper branch solid, lower branch dashed
    f_vol = port_vol(W_front, sigma_m)
    f_ret = port_return(W_front, mew_v)
    upper = f_ret >= port_return(w_mv, mew_v)
    ax.plot(f_vol[upper], f_ret[upper], "k-", lw=2, label="efficient frontier", zorder=3)
    ax.plot(f_vol[~upper], f_ret[~upper], "k--", lw=1, alpha=0.5,
            label="inefficient branch", zorder=3)

    # extra constraints if given
    for label, (W, colour) in (extra or {}).items():
        ax.plot(port_vol(W, sigma_m), port_return(W, mew_v),
                color=colour, lw=2, label=label, zorder=3)

    # capital market line: from (0, rf) through the tangency portfolio
    s_tan = sharpe(w_tan, mew_v, sigma_m, rf)
    xmax = 1.5 * stock_vol.max()
    xs = np.array([0, xmax])
    ax.plot(xs, rf + s_tan * xs, color="tab:red", lw=1.2, label="capital market line", zorder=2)

    # individual stocks
    ax.scatter(stock_vol, mew_v, c="tab:blue", s=30, zorder=4, label="individual assets")
    for i, t in enumerate(tickers):
        ax.annotate(t, (stock_vol[i], mew_v[i]), xytext=(4, 4),
                    textcoords="offset points", fontsize=8)

    # special portfolios
    w_eq = np.ones(N) / N
    for w, name, marker, colour in [
        (w_mv, "min variance", "o", "green"),
        (w_tan, "tangency", "*", "red"),
        (w_eq, "equal weight", "s", "orange"),
    ]:
        ax.scatter(port_vol(w, sigma_m), port_return(w, mew_v), marker=marker,
                   c=colour, s=150 if marker == "*" else 70,
                   edgecolor="k", zorder=5, label=name)

    ax.set_xlim(0, xmax)
    ax.set_ylim(min(-0.05, mew_v.min() - 0.02), 1.5 * mew_v.max())
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("annualised volatility")
    ax.set_ylabel("annualised expected return")
    ax.set_title("Efficient frontier (training period")
    ax.legend(loc="upper left", markerscale=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, ax