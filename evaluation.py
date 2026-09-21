import numpy as np
import pandas as pd
from portfolio import port_return, port_vol, sharpe

# takes in all our different portfolios
def evaluation(portfolio, mew, sigma, test, rf):
    test_v = test.to_numpy()
    rows = {}
    for name, w in portfolio.items():
        # out of sample : portfolio's actually daily returns in test period
        daily = test_v @ w
        oos_ret = daily.mean() * 252
        oos_vol = daily.std(ddof=1) * np.sqrt(252)
        oos_sh = (oos_ret - rf) / oos_vol

        # in sample: from training mew and sigma
        is_sh = sharpe(w, mew, sigma, rf)

        # max drawdown: worst peak-to-trough fall of the cumulative curve
        cum = np.concatenate([[1.0], (1 + daily).cumprod()])
        max_dd = (cum / np.maximum.accumulate(cum) - 1).min()

        rows[name] = {
            "IS return": port_return(w, mew),
            "IS vol": port_vol(w, sigma),
            "IS Sharpe": is_sh,
            "OOS return": oos_ret,
            "OOS vol": oos_vol,
            "OOS Sharpe": oos_sh,
            "Sharpe gap": is_sh - oos_sh,
            "gross leverage": np.abs(w).sum(),
            "max weight": w.max(),
            "min weight": w.min(),
            "max drawdown": max_dd,
        }

    return pd.DataFrame(rows).T