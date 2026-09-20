import numpy as np
import cvxpy as cp

def port_return(w, mew):
    return w @ mew

def port_vol(w, sigma):
    var = ((w @ sigma) * w).sum(axis=-1)
    return np.sqrt(var)
def sharpe(w, mew, sigma, rf):
    return (port_return(w, mew) - rf) / port_vol(w, sigma)

def min_variance(sigma):
    # a quick derivation using lagrangian multipliers gives that w* = (sigma)-1 * 1 / 1^T * (sigma)-1 * 1
    ones = np.ones(sigma.shape[0])
    z = np.linalg.solve(sigma, ones)
    # denominator is just the sum of entries to z, so...
    return z / z.sum()


def tangency(mew, sigma, rf):
    # similar lagrangian derivation method this time with ratios
    m = mew - rf
    z = np.linalg.solve(sigma, m)
    return z / z.sum()

def constrained_frontier_point(sigma, mew, target=None, long_only=True, cap=None):
    # constrained problem means lagrangian no longer enough
    # we describe it intuitively to cvxpy and let it solve it for us
    n = len(mew)
    w = cp.Variable(n)
    cons = [cp.sum(w) == 1]
    if target is not None:
        cons.append(mew @ w == target)
    if long_only:
        cons.append(w >= 0)
    if cap is not None:
        cons.append(w <= cap)
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(sigma))), cons)
    prob.solve()
    return w.value  # None if infeasible


def max_return(mew, cap=None):
    w = cp.Variable(len(mew))
    cons = [cp.sum(w) == 1, w >= 0] + ([w <= cap] if cap is not None else [])
    cp.Problem(cp.Maximize(mew @ w), cons).solve()
    return mew @ w.value
def sweep_frontier(sigma, mew, cap=None, k=60):
    w_mv = constrained_frontier_point(sigma, mew, target=None, cap=cap)
    r_lo = mew @ w_mv
    r_hi = max_return(mew, cap) - 1e-6
    ws = [constrained_frontier_point(sigma, mew, t, cap=cap)
          for t in np.linspace(r_lo, r_hi, k)]
    W = np.array([w for w in ws if w is not None])
    W = np.clip(W, 0, None)
    W /= W.sum(axis=1, keepdims=True)  # clean tiny negatives, re-normalise
    return w_mv, W

