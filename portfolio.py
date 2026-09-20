import numpy as np

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

