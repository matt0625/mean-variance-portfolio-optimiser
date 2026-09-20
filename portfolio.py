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
    numerator = np.linalg.solve(sigma, 1)
    # denominator is just the sum of entries to z, so..
    return numerator / np.sum(numerator)



def tangency(mew, sigma, rf):
    return