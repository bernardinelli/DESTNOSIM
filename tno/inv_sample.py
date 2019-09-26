'''
Code adapted from https://tmramalho.github.io/blog/2013/12/16/how-to-do-inverse-transformation-sampling-in-scipy-and-numpy/
'''

import numpy as np
import scipy.interpolate as interpolate

def inverse_transform_sampling(x_values, y_values, n_bins=40, n_samples=1000):
    hist, bin_edges = np.histogram(x_values, bins=n_bins, density=True, weights = y_values)
    cum_values = np.zeros(bin_edges.shape)
    cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))
    inv_cdf = interpolate.CubicSpline(bin_edges, cum_values)
    r = np.random.rand(n_samples)
    return inv_cdf(r)


def inverse_cdf(x_values, y_values, n_bins=40):
    hist, bin_edges = np.histogram(x_values, bins=n_bins, density=True, weights = y_values)
    cum_values = np.zeros(bin_edges.shape)
    cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))
    inv_cdf = interpolate.CubicSpline(bin_edges, cum_values)
    return inv_cdf

def inverse_cdf_histogram(hist, bin_edges):
    '''
    Assumes that the histogram already exists somewhere, returns the sampling cdf
    '''
    cum_values = np.zeros(bin_edges.shape)
    cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))
    inv_cdf = interpolate.CubicSpline(bin_edges, cum_values)
    return inv_cdf