'''
Code adapted from https://tmramalho.github.io/blog/2013/12/16/how-to-do-inverse-transformation-sampling-in-scipy-and-numpy/
'''

import numpy as np
import scipy.interpolate as interpolate

def inverse_transform_sampling(x_values, y_values, n_bins=40, n_samples=1000, kind = 'cubic'):
    '''
    Performs an inverse transform sampling algorithm for a distribution with provided x and y values

    Arguments:
    - x_values: range of the distribution
    - y_values: correspondend y values for the pdf, must be normalized
    - n_bins: number of bins for histogramming
    - n_samples: number of samples for the histogramming process
    - kind: interpolation type from scipy.interpolate.interp1d
    '''
    hist, bin_edges = np.histogram(x_values, bins=n_bins, density=True, weights = y_values)
    cum_values = np.zeros(bin_edges.shape)
    cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))
    inv_cdf = interpolate.interp1d(cum_values, bin_edges, kind)
    r = np.random.rand(n_samples)
    return inv_cdf(r)


def inverse_cdf(x_values, y_values, n_bins=40, kind = 'cubic'):
    '''
    Performs an inverse transform sampling algorithm for a cdf with provided x and y values

    Arguments:
    - x_values: range of the distribution
    - y_values: correspondend y values for the cdf, must be normalized
    - n_bins: number of bins for histogramming
    - n_samples: number of samples for the histogramming process
    - kind: interpolation type from scipy.interpolate.interp1d
    '''

    hist, bin_edges = np.histogram(x_values, bins=n_bins, density=True, weights = y_values)
    cum_values = np.zeros(bin_edges.shape)
    cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))
    inv_cdf = interpolate.interp1d(cum_values, bin_edges, kind)
    return inv_cdf

def inverse_cdf_histogram(hist, bin_edges, kind = 'cubic'):
    '''
    Assumes that the histogram already exists somewhere, returns the sampling cdf

    Arguments:
    - hist: histogram values at each bin (numpy style) 
    - bin_edges: edges of the bins (numpy style)
    - kind: interpolation type from scipy.interpolate.interp1d
    '''
    cum_values = np.zeros(bin_edges.shape)
    cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))
    inv_cdf = interpolate.interp1d(cum_values, bin_edges, kind)
    return inv_cdf