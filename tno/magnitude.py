'''
Series of convenience functions for working with magnitudes
'''
import numpy as np 
import astropy.table as tb 

from scipy.optimize import minimize

alpha_i  =  1.492983987794142 
beta_i   = -0.12310383521046273 
alpha_z  =  1.65452092361025 
beta_z   = -0.13315208762781655


def generate_colors(gr):
	'''
	Generates g - i and g - z colors from a nominal g - r colors, according to a linear fit 
	'''
	gi = alpha_i*gr + beta_i
	gz = alpha_z*gr + beta_z
	return gi , gz  

def shot_noise(mag, band):
	'''
	Return mean shot noise error expected as a function of magnitude in a given band
	'''
	shot_noise_amp = {'g' : -10.237155096761413, 'r' : -10.306198159582513, 'i' : -10.182735894436972, 'z' : -10.015274570909892}

	return np.power(10, shot_noise_amp[band] + 0.4*mag)/3600

def detprob(m, params):
    '''
    logit function
    params = (m0, k, c)
    '''
    m50, k, c = params
    logit = c/(1+np.exp(k*(m-m50)))
    return logit


def minusLogP(params, mdet, mnon, res_collect):
    '''
    Takes logit parameters and list of detected and non-detected magnitudes. 
    Returns negative log of pdf
    '''

    if params[2] > 1.:
        # this ensures that c cannot continue to rise above 1 by referencing 
        # the value of the previous trial in the optimizer
        res_collect.append(res_collect[-1] + (params[2] - 1.)*1e5)
        return res_collect[-1] + 1e3
    elif params[2] <= 0.:
        return res_collect[-1] + 1e3
    else:
        pdet = detprob(mdet,params)
        pnon = detprob(mnon,params)    
        result = np.sum(np.log(pdet))
        result += np.sum(np.log(1-pnon))
        return -result

def find_m50(m_det, m_miss):
	'''
	Computes the (m50, k, c) logit fit
	'''
	results_collector = [0]
	return minimize(minusLogP, (23, 5, 1), method='Powell', args=(m_det, m_miss, results_collector), tol=1e-3)




class BaseLightCurve:
	'''
	Base method for lightcurves, returning no shift 
	'''
	def __init__(self):
		pass
	def _sample(self, times):
		return np.zeros_like(times)
	def __call__(self, x):
		return self._sample(x)

class RandomPhaseLightCurve(BaseLightCurve):
	'''
	Returns a randomly sampled sinusoidal lightcurve with a given peak-to-peak amplitude
	'''
	def __init__(self, amplitude):
		self.amplitude = amplitude

	def _sample(self, times):
		phases = np.random.sample(times.shape)*2 *np.pi 
		return self.amplitude * np.sin(phases)/2

class SinusoidalLightCurve(BaseLightCurve):
	'''
	Given a certain known lightcurve with a given amplitude, period and phase, returns each point of the lightcurve
	'''
	def __init__(self, amplitude, period, phase):
		self.amplitude = amplitude
		self.period = period
		self.phase = phase

	def _sample(self, times):
		return self.amplitude * np.sin(2 * np.pi * times/self.period + self.phase)

