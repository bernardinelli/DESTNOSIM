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

	Arguments:
	- gr: array of gr colors
	'''
	gi = alpha_i*gr + beta_i
	gz = alpha_z*gr + beta_z
	return gi , gz  

def shot_noise(mag, band):
	'''
	Return mean shot noise error expected as a function of magnitude in a given band

	Arguments:
	- mag: magnitudes for the shot noise error (array)
	- band: single band, must be one of 'g', 'r', 'i', 'z'
	'''
	shot_noise_amp = {'g' : -10.237155096761413, 'r' : -10.306198159582513, 'i' : -10.182735894436972, 'z' : -10.015274570909892}

	return np.power(10, shot_noise_amp[band] + 0.4*mag)/3600

def detprob_logit(m, params):
    '''
    logit function
    params = (m50, k, c)

    Arguments:
    - m: magnitude argument
    - params: tuple with m50, k and c
    '''
    m50, k, c = params
    logit = c/(1+np.exp(k*(m-m50)))
    return logit

def detprob_double(m, params):
    '''
    double logit function
    params = (m0, k1, k2, c)

     Arguments:
    - m: magnitude argument
    - params: tuple with m0, k1, k2 and c
    '''
    m50, k1, k2, c = params
    logit = c/(1+np.exp(k1*(m-m50)))/(1+np.exp(k2*(m-m50)))
    return logit

def detprob_piecewise(x, params, split):
	'''
	Piecewise logit function
	params = (r50_1, r50_2, kappa1, kappa2, c)

	Arguments:
	- m: magnitude argument
	- params: tuple with r50_1, c, r50_2, kappa1 and kappa2 
	'''

	r50_1, r50_2, kappa1, kappa2, c = params
	p1 = lambda x : detprob_logit(x, [r50_1, kappa1, c])
	p2 = lambda x : detprob_logit(x, [r50_2, kappa2, c])
	return np.piecewise(x,  [x<split, x>=split], [p1, p2])


def minusLogP(params, mdet, mnon, res_collect, detprob):
    '''
    Takes logit parameters and list of detected and non-detected magnitudes. 
    Returns negative log of pdf


    Arguments:
    - params: tuple of parameters for the fit
    - mdet: array of detected magnitudes
    - mnon: array of missed magnitudes
    - res_collect: list of previous failed results
    - detprob: functional form (eg detprob_logit or detprob_double)
    '''

    if params[-1] > 1.:
        # this ensures that c cannot continue to rise above 1 by referencing 
        # the value of the previous trial in the optimizer
        res_collect.append(res_collect[-1] + (params[-1] - 1.)*1e5)
        return res_collect[-1] + 1e7
    elif params[-1] <= 0.:
        return res_collect[-1] + 1e7
    else:
        pdet = detprob(mdet,params)
        pnon = detprob(mnon,params)    
        result = np.sum(np.log(pdet))
        result += np.sum(np.log(1-pnon))
        return -result

def find_m50(m_det, m_miss, detprob = detprob_logit, init = (23,0.9,5)):
	'''
	Fits the provided function by taking the product of the detected probabilities and 1-non-detection probabilities

	Arguments:
	- m_det: array of detected magnitudes
    - m_non: array of missed magnitudes
    - detprob: functional form (eg detprob_logit or detprob_double)
    - init: tuple of initialization values, must match the arguments of detprob
	'''
	results_collector = [0]
	return minimize(minusLogP, init, method='Powell', args=(m_det, m_miss, results_collector, detprob), tol=1e-3)


def magnitude_physical(albedo, diameter, band):
	'''
	Finds the absolute magnitude of an object given an albedo in a certain band and a diameter (in km)
	Assumes phi(alpha) = 1
	Solar AB magnitudes from Willmer 2018 ApJS 236 47 (https://iopscience.iop.org/article/10.3847/1538-4365/aabfdf)

	Arguments:
	- albedo: albedo value
	- diameter: diameter of the object in km
	- band: one of 'g', 'r', 'i', 'z', 'Y' for absolute magnitude look-up
	'''

	solar_m = {'g' : -26.52, 'r' : -26.96, 'i' : -27.05, 'z' : -27.07, 'Y' : -27.07}

	log_term = albedo * diameter * diameter/9e16  

	return solar_m[band] - 2.5*np.log10(log_term)


def size_from_mag(albedo, magnitude, band):
	'''
	Finds the diameter (in km) of an object given an albedo in a certain band and an absolute magnitude 
	Assumes phi(alpha) = 10
	Solar AB magnitudes from Willmer 2018 ApJS 236 47 (https://iopscience.iop.org/article/10.3847/1538-4365/aabfdf)

	Arguments:
	- albedo: albedo value
	- magnitude: absolute magnitude of the object
	- band: one of 'g', 'r', 'i', 'z', 'Y' for absolute magnitude look-up

	'''

	solar_m = {'g' : -26.52, 'r' : -26.96, 'i' : -27.05, 'z' : -27.07, 'Y' : -27.07}

	power = np.power(10, 0.4*(solar_m[band] - magnitude))  

	return np.sqrt(9e16 * power/albedo)


class BaseLightCurve:
	'''
	Base method for lightcurves, returning no shift independently of the phase
	'''
	def __init__(self):
		'''
		Initialization function. Doesn't do anything
		'''
		pass
	def _sample(self, times):
		'''
		Samples this lightcurve

		Argument:
		- times: array of times
		'''
		return np.zeros_like(times)
	def __call__(self, x):
		return self._sample(x)

class RandomPhaseLightCurve(BaseLightCurve):
	'''
	Returns a randomly sampled sinusoidal lightcurve with a given peak-to-peak amplitude
	'''
	def __init__(self, amplitude):
		'''
		Initialization function

		Argument:
		- amplitude: amplitude of the light curve (will oscillate between +- amplitude/2)
		'''
		self.amplitude = amplitude

	def _sample(self, times):
		'''
		Samples the light curve

		Argument:
		- times: array of times
		'''
		phases = np.random.sample(times.shape)*2 *np.pi 
		return self.amplitude * np.sin(phases)/2

class SinusoidalLightCurve(BaseLightCurve):
	'''
	Given a certain known lightcurve with a given amplitude, period and phase, returns each point of the lightcurve
	'''
	def __init__(self, amplitude, period, phase):
		'''
		Initialization function

		Argument:
		- amplitude: amplitude of the light curve (will oscillate between +- amplitude/2)
		- period: period of the oscillation
		- phase: phase of the oscillation, in radians
		'''
		self.amplitude = amplitude
		self.period = period
		self.phase = phase

	def _sample(self, times):
		'''
		Samples the light curve

		Argument:
		- times: array of times
		'''
		return self.amplitude * np.sin(2 * np.pi * times/self.period + self.phase)

