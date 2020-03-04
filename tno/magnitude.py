'''
Series of convenience functions for working with magnitudes
'''
import numpy as np 
import astropy.table as tb 

from scipy.integrate import trapz

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

