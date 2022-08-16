'''
Sampler classes for many different distributions. These are all based on the `BaseDistribution` class, 
and by subclassing it any distribution can be implemented. The JointDistribution allows one to construct a distribution of 
the form 
	alpha p_1(x) + (1-alpha) p_2(x),
where 0 <= alpha <= 1 and p_1, p_2 are two distributions

'''
import numpy as np
from .inv_sample import *

class BaseDistribution:
	'''
	Base class that all others inherit from. It corresponds to a delta function at zero. 
	'''
	def __init__(self):
		'''
		Initialization function. Doesn't do anything
		'''
		pass
	def __add__(self, other):
		return JointDistribution(self, other, 0.5)
	def __len__(self):
		return 0

	def sample(self, n):
		'''
		Samples from this distribution

		Arguments:
		- n: number of samples
		'''
		return np.zeros(n)


class AnalyticDistribution(BaseDistribution):
	'''	
	Given an analytic probability density function, samples from it using the inverse transform sampling algorithm from inv_sample.py
	'''
	def __init__(self, x_min, x_max, function, n_samp = 1000, n_bins = 100):
		'''
		Initialization function

		Arguments:
		- x_min: minimum value for the distribution
		- x_max: maximum value for the distribution
		- function: functional form of the distribution (eg a lambda function)
		- n_samp: number of samples for the inv_sample code
		- n_bins: number of bins for for the inv_sample code
		'''
		self.f = function
		self.x_min = x_min
		self.x_max = x_max
		self.n_samp = n_samp
		self.n_bins = n_bins
		self._constructSampler() 

	def _constructSampler(self):
		x = np.linspace(self.x_min, self.x_max, self.n_samp)
		y = self.f(x)
		self.invCDF = inverse_cdf(x, y, self.n_bins)

	def sample(self, n):
		'''
		Samples from this distribution

		Arguments:
		- n: number of samples
		'''
		r = np.random.rand(n)
		return self.invCDF(r)

	def __add__(self, other):
		x_min = np.max(self.x_min, other.x_min)
		x_max = np.min(self.x_max, other.x_max)
		f = lambda x : self.f(x) + other.f(x)
		n_samp = np.max(self.n_samp, other.n_samp)
		n_bins = np.max(self.n_bins, other.n_bins)

		return AnalyticDistribution(x_min, x_max, f, n_samp, n_bins)

class DistributionFromHistogram(AnalyticDistribution):
	'''
	Samples from the provided histogram using an inverse sampling algorithm
	'''
	def __init__(self, bins, pdf):
		'''
		Initialization function

		Arguments:
		- bins: bin edges of the histogram (numpy style)
		- pdf: value of the histogram at the bins (numpy style)
		'''
		self.bins = bins
		self.pdf = pdf 

		self._constructSampler()

	def _constructSampler(self):
		self.invCDF = inverse_cdf_histogram(bin_edges = self.bins, hist = self.pdf)

		
class DeltaFunction(BaseDistribution):
	'''
	Delta function centered at the provided location
	'''
	def __init__(self, loc):
		'''
		Initialization function

		Arguments:
		- loc: location of the delta function peak
		'''
		self.loc = loc
		pass
	def sample(self, n):
		'''
		Samples from this distribution

		Arguments:
		- n: number of samples
		'''

		return self.loc * np.ones(n)

class PowerLaw(BaseDistribution):
	'''
	Power law of the form x^slope, defined between x_min and x_max
	'''
	def __init__(self, slope, x_min, x_max):
		'''
		Initialization function

		Arguments:
		- slope: slope - 1 of the power law (for numpy reasons)
		- x_min: minimum value for the distribution
		- x_max: maximum value for the distribution
  
		'''
		self.slope = slope
		self.x_min = x_min
		self.x_max = x_max
		self.scale = x_max - x_min
		self.norm = self._normalization()

	def sample(self, n):
		'''
		Samples from this distribution

		Arguments:
		- n: number of samples
		'''

		samp = np.random.power(self.slope + 1, size=n)
		return self.x_min + self.scale * samp
	
	def _normalization(self):
		return (self.x_max**(self.slope + 1) - self.x_min**(self.slope + 1))/(self.slope + 1)

class JointDistribution(BaseDistribution):
	'''
	This returns a distribution that approximately samples alpha dist_1 + (1-alpha) dist_2
	This is not exact, as it forces the proportionalities for the sampling
	'''
	def __init__(self, distribution_1, distribution_2, alpha):
		'''
		Initialization function

		Arguments:
		- distribution_1: first distribution object, 100*alpha% of the samples will come from it 
		- distribution_2: second distribution object, 100*(1-alpha)% of the samples will come from it 
		- alpha: relative size between distributions
		'''
		self.dist_1 = distribution_1
		self.dist_2 = distribution_2
		if alpha > 1 or alpha < 0:
			raise ValueError("The value of alpha must be between 0 and 1")
		self.alpha = alpha

	def sample(self, n):
		'''
		Samples from this distribution

		Arguments:
		- n: number of samples
		'''

		n_frac = int(self.alpha*n)
		samp1 = self.dist_1.sample(n_frac)
		samp2 = self.dist_2.sample(n - n_frac)

		samp = np.append(samp1, samp2)
		perm = np.random.permutation(n)
		return samp[perm]

class Uniform(PowerLaw):
	'''
	Creates a uniform distribution between two different values
	'''
	def __init__(self, x_min, x_max):
		'''
		Initialization function
		
		Arguments:
		- x_min: minimum value for the distribution
		- x_max: maximum value for the distribution
		'''
		PowerLaw.__init__(self, 0, x_min, x_max)

class GaussianDistribution(BaseDistribution):
	def __init__(self, mu, sigma):
		self.mu = mu 
		self.sigma = sigma
	def sample(self, n):
		return np.random.normal(size=n, loc=self.mu, scale=self.sigma)
		

class Logarithmic(Uniform):
	'''
	Function for logarithmically distributed samples between two values
	'''
	def __init__(self, x_min, x_max):
		'''
		Initialization function
		
		Arguments:
		- x_min: minimum value for the distribution
		- x_max: maximum value for the distribution
		'''
		self._sampler = Uniform(np.log(x_min), np.log(x_max))
		self.x_min = x_min
		self.x_max = x_max

	def sample(self, n):
		'''
		Samples from this distribution

		Arguments:
		- n: number of samples
		'''
		return np.exp(self._sampler.sample(n))

class BrokenPowerLaw(AnalyticDistribution):
	'''
	Creates a broken power law with two different slopes, transitioning at a certain break point
	'''
	def __init__(self, slope_1, slope_2, x_min, x_max, x_break, x_norm = None):
		'''
		Initialization function

		Arguments:
		- slope_1: first slope, for x < x_break
		- slope_2: second slope, for x > x_break
		- x_min: minimum value for the distribution
		- x_max: maximum value for the distribution
		- x_break: break point of the distribution
		- x_norm: normalization for the x values, for numerical stability
		'''
		self.x_break = x_break
		self.slope_1 = slope_1
		self.slope_2 = slope_2
		if x_norm == None:
			x_norm = (x_min + x_max)/2
		self.x_norm = x_norm

		self.f = lambda x : np.piecewise(x, [ x <= self.x_break, x > self.x_break], [lambda x: np.power(x - x_norm, self.slope_1), 
														lambda x: np.power(x - x_norm, self.slope_2) * np.power(self.x_break - x_norm, self.slope_1 - self.slope_2)])
	
		AnalyticDistribution.__init__(self, x_min, x_max, self.f)

class DoublePowerLaw(AnalyticDistribution):
	'''
	Creates a double power law with two different slopes that become equal at a certain point x_eq
	'''

	def __init__(self, slope_1, slope_2, x_min, x_max, x_eq, x_shift):
		'''
		Initialization function

		Arguments:
		- slope_1: first slope, for x < x_break
		- slope_2: second slope, for x > x_break
		- x_min: minimum value for the distribution
		- x_max: maximum value for the distribution
		- x_eq: point where the two power laws are equal
		- x_shift: the power laws use (x - x_shift), this value defines the shift point
		'''

		self.x_eq = x_eq
		self.slope_1 = slope_1
		self.slope_2 = slope_2
		self.x_shift = x_shift
		self.x_min = x_min
		self.x_max = x_max
		c = np.power(10, (slope_2 - slope_1)*(x_eq - x_shift))
		self.c = c
		self.f = lambda x : (1 + c)/(np.power(10, -slope_1 * (x - x_shift)) + c * np.power(10, -slope_2 * (x - x_shift)))


		AnalyticDistribution.__init__(self, x_min, x_max, self.f)

class RollingPowerLaw(AnalyticDistribution):
	'''
	Creates a rolling power law with two different slopes, one linear and one quadratic in the exponential
	'''
	def __init__(self, slope_1, slope_2, x_min, x_max, x_shift):
		'''
		Initialization function

		Arguments:
		- slope_1: first slope, for x < x_break
		- slope_2: second slope, for x > x_break
		- x_min: minimum value for the distribution
		- x_max: maximum value for the distribution
		- x_shift: the power laws use (x - x_shift), this value defines the shift point
		'''
		self.slope = slope
		self.derivative = derivative
		self.x_shift = x_shift
		self.x_min = x_min
		self.x_max = x_max
		self.f = lambda x : np.power(10, slope * (x - x_shift) + derivative * (x - x_shift)**2 )
		AnalyticDistribution.__init__(self, x_min, x_max, self.f)


class BrownDistribution(AnalyticDistribution):
	'''
	Brown distribution (Brown 2001) of the form sin(x) * exp(-(x-x_center)**2/2 sigma**2)
	Assumes x is in degrees!
	'''
	def __init__(self, x_min, x_max, sigma, x_center):
		'''
		Initialization function

		Arguments:
		- x_min: minimum value for the distribution
		- x_max: maximum value for the distribution
		- sigma: sigma of the Gaussian term
		- x_center: center of the Gaussian term
		'''
		self.sigma = sigma
		self.f = lambda x : np.sin(x * np.pi/180) * np.exp(- ((x - x_center)/sigma)**2/2.)
		AnalyticDistribution.__init__(self, x_min, x_max, self.f)

class SinusoidalDistribution(AnalyticDistribution):
	'''
	Sinusoidal distribution
	'''
	def __init__(self, x_min, x_max):
		'''
		Initialization function
		
		Arguments:
		- x_min: minimum value for the distribution
		- x_max: maximum value for the distribution
		'''
		self.f = lambda x : np.sin(x * np.pi/180)
		AnalyticDistribution.__init__(self, x_min, x_max, self.f)

class RayleighDistribution(AnalyticDistribution):
	'''
	Rayleigh distribution of the form (x/sigma^2) * exp(- (x-x_center)^2 / 2 sigma^2)
	'''
	def __init__(self, x_min, x_max, sigma, x_center):
		'''
		Initialization function

		Arguments:
		- x_min: minimum value for the distribution
		- x_max: maximum value for the distribution
		- sigma: sigma of the Gaussian term
		- x_center: center of the Gaussian term
		'''

		self.sigma = sigma
		self.f = lambda x : (x /(sigma)**2)* np.exp(- ((x - x_center)/sigma)**2/2.)
		AnalyticDistribution.__init__(self, x_min, x_max, self.f)

class FunctionalUniform(Uniform):
	'''
	Applies a given function to a sample of the Uniform distribution
	'''
	def __init__(self, x_min, x_max, function):
		'''
		Initialization function

		Arguments:
		- x_min: minimum value for the distribution
		- x_max: maximum value for the distribution
		- function: function to be applied to the uniform samples
		'''

		self.f = function 
		self.uniform = Uniform(x_min, x_max)
		Uniform.__init__(self, x_min, x_max)

	def sample(self, n):
		'''
		Samples from this distribution

		Arguments:
		- n: number of samples
		'''

		s = self.uniform.sample(n)
		return self.f(s)
