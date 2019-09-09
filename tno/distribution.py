'''
Sampler classes for many different distributions. These are all based on the `BaseDistribution` class, 
and by subclassing it any distribution can be implemented. The JointDistribution allows one to construct a distribution of 
the form 
	alpha p_1(x) + (1-alpha) p_2(x),
where 0 <= alpha <= 1 and p_1, p_2 are two distributions

'''
import numpy as np
import inv_sample as ins

class BaseDistribution:
	'''
	Base class that all others inherit from. It corresponds to a delta function at zero. 
	'''
	def _init__(self):
		pass
	def __add__(self, other):
		return JointDistribution(self, other, 0.5)
	def __len__(self):
		return 0

	def sample(self, n):
		return np.zeros(n)


class AnalyticDistribution(BaseDistribution):
	'''	
	Given an analytic probability density function, samples from it
	'''
	def __init__(self, x_min, x_max, function, n_samp = 1000, n_bins = 100):
		self.f = function
		self.x_min = x_min
		self.x_max = x_max
		self.n_samp = n_samp
		self.n_bins = n_bins
		self._constructSampler() 

	def _constructSampler(self):
		x = np.linspace(self.x_min, self.x_max, self.n_samp)
		y = self.f(x)
		self.invCDF = ins.inverse_cdf(x, y, self.n_bins)

	def sample(self, n):
		r = np.random.rand(n)
		return self.invCDF(r)

class DeltaFunction(BaseDistribution):
	'''
	Delta function centered at loc
	'''
	def __init__(self, loc):
		self.loc = loc
		pass
	def sample(self, n):
		return self.loc * np.ones(n)

class PowerLaw(BaseDistribution):
	'''
	Power law of the form x^slope, defined between x_min and x_max
	'''
	def __init__(self, slope, x_min, x_max):
		self.slope = slope
		self.x_min = x_min
		self.x_max = x_max
		self.scale = x_max - x_min
		self.norm = self._normalization()

	def sample(self, n):
		samp = np.random.power(self.slope + 1, size=n)
		return self.x_min + self.scale * samp
	
	def _normalization(self):
		return (self.x_max**(self.slope + 1) - self.x_min**(self.slope + 1))/(self.slope + 1)

class JointDistribution(BaseDistribution):
	'''
	This returns alpha dist_1 + (1-alpha) dist_2
	'''
	def __init__(self, distribution_1, distribution_2, alpha):
		self.dist_1 = distribution_1
		self.dist_2 = distribution_2
		if alpha > 1 or alpha < 0:
			raise ValueError("The value of alpha must be between 0 and 1")
		self.alpha = alpha

	def sample(self, n):
		n_frac = int(self.alpha*n)
		samp1 = self.dist_1.sample(n_frac)
		samp2 = self.dist_2.sample(n - n_frac)

		samp = np.append(samp1, samp2)
		perm = np.random.permutation(n)
		return samp[perm]

class Uniform(PowerLaw):
	def __init__(self, x_min, x_max):
		PowerLaw.__init__(self, 0, x_min, x_max)

class Logarithm(Uniform):
	def __init__(self, x_min, x_max):
		self._sampler = Uniform(np.log(x_min), np.log(x_max))
		self.x_min = x_min
		self.x_max = x_max

	def sample(self, n):
		return np.exp(self._sampler.sample(n))

class DoublePowerLaw(AnalyticDistribution):
	def __init__(self, slope_1, slope_2, x_min, x_max, x_break, x_norm = None):
		self.x_break = x_break
		self.slope_1 = slope_1
		self.slope_2 = slope_2
		if x_norm == None:
			x_norm = (x_min + x_max)/2
		self.x_norm = x_norm

		self.f = lambda x : np.piecewise(x, [ x <= self.x_break, x > self.x_break], [lambda x: np.power(x/x_norm, self.slope_1), lambda x: np.power(x/x_norm, self.slope_2) * np.power(self.x_break/x_norm, self.slope_1 - self.slope_2)])
	
		AnalyticDistribution.__init__(self, x_min, x_max, self.f)

class BrownDistribution(AnalyticDistribution):
	def __init__(self, x_min, x_max, sigma):
		self.sigma = sigma
		self.f = lambda x : np.sin(x * np.pi/180) * np.exp(- (x/sigma)**2/2.)
		AnalyticDistribution.__init__(self, x_min, x_max, self.f)

