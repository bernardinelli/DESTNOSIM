'''
Sampler classes for many different distributions. These are all based on the `BaseDistribution` class, 
and by subclassing it any distribution can be implemented. The JointDistribution allows one to construct a distribution of 
the form 
	alpha p_1(x) + (1-alpha) p_2(x),
where 0 <= alpha <= 1 and p_1, p_2 are two distributions

'''
import numpy as np

class BaseDistribution:
	'''
	Base class that all others inherit from. It corresponds to a delta function at zero. 
	'''
	def _init__(self):
		pass
	def __add__(self, other):
		return JointDistribution(self, other, 0.5)
	def sample(self, n):
		return np.zeros(n)

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

class DoublePowerLaw(JointDistribution):
	def __init__(self, slope_1, slope_2, x_break, x_min, x_max):
		self.slope_1 = slope_1
		self.slope_2 = slope_2
		self.x_break = x_break
		self.x_min = x_min
		self.x_max = x_max		
		#self.norm = self._normalize()
		#self.frac = -(self.x_min**(1 + self.slope_1) - self.x_break**(1+self.slope_1))/(1 + self.slope_1)
		self.frac = (slope_2-1)/(slope_1-1) * x_break**(slope_2 - slope_1)

		self.norm = 1
		JointDistribution.__init__(self, PowerLaw(slope_1, x_min, x_break), PowerLaw(slope_2, x_break, x_max), self.frac/self.norm)

	def _normalize(self):
		a1 = self.slope_1
		a2 = self.slope_2 
		r1 = self.x_min 
		r2 = self.x_max
		rb = self.x_break

		t1 = rb**(1+a1)/(1+a1)  - (r1**(1+a1))/(1+a1) 
		t2 = (r2**(1+a2) - rb**(1+a2))/(1+a2)

		return t1 + t2

	'''def _normalize(self):
		a1 = self.slope_1
		a2 = self.slope_2
		r1 = self.x_min
		r2 = self.x_max
		rb = self.x_break
		num = - (1. + a2) * (r1**(1. + a1)) + (rb**a1) * ( (1. + a1) * r2 * ((r2/rb)**a2) + (a2 - a1)*rb)
		den = (1. + a1) * (1.+a2)
		return num/den'''


'''
Still doesn't work
class PowerLaw10:
	def __init__(self, slope, x_min, x_max, r_norm):
		self.slope = np.slope
		self.x_min = 10**(x_min - r_norm)
		self.x_max = 10**(x_max - r_norm)
		self.r_norm = r_norm
		self.scale = self.x_max/self.x_min

	def sample(self, n):
		samp = np.random.power(self.slope + 1, size=n)
		return np.log10(self.x_min + self.scale * samp) + self.r_norm
'''
