'''
Series of spatial distributions that will be added in the catalog
'''
import numpy as np
from scipy.stats import rv_continuous

class BaseDistribution:
	def _init__(self):
		pass
	def __add__(self, other):
		return JointDistribution(self, other, 0.5)

class PowerLaw(BaseDistribution):
	def __init__(self, slope, r_min, r_max):
		self.slope = slope
		self.r_min = r_min
		self.r_max = r_max
		self.scale = r_max - r_min

	def sample(self, n):
		samp = np.random.power(self.slope + 1, size=n)
		return self.r_min + self.scale * samp
		
class JointDistribution(BaseDistribution):
	'''
	This returns alpha dist_1 + (1-alpha) dist_2
	'''
	def __init__(self, distribution_1, distribution_2, alpha):
		self.dist_1 = distribution_1
		self.dist_2 = distribution_2
		self.alpha = alpha

	def sample(self, n):
		n_frac = int(self.alpha*n)
		samp1 = self.dist_1.sample(n_frac)
		samp2 = self.dist_2.sample(n - n_frac)

		samp = np.append(samp1, samp2)
		perm = np.random.permutation(n)

class Uniform(PowerLaw):
	def __init__(self, r_min, r_max):
		PowerLaw.__init__(self, 0, r_min, r_max)

class Logarithm(Uniform):
	def __init__(self, r_min, r_max):
		self._sampler = Uniform(np.log(r_min), np.log(r_max))
		self.r_min = r_min
		self.r_max = r_max

	def sample(self, n):
		return np.exp(self._sampler.sample(n))


		return samp[perm]

class DoublePowerLaw(JointDistribution):
	def __init__(self, slope_1, slope_2, breakloc, r_min, r_max):
		self.slope_1 = slope_1
		self.slope_2 = slope_2
		self.breakloc = breakloc
		self.r_min = r_min
		self.r_max = r_max		
		self.norm = self._normalize()
		self.frac = -(self.r_min**(1 + self.slope_1) - self.breakloc**(1+self.slope_1))/(1 + self.slope_1)

		self.PowerLaw1 = PowerLaw(slope_1, r_min, breakloc)
		self.PowerLaw2 = PowerLaw(slope_2, breakloc, r_max)
		JointDistribution.__init__(self, self.PowerLaw1, self.PowerLaw2, self.frac/self.norm)

	def _normalize(self):
		a1 = self.slope_1
		a2 = self.slope_2
		r1 = self.r_min
		r2 = self.r_max
		rb = self.breakloc
		num = - (1 + a2) * (r1**(1 + a1)) + (rb**a1) * ( (1 + a1) * r2 * ((r2/rb)**a2) + (a2 - a1)*rb)
		den = (1 + a1) * (1+a2)
		return num/den

'''
Still doesn't work
class PowerLaw10:
	def __init__(self, slope, r_min, r_max, r_norm):
		self.slope = np.slope
		self.r_min = 10**(r_min - r_norm)
		self.r_max = 10**(r_max - r_norm)
		self.r_norm = r_norm
		self.scale = self.r_max/self.r_min

	def sample(self, n):
		samp = np.random.power(self.slope + 1, size=n)
		return np.log10(self.r_min + self.scale * samp) + self.r_norm
'''
