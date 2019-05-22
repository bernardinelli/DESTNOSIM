'''
Series of spatial distributions that will be added in the catalog
'''
import numpy as np

class PowerLaw:
	def __init__(self, slope, r_min, r_max):
		self.slope = slope
		self.r_min = r_min
		self.r_max = r_max
		self.scale = r_max - r_min

	def sample(self, n):
		samp = np.random.power(self.slope + 1, size=n)
		return self.r_min + self.scale * samp

class Uniform(PowerLaw):
	def __init__(self, r_min, r_max):
		PowerLaw.__init__(self, 0, r_min, r_max)

class LogDist(Uniform):
	def __init__(self, r_min, r_max):
		self.sampler = Uniform(np.log(r_min), np.log(r_max))
		self.r_min = r_min
		self.r_max = r_max

	def sample(self, n):
		return np.exp(self.sampler.sample(n))
