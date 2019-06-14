'''
Sampler classes for many different distributions. These are all based on the `BaseDistribution` class, 
and by subclassing it any distribution can be implemented. The JointDistribution allows one to construct a distribution of 
the form 
	alpha p_1(x) + (1-alpha) p_2(x),
where 0 <= alpha <= 1 and p_1, p_2 are two distributions

'''
import numpy as np
from scipy.stats import rv_continuous
import inv_sample as ins
import matplotlib.pyplot as plt

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
	def _init__(self, loc):
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


#Still doesn't work
class PowerLaw10(BaseDistribution):
	def __init__(self, slope, x_min, x_max, r_norm):
		self.slope = slope
		self.x_min = 10**(x_min - r_norm)
		self.x_max = 10**(x_max - r_norm)
		self.r_norm = r_norm
		self.scale = self.x_max/self.x_min

	def sample(self, n):
		samp = np.random.power(self.slope, size=n)
		return np.log10(self.x_min + self.scale * samp) + self.r_norm
   
class BrokenPowerLaw(rv_continuous):

    def __init__(self, slope_list, break_list, name='BrokenPowerLaw'):
        #Lower bound
        a=np.min(break_list)
        #Upper bound
        b=np.max(break_list)
        super().__init__(a=a,b=b, name=name)
        number_slopes=len(slope_list)
        # Calculate the proper normalization of the PDF semi-analytically
        
        pdf_norms=[1.0]
        for slope_num in range(1,number_slopes):
            a=np.power(break_list[slope_num], slope_list[slope_num-1] - slope_list[slope_num])
            pdf_norms.append(a)
        pdf_norms = np.cumprod(pdf_norms)
        
        #CDF_OFFSET
        cdf_offsets=[]
        for counter in range (0, number_slopes):
            slope = slope_list[counter]+1
            norm = pdf_norms[counter]
            cdf_offsets.append((norm/slope)*(np.power(break_list[counter+1], slope)-np.power(break_list[counter], slope)))
        cdf_offsets = np.array(cdf_offsets)

        offset_sum = cdf_offsets.sum()
        cdf_offsets = np.cumsum(cdf_offsets)
        pdf_norms = pdf_norms/offset_sum
        cdf_offsets = cdf_offsets/offset_sum
        
        self.breaks = break_list
        self.slopes = slope_list
        self.pdf_norms = pdf_norms
        self.cdf_offsets = cdf_offsets
        self.num_segments = len(slope_list)
        return
  

    #Overwriting
    def _cdf(self, x):
        original_input = np.atleast_1d(x)
        empty_input = np.zeros_like(original_input)
        offset = 0.0
        for index in range(self.num_segments):
            if index>0:
                offset = self.cdf_offsets[index-1]
            idx = (self.breaks[index] < original_input) & (original_input <= self.breaks[index+1])
            slope = self.slopes[index]
            norm = self.pdf_norms[index]
            empty_input[idx] = (norm/(slope + 1)) * (np.power(original_input[idx], slope + 1) - np.power(self.breaks[index], slope + 1)) + offset
        return empty_input

    
BrokenPowerLawTest = BrokenPowerLaw([1.5, -1.5, -23.5], [10, 100.5, 1000.0, 10000.5])
rvs=BrokenPowerLawTest.rvs(size=1000)
count, bins, ignored = plt.hist(rvs, bins=100)
plt.show()
plt.clf()
        
        
def plot(Distribution, sample_size, bin_size):
    sample=Distribution.sample(sample_size)
    count, bins, ignored = plt.hist(sample, bins=bin_size)
    plt.show()
    plt.clf()

power_distr = PowerLaw10(1, 0, 100, 100)
log_distr = Logarithm(1,100)
plot(power_distr,1000,100)
plot(log_distr,1000,100)


'''
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

	def _normalize(self):
		a1 = self.slope_1
		a2 = self.slope_2
		r1 = self.x_min
		r2 = self.x_max
		rb = self.x_break
		num = - (1. + a2) * (r1**(1. + a1)) + (rb**a1) * ( (1. + a1) * r2 * ((r2/rb)**a2) + (a2 - a1)*rb)
		den = (1. + a1) * (1.+a2)
		return num/den
'''
