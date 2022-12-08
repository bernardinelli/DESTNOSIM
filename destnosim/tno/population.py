from __future__ import print_function
import numpy as np
import os
import subprocess
import astropy.table as tb 
import pickle 
import copy
from .transformation import * 
from .distribution import * 
from .popstat import *

orbdata = os.getenv('DESDATA')
if orbdata is None:
	orbdata = ''


def fibonacci_sphere(n_grid, angles = True):
	'''
	Generates a 3d Fibonacci sphere shell (see, eg, Gonzalez, A. 2010, Mathematical Geosciences, 42, 49, doi: 10.1007/s11004-009-9257-x) 
	for a certain n_grid. Generates 2n + 1 points

	Argument:
	- n_grid: grid size (integer), will generate 2n+1 points
	- angles: returns the generated angles as well as the 3D positions
	'''
	sin_phi = 2*np.arange(-n_grid, n_grid, 1, dtype=float)/(2*n_grid + 1.)
	theta = 2*np.pi * np.arange(-n_grid, n_grid, 1, dtype=float)/(0.5*(1. + np.sqrt(5)))**2
	x_shell = np.sqrt(1-sin_phi*sin_phi) * np.cos(theta)
	y_shell = np.sqrt(1-sin_phi*sin_phi) * np.sin(theta)
	z_shell = sin_phi
	if angles:
		return x_shell, y_shell, z_shell, sin_phi, theta
	else:
		return x_shell, y_shell, z_shell


class Population:
	'''
	Base class for all populations
	'''
	def __init__(self, n_objects, elementType, epoch):
		'''
			Initialization function

			Arguments:
			- n_objects: population size, will generate a (n x 6) array for the orbital elements or state vectors
			- elementType: type of element, use either 'keplerian' or 'cartesian' for Keplerian elements or state vectors, respectively
			- epoch: epoch of the elements, in years after J2000.0
		'''
		self.n_objects = n_objects
		self.elementType = elementType
		self.epoch = epoch
		self.elements = np.zeros((n_objects, 6)) 
		
		self.hasRADec = False

		if self.elementType == 'keplerian':
			self.state = 'F'
		else:
			self.state = 'T'
		self.heliocentric = False
	
	def generateMagnitudes(self, distribution, mag_type, band, colors = None, observer_pos = [1,0,0], helio = False, ecliptic = False, bands = ['g', 'r', 'i', 'z', 'Y']):
		'''
		Depends on magnitude distributions from `distribution.py`. 

		Arguments:
		- distribution: Magnitude distribution from which the magnitudes will be sampled. If a single distribution is provided, all magnitudes will be sampled from it 
		Can be a list/array of distributions, in which case the magnitude of object i will come from distribution[i]
		- mag_type: Defines if magnitudes are 'absolute' or 'apparent'. If absolute, requires observer_pos (3d array), helio and ecliptic (booleans) 
		- band: reference band for the magnitude (string)
		- colors: Dictionary of color transformations of the form band - colors[i]. colors can be None, in which case only one band is generated
		- observer_pos: 3d array or list with the location of the observer for conversion between absolute and apparent magnitudes
		- helio: boolean, uses heliocentric elements for the distance calculations (only required for orbital elements)
		- ecliptic: boolean, uses ecliptic-aligned elements for the distance calculations (only required for orbital elements)
		- bands: list of which bands the magnitudes are generated in
		'''
		if mag_type == 'absolute':
			name = 'H' 
		elif mag_type == 'apparent':
			name = 'm'
		else:
			raise ValueError("Magnitude type (mag_type) must be either absolute or apparent")
		
		if len(distribution) > 0:
			mag_band = np.zeros(self.n_objects)
			for i, dist in enumerate(distribution):
				mag_band[i] = dist.sample(1)
		else:
			mag_band = distribution.sample(self.n_objects)

		mag_table = tb.Table()

		mag_table[name + '_' + band] = 	mag_band

		for i in colors:
			mag_table[name + '_' + i] = mag_band - colors[i] 

		mag_table['ORBITID'] = range(self.n_objects)

		if mag_type == 'absolute':
			if not hasattr(self, 'r'):
				self.distanceToCenter(helio, ecliptic)
				self.delta = self.distanceToPoint(observer_pos, helio, ecliptic)

			mag_table['m_' + band] = mag_band + 5 * np.log10(self.r * self.delta)
			mag_table['H_' + band] = mag_band

			for i in colors:
				mag_table['m_' + i] = mag_band - colors[i]  + 5 * np.log10(self.r * self.delta)

		self.mag = mag_table

		band_stack = [] 

		for i in bands:
			band = mag_table['ORBITID', 'm_' + i]
			band.rename_column('m_' + i, 'MAG')
			band['BAND'] = i 
			band_stack.append(band)
		
		self.mag_obs = tb.vstack(band_stack)

	def generateLightCurve(self, lightcurve):
		'''
		Applies some sort of light curve to each detection. `lightcurve` can either be a single function in which you can call the lightcurve amplitude at a given observation time, 
		or an array containing the lightcurve for each object and indexed by ORBITID.

		Arguments:
		- lightcurve: list or individual LightCurve object
		'''

		if len(lightcurve) > 0:
			for i in range(self.n_objects):
				where = self.obs['ORBITID'] == i
				self.obs['MAG'][where] = self.obs['MAG'][where] + lightcurve[i](self.obs[where]['MJD'])
		else:
			self.obs['MAG'] = lightcurve(self.obs['MJD']) + self.obs['MAG']

	def createCopies(self, n_clones):
		'''
		Creates n_clones copies of each object, allowing for some posterior randomization
		'''

		self.elements = np.vstack([self.elements for i in range(n_clones)])
		self.n_objects *= n_clones

	def sampleElements(self, covariances, n_samples):
		'''
		Draws from the array of covariance matrices provided		

		Arguments:
		- covariances: 6D covariances of the orbital elements/state vectors
		- n_samples: number of samples to be drawn
		'''
		n_orig = copy.deepcopy(self.n_objects)
		self.covariance = covariances
		if n_samples > 1:
			self.createCopies(n_samples)

		for i in range(n_orig):
			samples = np.random.multivariate_normal(self.elements[i,:], covariances[i], n_samples)
			for j in range(n_samples):
				self.elements[i + j*n_orig,:] = samples[j]

	def distanceToCenter(self, helio = False, ecliptic = False):
		'''
		Computes distance to the center of mass of the system. helio and ecliptic are required if a transformation from orbital elements
		to cartesian vectors is needed

		Arguments:
		- helio: boolean, uses heliocentric elements for the distance calculations (only required for orbital elements)
		- ecliptic: boolean, uses ecliptic-aligned elements for the distance calculations (only required for orbital elements)
		'''

		r = dist_to_point(self.elements, self.epoch, self.elementType, np.array([0,0,0]), helio, ecliptic)
		self.r = r 

	def distanceToPoint(self, point, helio = False, ecliptic = False):
		'''
		Computes distance to the center of mass of the system. helio and ecliptic are required if a transformation from orbital elements
		to cartesian vectors is needed

		Arguments:
		- point: 3d position of the point to which the distances are being computed to
		- helio: boolean, uses heliocentric elements for the distance calculations (only required for orbital elements)
		- ecliptic: boolean, uses ecliptic-aligned elements for the distance calculations (only required for orbital elements)
		'''

		r = dist_to_point(self.elements, self.epoch, self.elementType, point, helio, ecliptic)
		return r 

	def computeStatistics(self, thresh = 90., transient_efficiency = 0.95529, drop_zero = True, linking_eff = 1):
		'''
		Computes ARC, ARCCUT and NUNIQUE for each member of the population and stores inside population.statistics. Requires a preliminary survey.observePopulation call

		Arguments:
		- thresh: threshold for triplet generation (in days). Use 60 for d < 50 au
		- transient_efficiency: efficiency of the transient generation process. Keeps transient_efficiency*100% of the transients in the catalog
		'''
		try:
			self.detections
		except:
			raise AttributeError("Population has no detections attribute. Perhaps you need to call survey.observePopulation first?")


		#consider only detections inside a CCD
		if drop_zero:
			ccdzero = np.where(self.detections['CCDNUM'] == 0)
			if len(ccdzero) > 0:
				self.detections.remove_rows(np.where(self.detections['CCDNUM'] == 0))

		#transient efficiency
		if 'RANDOM' not in self.detections.keys():
			self.detections['RANDOM'] = np.random.rand(len(self.detections))
			self.detections.remove_rows(np.where(self.detections['RANDOM'] > transient_efficiency))

		#index the table to make things easier
		self.detections.add_index('ORBITID')
		stat = tb.Table(names=['ORBITID', 'ARC', 'ARCCUT', 'NUNIQUE', 'NDETECT', 'TRIPLET', 'PAIR_1', 'PAIR_2'], dtype=['i8', 'f8', 'f8', 'i8', 'i8', 'b1', 'f8', 'f8'])
		
		ids, counts = np.unique(self.detections['ORBITID'], return_counts = True)

		for i in ids[counts > 1]:
			obj = self.detections.loc[i]
			times = np.array(obj['MJD'], dtype='float64') 
			times.sort()
			arc = np.max(times) - np.min(times)
			arccut = compute_arccut(times)
			nunique = compute_nunique(times)
			pt = find_triplet_time(times)			
			if pt[0] < thresh and pt[1] < thresh:
				has_trip = True 
			else:
				has_trip = False
			stat.add_row([i, arc, arccut, nunique, len(times), has_trip, pt[0], pt[1]])

		stack = [stat]
		ones = tb.Table()
		ones['ORBITID'] = ids[counts == 1]
		if len(ones) > 0:
			ones['ARC'] = 0.
			ones['ARCCUT'] = 0.
			ones['NUNIQUE'] = 1
			ones['NDETECT'] = 1
			ones['TRIPLET'] = False
			ones['PAIR_1'] = 0
			ones['PAIR_2'] = 0
			stack.append(ones)


		orbid = np.arange(len(self))
		zeros = tb.Table()
		zeros['ORBITID'] = orbid[np.isin(orbid, ids, invert = True)]
		if len(zeros) > 0:
			zeros['ARC'] = 0.
			zeros['ARCCUT'] = 0.
			zeros['NUNIQUE'] = 0
			zeros['NDETECT'] = 0
			zeros['TRIPLET'] = False
			zeros['PAIR_1'] = 0
			zeros['PAIR_2'] = 0
			stack.append(zeros)


		stat = tb.vstack(stack)
		stat.sort('ORBITID')

		stat['RANDOM'] = np.random.random(len(stat)) < linking_eff

		self.statistics = stat


	def randomizeElement(self, element, distribution):
		'''
		Randomizes element (must be an integer between 0 and 5) according to the provided distribution (from `distribution.py`)

		Arguments:
		- element: index of the element to be sampled
		- distribution: distribution for this element
		'''
		self.elements[:,element] = distribution.sample(self.n_objects)


	def removeMembers(self, indices):
		'''
		Deletes members of the population

		Arguments:
		- indices: array of indices that will be deletec
		'''
		self.elements = self.elements[indices]
		self.n_objects = len(self.elements)

	# Standard class methods
	def __str__(self):
		'''	
		String representation
		'''
		return "Population with {} objects. Elements are of type {}".format(self.n_objects, self.elementType)

	def __add__(self, other):
		'''
		Adds two populations together
		'''
		if self.elementType != other.elementType:
			raise ValueError("The elements of each population must be of the same type")
		if self.epoch != other.epoch:
			raise ValueError("The epochs must be the same")
		newpop = Population(self.n_objects + other.n_objects, self.elementType, self.epoch)
		newpop.elements[0:self.n_objects,:] = self.elements
		newpop.elements[self.n_objects:,:] = other.elements
		return newpop

	def __len__(self):
		return self.n_objects

	def write(self, filename):
		'''
		Saves the population to a pickle file with the specified name

		Arguments:
		- filename: filename for the saved file
		'''
		with open(filename, 'wb') as f:
			pickle.dump(self, f, protocol = 2)
	@staticmethod
	def read(filename):
		'''
		Static method, reads a file with the given filename

		Arguments:
		- filename: pickle file that will be read
		'''
		with open(filename, 'rb') as f:
			return pickle.load(f)

	def __getitem__(self, index):
		return self.elements[index,:]

	def checkInFootprint(self, footprint = orbdata + '/round17-poly.txt'):
		'''
		Checks which objects are inside the footprint
		
		Arguments:
		- footprint: file containing details of the footprint

		'''
		import matplotlib.path as mpath
		p = np.loadtxt(footprint)
		path = mpath.Path(p)

		if not self.hasRADec:
			self.getRADec()

		inside = path.contains_points(np.array([self.ra, self.dec]).T)

		self.elements = self.elements[inside]
		self.ra = self.ra[inside]
		self.dec = self.dec[inside]
		
		self.n_objects = len(self.elements)
		#self.n_grid = np.ceil(self.n_objects/2)

	def getRADec(self, heliocentric = False, ecliptic = False, observer_pos = np.array([0,0,0])):
		if self.elementType == 'keplerian':
			xv = keplerian_to_cartesian(self.elements, self.epoch, heliocentric, ecliptic)
		else:
			xv = self.elements
		xv_eq = rotate_to_ecliptic(xv)

		x_vec = xv_eq[:,:3] - observer_pos

		self.ra = np.arctan2(x_vec[:,1], x_vec[:,0]) * 180./np.pi

		self.dec = np.arcsin(x_vec[:,2]/np.sqrt(x_vec[:,0]**2 + x_vec[:,1]**2 + x_vec[:,2]**2)) * 180/np.pi

		self.hasRADec = True

	def generateSpaceRocks(self):
		import spacerocks
		import astropy.time as tm 
		import astropy.units as u 

		epoch = tm.Time(2000 + self.epoch, scale='tdb', format='jyear')
		epoch.format = 'mjd'
		units = spacerocks.Units()
		units.timescale = 'tdb'
		units.timeformat = 'mjd'
		units.speed = u.au/u.yr
		origin = 'sun' if self.heliocentric else 'ssb'

		names = [f'{i}' for i in range(len(self))]

		if self.elementType == 'keplerian':
			mu = SunGM if self.heliocentric else SolarSystemGM

			M = np.rad2deg((self.epoch - self.elements[:,5])*(np.sqrt(mu)/np.power(self.elements[:,0],3./2)))

			rock = spacerocks.SpaceRock(a = self.elements[:,0], e = self.elements[:,1], inc = self.elements[:,2],
										node = self.elements[:,3], arg = self.elements[:,4], M = M,
										epoch = len(self) * [epoch.mjd], units = units, name=names, origin=origin, frame='eclipJ2000')

		else:
			frame = 'eclipJ2000' if self.ecliptic else 'J2000'
			rock = spacerocks.SpaceRock(x = self.elements[:,0], y = self.elements[:,1], z = self.elements[:,2],
										vx = self.elements[:,3], vy = self.elements[:,4], vz = self.elements[:,5],
										epoch = len(self) * [epoch.mjd], units = units, frame = frame, name=names, origin=origin)

		return rock

class ElementPopulation(Population):
	'''
	Series of orbital elements that are submitted to `DESTracks`. The code requires (a,e,i, lan, aop, top). The user should submit at least 6 elements, and the code will convert to the right six if needed
	Units should be AU, degrees and years after J2000
	Input should be a dictionary of arrays with the proper orbital elements
	'''
	def __init__(self, elements, epoch, heliocentric = False):
		'''
		Initialization function:
		- elements: dictionary of orbital elements. Example: {'a': [30], 'e' : [0.1], 'i': [0], 'aop': [0], 'lan' : [0], 'top': [0]} 
		- epoch: epoch of the elements (years)
		- heliocentric: defines if heliocentric elements are being used


		Many different combinations of orbital elements can be provided, here is a list of all accepted elements:
		- Semi-major axis: provide as 'a' (au)
		- Eccentricity: provide as 'e'
		- Perihelion: provide as 'q' (au)
		- Inclination: provide as 'i' (degrees)
		- Argument of perihelion: provide as 'omega' or 'aop' (degrees)
		- Longitude of ascending node: provide as 'Omega' or 'lan' (degrees)
		- Longitude of perihelion: provide as 'varpi' or 'lop' (degrees)
		- Time of perihelion passage: provide as 'top' or 'T_p' (years)
		- Mean anomaly: provide as 'man' or 'M' (degrees)
		'''
		self.input = elements
		self._keys = elements.keys()
		n = len(elements[list(elements.keys())[0]])
		Population.__init__(self, n, 'keplerian', epoch)
		self._organizeElements()
		self.heliocentric = heliocentric

	def _organizeElements(self):
		mu = SunGM if self.heliocentric else SolarSystemGM
		if 'a' in self._keys:
			self.elements[:,0] = self.input['a']
		elif 'q' in self._keys and 'e' in self._keys:
			self.elements[:,0] = self.input['q']/(1 - self.input['e'])
		else:
			raise ValueError('Please provide either semi-major axis (a) or perihelion (q)/eccentricity (e) as input')
		
		if 'e' in self._keys:
			self.elements[:,1] = self.input['e']
		elif 'q' in self._keys and 'a' in self._keys:
			self.elements[:,1] = 1 - self.input['q']/self.input['a'] 
		else:
			raise ValueError("Please provide either eccentricity (e) or semi-major axis (a)/perihelion (q) as input")

		if 'i' in self._keys:
			self.elements[:,2] = self.input['i']
		else:
			raise ValueError("Please provide inclination (i) as input")

		if 'lan' in self._keys:
			self.elements[:,3] = self.input['lan']
		elif 'Omega' in self._keys:
			self.elements[:,3] = self.input['Omega']
		elif 'lop' in self._keys and 'aop' in self._keys:
			self.elements[:,3] = self.input['lop'] - self.input['aop']
		elif 'varpi' in self._keys and 'omega' in self._keys:
			self.elements[:,3] = self.input['varpi'] - self.input['omega']
		else:
			raise ValueError("Please provide either longitude of ascending node (lan) or longitude of perihelion (lop)/argument of perihelion (aop) as input")

		if 'aop' in self._keys:
			self.elements[:,4] = self.input['aop']
		elif 'omega' in self._keys:
			self.elements[:,4] = self.input['omega']
		elif 'lop' in self._keys and 'lan' in self._keys:
			self.elements[:,4] = self.input['lop'] - self.input['lan']
		elif 'varpi' in self._keys and 'Omega' in self._keys:
			self.elements[:,3] = self.input['varpi'] - self.input['Omega']
		else:
			raise ValueError("Please provide either argument of perihelion (aop) or longitude of perihelion (lop)/longitude of ascending node (lan) as input")

		if 'top' in self._keys:
			self.elements[:,5] = self.input['top']
		elif 'T_p' in self._keys:
			self.elements[:,5] = self.input['T_p']
		elif 'man' in self._keys:
			self.elements[:,5] = self.epoch - self.input['man'] * np.power(self.elements[:,0], 3./2) * (np.pi/180) / np.sqrt(mu)
		elif 'M' in self._keys:
			self.elements[:,5] = self.epoch - self.input['M'] * np.power(self.elements[:,0], 3./2) * (np.pi/180) / np.sqrt(mu)
		else:
			raise ValueError("Please provide either time of perihelion passage (top) or mean anomaly (man)/semi-major axis (a) as input")

	def randomizeAngle(self, element, min_angle = 0, max_angle = 360, uniform = True):
		'''
		Randomizes argument of perihelion 'aop' or longitude of ascending node 'lan' by either randomizing the angles or sampling all possible values
		between min_angle (= 0 deg by default) and max_angle (= 360 deg by default)

		Arguments:
		- element: 'lan' or 'aop' (or 3 or 4, for lan and aop, respectively)
		- min_angle: minimum angle (degrees)
		- max_angle: maximum angle (degrees)
		- uniform: generates in a uniform manner that covers the entire parameter space, instead of sampling from a uniform distribution
		'''
		eldic = {'lan' : 3, 'aop' : 4}
		if type(element) is str:
			element = eldic[element]
		if uniform:
			self.elements[:,element] = np.linspace(min_angle, max_angle, self.n_objects) 
		else:
			self.elements[:,element] = np.random.rand(self.n_objects)*(max_angle - min_angle) + min_angle

	def randomizeToP(self, min_angle = -180., max_angle = 180., uniform = True, heliocentric = False):
		'''
		Randomizes the Time of Perihelion passage by randomizing the true anomaly between min_angle ( =-180 deg, by default) and max_angle (180 deg)
		in either a uniform fashion (i.e. covers the entire parameter space) or drawing ran
		Arguments:
		- element: 'lan' or 'aop' (or 3 or 4, for lan and aop, respectively)
		- min_angle: minimum angle (degrees)
		- max_angle: maximum angle (degrees)
		- uniform: generates in a uniform manner that covers the entire parameter space, instead of sampling from a uniform distribution
		- heliocentric: boolean, defines the GM for the conversion between M and T_p
		'''
		mu = SunGM if heliocentric else SolarSystemGM

		if uniform:
			self.elements[:,5] = self.epoch - (np.linspace(min_angle, max_angle, self.n_objects)*np.pi/180.) * np.power(self.elements[:,0], 3./2)/np.sqrt(mu)
		else:
			self.elements[:,5] = self.epoch - (np.random.rand(self.n_objects) * (max_angle - min_angle) + min_angle) * np.pi/180. * np.power(self.elements[:,0], 3./2)/np.sqrt(mu)

	def transformElements(self, heliocentric = False, ecliptic = False):
		'''
		Transforms the population to a CartesianPopulation. Depends on being heliocentric or barycentric and ecliptic aligned or equatoriallly aligned

		Arguments:
		- heliocentric: boolean, defines if heliocentric aligned 
		- ecliptic: boolean, defines if ecliptic aligned
		'''
		xv = keplerian_to_cartesian(self.elements, self.epoch, heliocentric, ecliptic)
		xv_dict = {'x' : xv[:,0], 'y' : xv[:,1], 'z' : xv[:,2], 'vx' : xv[:,3], 'vy' : xv[:,4], 'vz' : xv[:,5]}
		return CartesianPopulation(xv_dict, self.epoch)

	def toBarycentric(self, barycenter_coordinates):
		'''
		Transforms a set of heliocentric orbital elements to barycentric orbital elements
		Arguments:
		- barycenter_coordinates: 6D barycenter state vector (AU, AU/yr)
		'''
		if not self.heliocentric:
			raise ValueError("Population is already in barycentric elements")
		else:
			new_aei = helio_to_bary(self.elements, "keplerian", self.epoch, barycenter_coordinates)
			aei_dict = {'a' : new_aei[:,0], 'e' : new_aei[:,1], 'i' : new_aei[:,2], 'lan' : new_aei[:,3], 'aop' : new_aei[:,4], 'T_p' : new_aei[:,5]}
			return ElementPopulation(aei_dict, self.epoch, False)




class CartesianPopulation(Population):
	'''
	Base class for a population using state vectors to be submitted to `DESTracks`. Positions are always in AU, velocities in AU/yr
	Input: a dictionary with (x,y,z,vx,vy,vz) entries 
	'''
	def __init__(self, elements, epoch, ecliptic = False, heliocentric = False):
		'''
		Initialization function

		Arguments:
		- dictionary of elements containing 'x', 'y', 'z', 'vx', 'vy', 'vz'
		- epoch: epoch of the elements (years)
		- heliocentric: boolean, defines if heliocentric elements are being used
		- ecliptic: boolean, defines if ecliptic-aligned or equatorially aligned
		'''
		self.input = elements
		n = len(elements[list(elements.keys())[0]])
		Population.__init__(self, n, 'cartesian', epoch)
		self.elements[:,0] = self.input['x']
		self.elements[:,1] = self.input['y']
		self.elements[:,2] = self.input['z']
		self.elements[:,3] = self.input['vx']
		self.elements[:,4] = self.input['vy']
		self.elements[:,5] = self.input['vz']

		self.ecliptic = ecliptic
		self.heliocentric = heliocentric

	def transformElements(self, heliocentric = False, ecliptic = False):
		'''
		Transforms the population to a ElementPopulation. Depends on being heliocentric or barycentric and ecliptic aligned or equatoriallly aligned
		
		Arguments:
		- heliocentric: boolean, defines if heliocentric elements are being used
		- ecliptic: boolean, defines if ecliptic-aligned or equatorially aligned
		'''
		aei = cartesian_to_keplerian(self.elements, self.epoch, heliocentric, ecliptic)
		aei_dict = {'a' : aei[:,0], 'e' : aei[:,1], 'i' : aei[:,2], 'Omega' : aei[:,3], 'omega' : aei[:,4], 'T_p' : aei[:,5]}
		return ElementPopulation(aei_dict, self.epoch)

	def rotateElements(self):
		'''
		Rotates between equatorial and ecliptic state vectors
		'''
		self.elements = rotate_to_ecliptic(self.elements, not self.ecliptic)
		self.ecliptic = not self.ecliptic
	

class IsotropicPopulation(CartesianPopulation):
	'''
	Generates two Fibonacci spheres (one for velocities and one for positions) objects are sampled across all possible parameters
	See detailed description in Bernardinelli et al 2020a for the detailed explanation
	'''
	def __init__(self, n_grid, epoch, drop_outside = True, ecliptic = False, footprint = orbdata + '/round17-poly.txt'):
		'''
		Initialization function

		Arguments:
		- n_grid: grid size for the Fibonacci sphere, creates 2n+1 points
		- epoch: epoch for the elements (years)
		- drop_outside: boolean, deletes objects outside the specified footprint (eg the DES footprint)
		- ecliptic: boolean, defines if ecliptic aligned
		- footprint: file containing details of the footprint
		'''
		### Warning: n_grid defines the _grid size_, which leads to 2n+1 objects
		size = len(np.arange(-n_grid, n_grid, 1, dtype=float))
		Population.__init__(self, size, 'cartesian', epoch)
		self.n_grid = n_grid
		self._generateShell()
		self.hasRADec = True
		if drop_outside:
			self.checkInFootprint(footprint)
			self.n_grid = int(self.n_objects/2)
		self._generateVelocityShell()
		
		self.ecliptic = ecliptic

		if ecliptic:
			print("DESTracks requires equatorial state vectors! Remember to rotate first.")

	def _generateShell(self):
		'''
		Places objects on a spherical shell of radius 1 AU
		'''
		x,y,z, sin_dec, ra = fibonacci_sphere(self.n_grid)
		self.elements[:,0] = x
		self.elements[:,1] = y
		self.elements[:,2] = z

		self.ra = np.mod(180*ra/np.pi, 360)
		self.ra[self.ra>180] -= 360 

		self.dec = 180*np.arcsin(sin_dec)/np.pi


	def generateDistances(self, distribution):
		'''
		Generates distances according to the provided distribution

		Arguments:
		- distribution: distribution object of distances to be used
		''' 
		r = distribution.sample(self.n_objects)

		self.elements[:,0] *= r
		self.elements[:,1] *= r
		self.elements[:,2] *= r
		self.r = r

	def _generateVelocityShell(self):
		vx, vy, vz = fibonacci_sphere(self.n_grid, False)
		perm = np.random.permutation(self.n_objects)
		self.elements[:,3] = vx[perm]
		self.elements[:,4] = vy[perm]
		self.elements[:,5] = vz[perm]

	def generateVelocities(self, distribution):
		'''
		Places the velocity from the 1 AU/yr shell into more realistic units scaled by the escape velocity and the provided distribution

		Arguments:
		- distribution: distribution object of distances to be used

		'''
		v_escape = np.sqrt(2*SolarSystemGM/self.r)
		v_scale = distribution.sample(self.n_objects)

		self.elements[:,3] *= v_escape * v_scale
		self.elements[:,4] *= v_escape * v_scale
		self.elements[:,5] *= v_escape * v_scale

		self.v = v_escape * v_scale

