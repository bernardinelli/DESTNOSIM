from __future__ import print_function
import numpy as np
import os
import subprocess
import astropy.table as tb 

GM = 2*np.pi



def fibonacci_sphere(n_grid):
	sin_phi = 2*np.arange(-n_grid, n_grid, 1, dtype=float)/(2*n_grid + 1.)
	theta = 2*np.pi * np.arange(-n_grid, n_grid, 1, dtype=float)/(0.5*(1. + np.sqrt(5)))**2
	x_shell = np.sqrt(1-sin_phi*sin_phi) * np.cos(theta)
	y_shell = np.sqrt(1-sin_phi*sin_phi) * np.sin(theta)
	z_shell = sin_phi
	return x_shell, y_shell, z_shell

#def brown_distribution():

class Population:
	def __init__(self, n_objects, elementType, epoch):
		self.n_objects = n_objects
		self.elementType = elementType
		self.epoch = epoch
		self.elements = np.zeros((n_objects, 6)) 

		if self.elementType == 'elements':
			self.state = 'F'
		else:
			self.state = 'T'
	
	def generateMagnitudes(self, distribution):
		'''
		Depends on magnitude distributions from `magnitude.py`
		'''
		return None


class IsotropicPopulation(Population):
	def __init__(self, n_grid, epoch):
		### Warning: n_grid defines the _grid size_, which leads to 2n+1 objects
		Population.__init__(self, 2*n_grid, 'cartesian', epoch)
		self.n_grid = n_grid
		self._generateShell()
		self._generateVelocityShell()

	def _generateShell(self):
		'''
		Places objects on a spherical shell of radius 1 AU
		'''
		x,y,z = fibonacci_sphere(self.n_grid)
		self.elements[:,0] = x
		self.elements[:,1] = y
		self.elements[:,2] = z

	def generateDistances(self, distribution):
		'''
		Generates distances according to the provided distribution
		''' 
		r = distribution.sample(self.n_objects)

		self.elements[:,0] *= r
		self.elements[:,1] *= r
		self.elements[:,2] *= r
		self.r = r

	def _generateVelocityShell(self):
		vx, vy, vz = fibonacci_sphere(self.n_grid)
		perm = np.random.permutation(self.n_objects)

		self.elements[:,3] = vx[perm]
		self.elements[:,4] = vy[perm]
		self.elements[:,5] = vz[perm]

	def generateVelocities(self, distribution):
		v_circ = GM/np.sqrt(self.r)
		v_scale = distribution.sample(self.n_objects)

		self.elements[:,3] = v_circ * v_scale
		self.elements[:,4] = v_circ * v_scale
		self.elements[:,5] = v_circ * v_scale


class ElementPopulation(Population):
	'''
	Series of orbital elements that are submitted to `DESTracks`. We need (a,e,i, lan, aop, top). So the user should submit at least 6 elements, and this can convert to the right six if needed
	Units should be AU, degrees and years after J2000
	Input should be a dictionary of arrays with the proper orbital elements
	'''
	def __init__(self, elements, epoch):
		self.input = elements
		n = len(elements[list(elements.keys())[0]])
		Population.__init__(self, n, 'elements', epoch)
		self._organizeElements()

	def _organizeElements(self):
		if 'a' in self.input:
			self.elements[:,0] = self.input['a']
		elif 'q' in self.input and 'e' in self.input:
			self.elements[:,0] = self.input['q']/(1 - self.input['e'])
		else:
			raise ValueError('Please provide either semi-major axis (a) or perihelion (q)/eccentricity (e) as input')
		
		if 'e' in self.input:
			self.elements[:,1] = self.input['e']
		elif 'q' in self.input and 'a' in self.input:
			self.elements[:,1] = 1 - self.input['q']/self.input['a'] 
		else:
			raise ValueError("Please provide either eccentricity (e) or semi-major axis (a)/perihelion (q) as input")

		if 'i' in self.input:
			self.elements[:,2] = self.input['i']
		else:
			raise ValueError("Please provide inclination (i) as input")

		if 'lan' in self.input:
			self.elements[:,3] = self.input['lan']
		elif 'lop' in self.input and 'aop' in self.input:
			self.elements[:,3] = self.input['lop'] - self.input['aop']
		else:
			raise ValueError("Please provide either longitude of ascending node (lan) or longitude of perihelion (lop)/argument of perihelion (aop) as input")

		if 'aop' in self.input:
			self.elements[:,4] = self.input['aop']
		elif 'lop' in self.input and 'lan' in self.input:
			self.elements[:,4] = self.input['lop'] - self.input['lan']
		else:
			raise ValueError("Please provide either argument of perihelion (aop) or longitude of perihelion (lop)/longitude of ascending node (lan) as input")

		if 'top' in self.input:
			self.elements[:,5] = self.input['top']
		elif 'man' in self.input:
			self.elements[:,5] = self.epoch - self.input['man'] * np.power(self.input['a'], 3./2)
		else:
			raise ValueError("Please provide either time of perihelion passage (top) or mean anomaly (man)/semi-major axis (a) as input")


class CartesianPopulation(Population):
	'''
	Input: a dictionary with (x,y,z,vx,vy,vz) entries 
	'''
	def __init__(self, elements, epoch):
		self.input = elements
		n = len(elements[list(elements.keys())[0]])
		Population.__init__(self, n, 'cartesian', epoch)
		self.elements[:,0] = self.input['x']
		self.elements[:,1] = self.input['y']
		self.elements[:,2] = self.input['z']
		self.elements[:,3] = self.input['vx']
		self.elements[:,4] = self.input['vy']
		self.elements[:,5] = self.input['vz']




	

