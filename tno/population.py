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

	def createObservations(self, survey, outputfile):
		orbitspp = os.getenv('ORBITSPP')
		with open('elements.txt', 'w') as f:
			for j,i in enumerate(self.elements):
				print(j, i[0],i[1],i[2],i[3],i[4],i[5], file = f)

		print(' '.join([orbitspp + '/DESTracks', '-cornerFile={}'.format(survey.corners), 
						'-exposureFile={}'.format(survey.track), '-tdb0={}'.format(self.epoch), '-positionFile={}'.format(outputfile)
						,'-readState={}'.format(self.state) ,'< elements.txt']))

		subprocess.call([orbitspp + '/DESTracks', '-cornerFile={}'.format(survey.corners), 
						'-exposureFile={}'.format(survey.track), '-tdb0={}'.format(self.epoch), '-positionFile={}'.format(outputfile)
						,'-readState={}'.format(self.state) ,'< elements.txt'])

		return tb.Table.read(outputfile)


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


