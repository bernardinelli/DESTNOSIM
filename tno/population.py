from __future__ import print_function
import numpy as np
import os
import subprocess
import astropy.table as tb 


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

		if self.state = 'elements':
			self.state = 'F'
		else:
			self.state = 'T'

	def generateObservations(self, survey, outputfile):
		orbitspp = os.getenv('ORBITSPP')
		with open('elements.dat', 'w') as f:
			for i in self.elements:
				print(i[0],i[1],i[2],i[3],i[4],i[5], file = f)

		subprocess.call([orbitspp + '/DESTracks', '-cornerFile {}'.format(survey.corner), 
						'-exposureFile {}'.format(survey.track), '-tdb0 {}'.format(self.epoch), '-positionFile {}'.format(outputfile)
						,'-readState {}'.format(self.state) ,' < elements.dat'])

		return tb.Table.read(outputfile)



		

	def generateMagnitudes(self, distribution):
		'''
		Depends on magnitude distributions from `magnitude.py`
		'''
		return None


#class IsotropicPopulation(Population):