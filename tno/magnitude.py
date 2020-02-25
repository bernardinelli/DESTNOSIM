'''
Series of convenience functions for working with magnitudes
'''
import os 
import numpy as np 
import astropy.table as tb 

from scipy.integrate import simps

orbdata = os.getenv('DESDATA')

ref_lambda = {'g' : 476.60, 'r' : 640.61, 'i' : 779.49, 'z' : 917.44, 'Y' : 987.45}

def create_filters(filter_file = '{}/bandpasses.fits'.format(orbdata)):
	'''
	Creates a dictionary of interpolated transmission functions for the DES filters 
	Bandpasses from Burke et al 2018 AJ 155 41
	'''
	filter_table = tb.Table.read(filter_file)
	#filter_table LAMBDA is in angstroms! Solar spectra used is in nanometers, so I'll convert first
	filter_table['l'] = filter_table['LAMBDA']/10
	filters = {}

	
	filters['g'] = lambda x : np.interp(x, filter_table['l'], filter_table['g'])
	filters['r'] = lambda x : np.interp(x, filter_table['l'], filter_table['r'])
	filters['i'] = lambda x : np.interp(x, filter_table['l'], filter_table['i'])
	filters['z'] = lambda x : np.interp(x, filter_table['l'], filter_table['z'])
	filters['Y'] = lambda x : np.interp(x, filter_table['l'], filter_table['Y'])

	filters['LAMBDA'] = filter_table['l']

	return filters

des_filters = create_filters()

def solar_spectrum(spec_file = '{}/solarspectrum.dat'.format(orbdata)):
	'''
	Interpolated solar spectrum
	Spectrum from Meftah et al 2018 A&A 611
	'''
	l, f = np.loadtxt(spec_file, unpack=True)

	return lambda x: np.interp(x, l, f)

sun_spec = solar_spectrum()


def reddening_colors(slope, ref_band):
	''' 
	Reddens the solar spectrum with a given spectral slope 
	Passband centers from Burke et al 2018 AJ 155 41
	'''

	l_ref = ref_lambda[ref_band]

	reddened = lambda x: sun_spec(x) * np.power(1 - slope, -(x - l_ref)/100)
	flux = {}

	for i in ['g', 'r', 'i', 'z', 'Y']:
		spec = lambda x : reddened(x) * des_filters[i](x)/x
		flux[i] = simps(spec(des_filters['LAMBDA']), des_filters['LAMBDA'])

	colors = {}
	for i in ['g', 'r', 'i', 'z', 'Y']:
		colors['{} - {}'.format(i, ref_band)] = -2.5 * np.log10(flux[i]/flux[ref_band])

	return colors






class LightCurve:
	pass
	#def __call__()

class RandomPhaseLightCurve(LightCurve):
	pass

