import numpy as np
from itertools import chain
from ccd import *


class DECamExposure:
	'''
	Base class for a DECam exposure
	'''
	def __init__(self, expnum, ra, dec, mjd_mid, band):
		self.expnum = expnum
		self.ra = ra
		if self.ra > 180:
			self.ra -= 360
		self.dec = dec
		self.mjd = mjd_mid
		self.band = band

	def gnomonicProjection(self, ra_list, dec_list):
		'''
		Gnomonic projection centered on this exposure
		'''
		ra_list = np.array(ra_list)
		dec_list = np.array(dec_list)
		ra_list[np.where(ra_list > 180)] = ra_list[np.where(ra_list > 180)] - 360
	
	
		c_dec_0 = np.cos(np.pi*self.dec/180.)
		s_dec_0 = np.sin(np.pi*self.dec/180.)
		c_dec   = np.cos(np.pi*dec_list/180.)
		s_dec   = np.sin(np.pi*dec_list/180.)
		c_ra    = np.cos(np.pi*(ra_list-self.ra)/180.)
		s_ra    = np.sin(np.pi*(ra_list-self.ra)/180.)

		cos_c = s_dec_0*s_dec + c_dec_0*c_dec*c_ra

		x = c_dec*s_ra/cos_c
		y = (c_dec_0*s_dec - s_dec_0*c_dec*c_ra)/cos_c

		return 180*x/np.pi, 180*y/np.pi

	def inverseGnomonic(self, x, y):
		'''
		Inverts the Gnomonic projection centered on the exposure
		'''
		x = np.array(x) * np.pi/180 
		y = np.array(y) * np.pi/180
		ra_rad = np.pi*self.ra/180.
		dec_rad = np.pi*self.dec/180.
		den = np.sqrt(1 + x**2 + y**2)
		sin_dec = (np.sin(dec_rad) + y*np.cos(dec_rad))/den
		sin_ra = x/(np.cos(dec_rad) * den)
		return 180*(np.arcsin(sin_ra))/np.pi + ra_rad, 180*np.arcsin(sin_dec)/np.pi 
 

	def checkInCCD(self, ra_list, dec_list, ccd_tree = None):
		'''
		Checks if a list of RAs and Decs are inside a DECam CCD, returns indices that are inside and which CCD they belong to
		'''
		if ccd_tree == None:
			ccd_tree = create_ccdtree()

		x, y = self.gnomonicProjection(ra_list, dec_list)

		tree_data = np.array([x, 2*y]).T
		index = np.arange(len(ra_list))
		if len(x) > 0:
			tree = cKDTree(tree_data)
			## CCD size = 0.149931 deg
			inside_CCD = ccd_tree.query_ball_tree(tree, 0.149931, p = np.inf)
			#this is probably the most complicated Python line ever written
			if inside_CCD != None:
				ccd_id = [len(inside_CCD[i])*[ccdnums[ccdBounds.keys()[i]]] for i in range(len(inside_CCD)) if len(inside_CCD[i]) > 0]
				inside_CCD = np.array(list(chain(*inside_CCD)))
				ccd_id = list(chain(*ccd_id))
				return index[inside_CCD], ccd_id
			else:
				return None, None
		else:
			return None, None

	def __str__(self):
		return 'DECam exposure {} taken with {} band'.format(self.expnum, self.band)


class Survey:
	'''
	A survey is just a series of pointings. Ideally, we'd have one `exposure.positions.fits` file for DESTracks usage
	'''
	def __init__(self, expnum, ra, dec, mjd, band, track = None, corners = None):
		self.ra = ra 
		self.dec = dec 
		self.mjd = mjd 
		self.expnum = expnum
		self.band = band

		self.track = track
		self.corners = corners

	def createExposures(self):
		self.exposures = {}
		for ra,dec,mjd,n,b in zip(self.ra, self.dec, self.mjd, self.expnum, self.band):
			self.exposures[n] = DECamExposure(n, ra, dec, mjd, b)

