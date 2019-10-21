from __future__ import print_function
import numpy as np
from itertools import chain
import ccd
import os 
import subprocess
import astropy.table as tb 
from scipy.spatial import cKDTree

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
		x = np.array(x) * np.pi/180.
		y = np.array(y) * np.pi/180.
		ra_rad = np.pi*self.ra/180.
		dec_rad = np.pi*self.dec/180.
		den = np.sqrt(1 + x**2 + y**2)
		sin_dec = (np.sin(dec_rad) + y*np.cos(dec_rad))/den
		sin_ra = x/(np.cos(dec_rad) * den)
		return 180.*(np.arcsin(sin_ra))/np.pi + self.ra, 180.*np.arcsin(sin_dec)/np.pi 
 

	def checkInCCDFast(self, ra_list, dec_list, ccd_tree = None, ccd_keys = None, ccdsize = 0.149931):
		'''
		Checks if a list of RAs and Decs are inside a DECam CCD in an approximate way, returns indices that are inside and which CCD they belong to
		'''
		if ccd_tree == None:
			ccd_tree, ccd_keys = ccd.create_ccdtree()

		x, y = self.gnomonicProjection(ra_list, dec_list)

		tree_data = np.array([x, 2*y]).T
		index = np.arange(len(ra_list))
		if len(x) > 0:
			tree = cKDTree(tree_data)
			## CCD size = 0.149931 deg
			inside_CCD = ccd_tree.query_ball_tree(tree, ccdsize, p = np.inf)
			#this is probably the most complicated Python line ever written
			if inside_CCD != None: 
				ccd_id = [len(inside_CCD[i])*[ccd.ccdnums[ccd_keys[i]]] for i in range(len(inside_CCD)) if len(inside_CCD[i]) > 0]
				inside_CCD = np.array(list(chain(*inside_CCD)))
				if len(inside_CCD) > 0:
					return index[inside_CCD], list(chain(*ccd_id))
				else:
					return [], None
			else:
				return [], None
		else:
			return [], None

	def __str__(self):
		return 'DECam exposure {} taken with {} band. RA: {} Dec: {} MJD: {}'.format(self.expnum, self.band, self.ra, self.dec, self.mjd)

	def checkInCCDRigorous(self, ra_list, dec_list, ccd_list):
		'''
		Checks if the object is really inside the CCD using the corners table and a ray tracing algorithm
		'''
		from ccd import ray_tracing

		try:
			self.corners
		except:
			raise AttributeError("Exposure doesn't have dict of corners. Perhaps run Survey.collectCorners?")
		
		inside = []

		for ra, dec, ccd in zip(ra_list, dec_list, ccd_list):
			inside.append(ray_tracing(ra, dec, self.corners[ccd]))
		
		return inside






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
		'''
		Creates a dictionary of DECamExposures for the Survey
		'''
		self.exposures = {}
		for ra,dec,mjd,n,b in zip(self.ra, self.dec, self.mjd, self.expnum, self.band):
			self.exposures[n] = DECamExposure(n, ra, dec, mjd, b)

	def createObservations(self, population, outputfile):
		'''
		Calls ORBITSPP/DESTracks to generate observations for the input population, saves them in the outputfile 
		and returns this table
		'''
		orbitspp = os.getenv('ORBITSPP')
		with open('{}.txt'.format(outputfile), 'w') as f:
			for j,i in enumerate(population.elements):
				print(j, i[0],i[1],i[2],i[3],i[4],i[5], file = f)
				
		with open('{}.txt'.format(outputfile), 'r') as f:

			print(' '.join([orbitspp + '/DESTracks', '-cornerFile={}'.format(self.corners), 
							'-exposureFile={}'.format(self.track), '-tdb0={}'.format(population.epoch), '-positionFile={}.fits'.format(outputfile)
							,'-readState={}'.format(population.state) ,'< {}.txt'.format(outputfile)]))

			subprocess.call([orbitspp + '/DESTracks', '-cornerFile={}'.format(self.corners), 
							'-exposureFile={}'.format(self.track), '-tdb0={}'.format(population.epoch), '-positionFile={}.fits'.format(outputfile)
							,'-readState={}'.format(population.state)], stdin = f)

		population.observations =  tb.Table.read(outputfile + '.fits')

	def __getitem__(self, key):
		try:
			return self.exposures[key]
		except AttributeError:
			raise AttributeError("Survey does not have a list of DECamExposures!")
		except KeyError:
			raise KeyError("Exposure {} not in survey".format(key))

	def __len__(self):
		return len(self.expnum)

	def collectCorners(self):
		'''	
		Uses the CCD corners table to build a list of CCDs
		'''
		if self.corners == None:
			raise ValueError("No table of CCD corners!")
		else:
			corners = tb.Table.read(self.corners)
		try:
			self.exposures
		except AttributeError:
			self.createExposures()

		
		corners.add_index('expnum')

		for i in self.exposures:
			try:
				exp = corners.loc[i]

				self.exposures[i].corners = {}

				for j in exp:
					ra = j['ra'][:-1]
					dec = j['dec'][:-1]

					self.exposures[i].corners[j['ccdnum']] = [[r,d] for r,d in zip(ra,dec)]
			except:
				self.exposures[i].corners = None






