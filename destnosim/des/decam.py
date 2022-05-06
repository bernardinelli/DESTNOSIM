from __future__ import print_function
import numpy as np
from itertools import chain
from .ccd import *
import os 
import subprocess
import astropy.table as tb 
from scipy.spatial import cKDTree
from astropy.time import Time  
import spacerocks 

class DECamExposure:
	'''
	Base class for a DECam exposure. Contains a bunch of useful functions to deal with coordinates
	near the exposure
	'''
	def __init__(self, expnum, ra, dec, mjd_mid, band):
		'''
		Initialization function

		Arguments:
		- expnum: Exposure number 
		- ra: R.A. pointing (degrees)
		- dec: Declination pointing (degrees)
		- mjd_mid: midpoint time of the exposure (mjd)
		- band: exposure filter
		'''
		self.expnum = expnum
		self.ra = ra
		if self.ra > 180:
			self.ra -= 360
		self.dec = dec
		self.mjd = mjd_mid
		self.band = band
		self.corners = {}

	def gnomonicProjection(self, ra, dec):
		'''
		Computes the Gnomonic projection of the positions supplied centered on this exposure
		
		Arguments:
		- ra: list (or numpy array) of R.A.s (degrees)
		- dec: list (or numpy array) of Decs (degrees)
		
		Returns Gnomonic x and y, in degrees
		'''
		ra_list = np.array(ra)
		dec_list = np.array(dec)
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
		Inverts Gnomonic positions from a projection centered on the exposure
		
		Arguments:
		- x: list or array of Gnomonic x (degrees)
		- y: list or array of Gnomonic y (degrees)

		Returns R.A. and Decs, in degrees
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
		Checks if a list of RAs and Decs are inside a DECam CCD in an approximate way, returns indices of positions that are inside a CCD and which CCD they belong to

		Arguments:
		- ra_list: list of R.A.s (degrees)
		- dec_list: list of Decs (degrees)
		- ccd_tree: kD tree of the CCD positions. Can be generated using ccd.create_ccdtree()
		- ccd_keys: list of CCD correspondences with the kD tree. Can be generated using ccd.create_ccdtree()
		- ccdsize: Size, in degrees, of the Gnomonic x direction (smallest side).

		Returns:
		- List of CCD positions for each RA/Dec pair (empty if there is no CCD match)
		- List of CCDs in which each point belongs to
		'''
		if ccd_tree == None:
			ccd_tree, ccd_keys = create_ccdtree()

		x, y = self.gnomonicProjection(ra_list, dec_list)

		tree_data = np.array([x, 2*y]).T
		index = np.arange(len(ra_list))
		if len(x) > 0:
			tree = cKDTree(tree_data)
			## CCD size = 0.149931 deg
			inside_CCD = ccd_tree.query_ball_tree(tree, ccdsize, p = np.inf)
			#this is probably the most complicated Python line ever written
			if inside_CCD != None: 
				ccd_id = [len(inside_CCD[i])*[ccdnums[ccd_keys[i]]] for i in range(len(inside_CCD)) if len(inside_CCD[i]) > 0]
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
		'''
		String representation function
		'''
		return 'DECam exposure {} taken with {} band. RA: {} Dec: {} MJD: {}'.format(self.expnum, self.band, self.ra, self.dec, self.mjd)

	def checkInCCDRigorous(self, ra_list, dec_list, ccd_list):
		'''
		Checks if the object is really inside the CCD using the corners table and a ray tracing algorithm
		
		Arguments:
		- ra_list: list of R.A.s (degrees)
		- dec_list: list of Decs (degrees)
		- ccd_list: list of CCDs to be checked by the ray tracing algorithm (coming, eg, from self.checkInCCDFast). Must match RA/Dec shape 

		Returns:
		- List of booleans if the RA/Dec pair belongs to its correspondent CCD.
		'''

		try:
			self.corners
		except:
			raise AttributeError("Exposure doesn't have dict of corners. Perhaps run Survey.collectCorners?")
		
		inside = []

		for ra, dec, ccd in zip(ra_list, dec_list, ccd_list):
			try:
				inside.append(ray_tracing(ra, dec, self.corners[ccd]))
			except KeyError:
				inside.append(False)
		
		return inside




class Survey:
	'''
	A survey is just a series of exposures. Ideally, we'd have one `exposure.positions.fits` file for DESTracks usage
	This function includes convenience calls for $ORBITSPP to generate positions of a population of objects
	'''
	def __init__(self, expnum, ra, dec, mjd, band, track = None, corners = None):
		'''
		Initialization function

		Arguments:
		- expnum: list of exposure numbers
		- ra: list of R.A. pointings
		- dec: list of Dec pointings
		- mjd: list of midpoint MJDs for the exposures
		- band: list of filters for each exposure
		- track: Path of FITS file containing the exposures for use with $ORBITSPP
		- corners: Path of CCD corners FITS file containing the CCD corners of all exposures for use with $ORBITSPP
		'''
		self.ra = ra 
		self.dec = dec 
		self.mjd = mjd 
		self.expnum = expnum
		self.band = band

		self.track = track
		self.corners = corners

		self._hascorners = False

	def createExposures(self):
		'''
		Creates a dictionary of DECamExposures for the Survey inside self.exposures
		'''
		self.exposures = {}
		for ra,dec,mjd,n,b in zip(self.ra, self.dec, self.mjd, self.expnum, self.band):
			self.exposures[n] = DECamExposure(n, ra, dec, mjd, b)

	def _createEarthSpaceRock(self):
		'''
		Creates the pre-computed Earth positions from SpaceRocks
		'''
		import astropy.units as u
		table = tb.Table.read(self.track)

		self.times = Time(self.mjd, format='mjd', scale='tdb')
		units = spacerocks.Units()
		units.timescale = 'tdb'
		units.timeformat = 'mjd'
		units.speed = u.au/u.yr

		self.earth =  spacerocks.SpaceRock(x = table['observatory'][:,0], y = table['observatory'][:,1], z = table['observatory'][:,2],
										   vx = table['velocity'][:,0], vy = table['velocity'][:,1], vz = table['velocity'][:,2],
										   epoch = self.times, units = units, origin = 'ssb', frame='J2000')

	def createObservations(self, population, outputfile, useold = False, ra0 = 10, dec0 = -20, radius = 85):
		'''
		Calls $ORBITSPP/DESTracks to generate observations for the input population, saves them in the outputfile 
		and returns this table

		Arguments:
		- population: Population object from tno/population containing the input orbits
		- outputfile: Path for the output FITS file where the observations will be saved, .fits extension will be appended
		- useold: boolean, if True will check if outputfile already exists and read it, skipping the DESTracks call
		- ra0: R.A. center of the observation field
		- dec0: Dec center of the observation
		- radius: search radius for the exposures
		
		For DES usage, the last three parameters should remain constant!

		Results are stored in population.observations
		'''
		if useold and os.path.exists(outputfile + '.fits'):
			population.observations =  tb.Table.read(outputfile + '.fits')
			return None

		if population.heliocentric:
			raise ValueError("Please use barycentric elements!")

		orbitspp = os.getenv('ORBITSPP')
		with open('{}.txt'.format(outputfile), 'w') as f:
			for j,i in enumerate(population.elements):
				print(j, i[0],i[1],i[2],i[3],i[4],i[5], file = f)
				
		with open('{}.txt'.format(outputfile), 'r') as f:

			print(' '.join([orbitspp + '/DESTracks', '-cornerFile={}'.format(self.corners), 
							'-exposureFile={}'.format(self.track), '-tdb0={}'.format(population.epoch), '-positionFile={}.fits'.format(outputfile)
							,'-readState={}'.format(population.state) , '-ra0={}'.format(ra0), 
							'-dec0={}'.format(dec0),'-radius={}'.format(radius), '< {}.txt'.format(outputfile)]))

			subprocess.run(' '.join([orbitspp + '/DESTracks', '-cornerFile={}'.format(self.corners), 
							'-exposureFile={}'.format(self.track), '-tdb0={}'.format(population.epoch), '-positionFile={}.fits'.format(outputfile)
							,'-readState={}'.format(population.state) , '-ra0={}'.format(ra0), 
							'-dec0={}'.format(dec0),'-radius={}'.format(radius), ' < {}.txt'.format(outputfile)]), stdin = f, shell = True)

		if not os.path.exists(outputfile + '.fits'):
			raise ValueError("$ORBITSPP call did not terminate succesfully!")


		population.observations =  tb.Table.read(outputfile + '.fits')

	def createObservationsSpacerocks(self, population):
		'''
		Calls the Spacerocks backend to generate observations for the input population


		Arguments:
		- population: Population object for the input orbits
		'''
		## first set up times and do spacerock stuff

		self.createEarthSpaceRock()

		rock = population._generateSpaceRocks()
		prop, planets, sim = rock.propagate(epochs = self.times.mjd, model='ORBITSPP')
		del planets, sim 
		obs = prop.observe(observer=self.earth)


		## gather data into something useable

		t = tb.Table()
		t['RA'] = obs.ra.deg 
		t['RA'][t['RA'] > 180] -= 360
		t['DEC'] = obs.dec.deg 
		t['EXPNUM'] = len(population) * list(self.expnum)
		t['ORBITID'] = len(self.expnum) * list(range(len(population)))

		exp = tb.Table()
		exp['EXPNUM'] = np.array(self.expnum)
		exp['RA_CENTER'] = np.array(self.ra)
		exp['RA_CENTER'][exp['RA_CENTER'] > 180] -= 360
		exp['DEC_CENTER'] = np.array(self.dec) 

		t = tb.join(t, exp)
		t['DELTA'] = np.sqrt( (( t['RA'] - t['RA_CENTER']) * np.cos(t['DEC_CENTER'] * np.pi/180))**2 +  (t['DEC'] - t['DEC_CENTER'])**2)

		t = t[t['DELTA'] < 1.5]

		theta = bulk_gnomonic(np.array(t['RA']), np.array(t['DEC']), np.array(t['RA_CENTER']), np.array(t['DEC_CENTER']))
		#rescale for kD tree
		theta[:,1] *= 2

		ccd_tree, ccd_keys = create_ccdtree()

		tree = cKDTree(theta)
		# kD tree ccd checker
		inside_CCD = ccd_tree.query_ball_tree(tree, 0.149931 * 1.001, p = np.inf)
		
		if inside_CCD != None: 
			ccd_id = [len(inside_CCD[i])*[ccdnums[ccd_keys[i]]] for i in range(len(inside_CCD)) if len(inside_CCD[i]) > 0]
			inside_CCD = np.array(list(chain(*inside_CCD)))
			if len(inside_CCD) > 0:
				ccdlist = list(chain(*ccd_id))
			else:
				print('No observations!')
				self.population = tb.Table(column=['RA', 'DEC', 'EXPNUM', 'ORBITID'])
				return None
		else:
			print('No observations!')
			self.population = tb.Table(column=['RA', 'DEC', 'EXPNUM', 'ORBITID'])
			return None 

		t = t[inside_CCD]
		t['CCDNUM'] = ccdlist




		# we need the exposure objects here for the proper CCD stuff

		try:
			self.exposures
		except AttributeError:
			self.createExposures()

		if not self._hascorners:
			self.collectCorners()


		inside_ccd = np.zeros(len(t), dtype='bool')


		for i in range(len(t)):

			# proper corners
			if t[i]['CCDNUM'] not in self.exposures[t[i]['EXPNUM']].corners:
				ins = False
			else:
				ins = ray_tracing(t[i]['RA'], t[i]['DEC'], self.exposures[t[i]['EXPNUM']].corners[t[i]['CCDNUM']])

			inside_ccd[i] = ins 


		obs = t[inside_ccd]
		obs.sort(['ORBITID','EXPNUM'])

		population.observations = obs['RA', 'DEC', 'EXPNUM', 'CCDNUM', 'ORBITID']

	def __getitem__(self, key):
		'''
		Allows DECamExposures to be accessed by indexing the Survey object
		'''
		try:
			return self.exposures[key]
		except AttributeError:
			raise AttributeError("Survey does not have a list of DECamExposures!")
		except KeyError:
			raise KeyError("Exposure {} not in survey".format(key))

	def __len__(self):
		'''
		Returns the number of exposures
		'''
		return len(self.expnum)

	def collectCorners(self):
		'''	
		Uses the CCD corners table to build a list of CCDs
		'''
		if self.corners == None:
			raise ValueError("No table of CCD corners! Set Survey.corners first.")
		else:
			corners = tb.Table.read(self.corners)
		try:
			self.exposures
		except AttributeError:
			self.createExposures()

		corners = corners[np.isin(corners['expnum'], self.expnum)]
		#corners.add_index('expnum')

		for i in corners:
			rac = i['ra'][:-1]
			decc = i['dec'][:-1]

			self.exposures[i['expnum']].corners[i['ccdnum']] = np.array([rac,decc]).T

		self._hascorners = True





