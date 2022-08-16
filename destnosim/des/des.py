from .decam import *
import os 
import astropy.table as tb
from astropy.wcs import WCS

try:
	import pixmappy
	pix = True
except ModuleNotFoundError:
	pix = False

class DESExposure(DECamExposure):
	'''
	DECamExposure with additional quantities from DES analysis. Examples: astrometry covariances and exposure completeness
	'''
	def __init__(self, expnum, ra, dec, mjd_mid, band, m50 = None, c = None, k = None, cov = None):
		'''
		Initialization class

		Arguments:
		- expnum: Exposure number 
		- ra: R.A. pointing (degrees)
		- dec: Declination pointing (degrees)
		- mjd_mid: midpoint time of the exposure (mjd)
		- band: exposure filter
		- m50: magnitude of 50% completeness
		- c: efficiency for completeness
		- k: transition slope for completeness
		- cov: astrometric covariance matrix
		'''
		DECamExposure.__init__(self,expnum, ra, dec, mjd_mid, band)
		self.m50 = m50 
		self.c = c
		self.k = k
		#cov is in mas^2
		xx = cov[0]
		yy = cov[1]
		xy = cov[2]
		self.cov = np.array([[xx, xy], [xy, yy]]) * (1/3600000**2)
		

	def probDetection(self, m):
		'''
		Computes the detection probability of something with magnitude m in this exposure
		
		Arguments:
		- m: magnitude (numpy array or float)
		
		Returns:
		- detection probability, same shape as m
		'''
		if self.m50 != None:
			return self.c/(1. + np.exp(self.k * (m - self.m50)))
		else:
			return np.zeros_like(m)

	def getPixelCoordinates(self, ra, dec, ccd, c = 0.61):
		'''
		Given a list of RAs and Decs on a given CCD, returns the x,y coordinates of the detections
		using the pixmappy WCSs

		Arguments:
		- ra: list of R.A.s (degrees)
		- dec: list of Decs (degrees)
		- ccd: list of CCD numbers for WCS mapping
		'''
		try:
			return self.wcs[ccd].toPix(ra, dec, c = c)
		except:
			raise ValueError("No CCD solution for {}/{}!".format(self.expnum, ccd))

	def createWCSDict(self, pmc = None):
		'''
		Uses pixmappy to grab the CCD WCS solution

		Arguments:
		- pmc: pixmappy DESMaps() instance (optional)
		'''
		if not pix:
			raise ModuleNotFoundError("You need pixmappy for this function!")
		self.wcs = {}
		self.fullwcs = True
		if pmc == None:
			from pixmappy import DESMaps
			pmc = DESMaps()

		for i in range(1,63):
			if i != 61:
				try:
					self.wcs[i] = pmc.getDESWCS(self.expnum, i)
				except:
					self.wcs[i] = None
					if i != 31 or i != 2:
						self.fullwcs = False
	
	def getWCS(self, ccdnum, pmc = None):
		'''
		Uses pixmappy to grab the CCD WCS solution

		Arguments:
		- ccdnum: CCD number for solution
		- pmc: pixmappy DESMaps() instance (optional)

		'''
		if not pix:
			raise ModuleNotFoundError("You need pixmappy for this function!")

		if pmc == None:
			from pixmappy import DESMaps
			pmc = DESMaps()

		return pmc.getDESWCS(self.expnum, ccdnum)

	def makeWCS(self, expinfo):
		'''
		Generates a WCS dictionary using astropy.wcs and the wcs table for the exposure. 
		Note that these are not the pixmappy solutions, so are less accurate

		Arguments:
		- expinfo: table of WCS information for the exposures coming from FITS headers
		'''
		self.wcs_db = {}
		for i in expinfo:
			d = {k:i[k] for k in i.keys()}
			self.wcs_db[i['CCDNUM']] = WCS(header=d)

	def samplePosError(self, shotnoise, n_samples):
		'''
		Finds n_samples position errors given the shot noise for each value and the
		atmospheric turbulence error matrix

		Arguments:
		- shotnoise: shot noise error in degrees
		- n_samples: number of samples from the covariance matrix to be drawn
		'''

		#shot_err = shotnoisedist.sample(n_samples)

		atm_err = np.random.multivariate_normal(np.array([0,0]), self.cov, n_samples)

		gauss = np.random.normal(size=(n_samples, 2))

		for i in range(n_samples):
			gauss[i] *= shotnoise[i]

		err = gauss + atm_err

		err[:,0] *= np.cos(self.dec * np.pi/180)

		return err

	

class DES(Survey):
	'''
	Full DES survey based on one of the release files. Examples: y4a1, y4a1c, y6a1, y6a1c
	Main change between this and Survey is the support for completeness and atmospheric turbulence terms in the exposures
	'''
	def __init__(self, release):
		'''
		Initialization function

		Arguments:
		- release: name of the release, all data files need be of the form release.EXTENSION
		'''
		orbdata = os.getenv('DESDATA')

		self.release = release
		track ='{}/{}.exposures.positions.fits'.format(orbdata, self.release)
		corners = '{}/{}.ccdcorners.fits.gz'.format(orbdata, self.release)
		
		exp = tb.Table.read(track, 1)
		try:
			Survey.__init__(self, exp['EXPNUM'], exp['RA'], exp['DEC'], exp['MJD_MID'], exp['BAND'], track = track, corners = corners)
		except:
			Survey.__init__(self, exp['expnum'], exp['ra'], exp['dec'], exp['mjd_mid'], exp['filter'], track = track, corners = corners)
		self.exp = exp

		if 'm50' in exp.keys():
			self.m50 = exp['m50']
			self.c = exp['c']
			self.k = exp['k']
		else:
			self.m50 = len(exp) * [None]
			self.c = len(exp) * [None]
			self.k = len(exp) * [None]

		if 'cov' in exp.keys():
			self.cov = exp['cov']
		else:
			self.cov = len(exp) * [np.array([[0,0],[0,0]])]


	def createExposures(self):
		'''
		Makes a dictionary of DESExposures, which can be accessed by calling DES[exposure]
		'''
		self.exposures = {}
		for ra, dec, mjd, n, b, m50, c, k, cov in zip(self.ra, self.dec, self.mjd, self.expnum, self.band, self.m50, self.c, self.k, self.cov):
			self.exposures[n] = DESExposure(n, ra, dec, mjd, b, m50, c, k, cov)

	def observePopulation(self, population, lightcurve = None, keepall = False):
		'''
		Uses the population's magnitudes to check if a detection is observable given each exposure completeness

		Arguments:
		- population: Population object to be observed, requires magnitudes of each object
		- lightcurve: list (or single) of lightcurve objects to these can be applied for the objects. If a list is provided, make sure len(lightcurve) == len(population)
		- keepall: boolean, will split orbits between detections and non-detections if True
		'''
		try:
			population.observations
		except:
			raise AttributeError("Population has no observations")

		try:
			self.exposures
		except:
			raise AttributeError("Survey does not have a list of DESExposures")

		exp = tb.Table()
		exp['EXPNUM'] = self.expnum
		exp['BAND'] = self.band
		exp['m_50'] = self.m50
		exp['C'] = self.c 
		exp['K'] = self.k 
		exp['MJD'] = self.mjd

		#we don't need all exposures - should save memory
		exp = exp[np.isin(exp['EXPNUM'], population.observations['EXPNUM'])]

		obs = tb.join(population.observations, exp)

		# remove what doesn't have observations before joining!
		mags = population.mag_obs[np.isin(population.mag_obs['ORBITID'], np.unique(obs['ORBITID']))]
		
		obs = tb.join(obs, mags)

		population.obs = obs

		if lightcurve is not None:
			population.generateLightCurve(lightcurve)

		del exp, mags


		population.obs['DETPROB'] = population.obs['C']/(1. + np.exp(population.obs['K'] * (population.obs['MAG'] - population.obs['m_50'])))


		population.obs['RANDOM'] = np.random.rand(len(population.obs))

		del population.obs['m_50', 'C', 'K']

		if not keepall:
			ndet = population.obs[population.obs['DETPROB'] < population.obs['RANDOM']]
			obs = population.obs[population.obs['DETPROB'] > population.obs['RANDOM']]
			population.ndet = ndet
		else:
			obs = population.obs

		population.detections = obs






