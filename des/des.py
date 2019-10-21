from decam import *
import os 
import astropy.table as tb
from astropy.wcs import WCS

class DESExposure(DECamExposure):
	'''
	DECamExposure with value added quantities from DES analysis. Examples: better astrometry and exposure completeness
	'''
	def __init__(self, expnum, ra, dec, mjd_mid, band, m50 = None, c = None, k = None, cov = None):
		DECamExposure.__init__(self,expnum, ra, dec, mjd_mid, band)
		self.m50 = m50 
		self.c = c
		self.k = k
		self.cov = cov

	def probDetection(self, m):
		'''
		Computes the detection probability of something with magnitude m in this exposure
		'''
		if self.m50 != None:
			return self.c/(1. + np.exp(self.k * (m - self.m50)))
		else:
			return np.zeros_like(m)

	def getPixelCoordinates(self, ra, dec, ccd):
		'''
		Given a list of RAs and Decs on a given CCD, returns the x,y coordinates of the detections
		'''
		try:
			return self.wcs[ccd].toXY(ra, dec)
		except:
			raise ValueError("No CCD solution for {}/{}!".format(self.expnum, ccd))


	def createWCSDict(self, pmc = None):
		'''
		Uses pixmappy to grab the CCD WCS solution
		'''
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
		'''
		if pmc == None:
			from pixmappy import DESMaps
			pmc = DESMaps()

		return pmc.getDESWCS(self.expnum, ccdnum)

	def makeWCS(self, expinfo):
		'''
		Generates a WCS dictionary using astropy.wcs and the wcs table for the exposure. 
		Note that these are not the pixmappy solutions, so are less accurate
		'''
		self.wcs_db = {}
		for i in expinfo:
			d = {k:i[k] for k in i.keys()}
			self.wcs_db[i['CCDNUM']] = WCS(header=d)

	def samplePosError(self, shotnoisedist, n_samples):
		'''
		Finds n_samples position errors given a shot noise distribution (derived from the distribution.py file) and the
		atmospheric turbulence error matrix
		'''

		shot_err = shotnoisedist.sample(n_samples)

		atm_err = np.random.multivariate_normal(np.array([0,0]), self.cov, n_samples)

		ones = np.zeros((n_samples, 2))

		for i in range(n_samples):
			ones[i] *= shot_err[i]

		err = ones + atm_err

		err[:,1,1] *= np.cos(self.dec * np.pi/180)

		return err

	

class DES(Survey):
	'''
	Full DES survey based on one of the release files. Examples: alldes, alldes6, y4a1, y4a1c
	Main change between this and Survey is the support for completeness and atmospheric turbulence terms in the exposures
	'''
	def __init__(self, release):
		orbdata = os.getenv('DESDATA')

		self.release = release
		track ='{}/{}.exposure.positions.fits'.format(orbdata, self.release)
		corners = '{}/{}.ccdcorners.fits.gz'.format(orbdata, self.release)
		
		exp = tb.Table.read(track, 1)
		try:
			Survey.__init__(self, exp['EXPNUM'], exp['RA'], exp['DEC'], exp['MJD_MID'], exp['BAND'], track = track, corners = corners)
		except:
			Survey.__init__(self, exp['expnum'], exp['ra'], exp['dec'], exp['mjd_mid'], exp['filter'], track = track, corners = corners)
		self.exp = exp

		if 'M_50' in exp.keys():
			self.m50 = exp['M_50']
			self.c = exp['C']
			self.k = exp['K']
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

	def observePopulation(self, population, keepall = True):
		'''
		Uses the population's magnitudes to check if a detection is observable given the exposure completeness
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

		#we don't need all exposures - should save memory
		exp = exp[np.isin(exp['EXPNUM'], population.observations['EXPNUM'])]

		obs = tb.join(population.observations, exp)

		# remove what doesn't have observations before joining!
		mags = population.mag_obs[np.isin(population.mag_obs['ORBITID'], np.unique(obs['ORBITID']))]
		
		obs = tb.join(obs, mags)

		del exp, mags
		#obs.add_index('EXPNUM')

		'''det = []
		for i in np.unique(obs['EXPNUM']):
			exp_obs = tb.Table(obs.loc[i])

			try:
				exp_obs['DETPROB'] = self[i].probDetection(exp_obs['MAG'])
				det.append(exp_obs)
			except:
				pass

		det = tb.vstack(det)'''

		obs['DETPROB'] = obs['C']/(1. + np.exp(obs['K'] * (obs['MAG'] - obs['m_50'])))


		obs['RANDOM'] = np.random.rand(len(obs))

		del obs['m_50', 'C', 'K']

		if not keepall:
			obs = obs[obs['DETPROB'] > obs['RANDOM']]

		population.detections = obs






