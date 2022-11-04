from .decam import *
from ..tno.magnitude import * 

class PointingGroup(DECamExposure):
	def __init__(self, ra, dec, exposure_list, expnum1, expnum2, mjd1, mjd2, band, m25 = None, k1 = None, k2 = None, c = None):
		self.exposure_list = exposure_list
		self.expnum1 = expnum1
		self.expnum2 = expnum2
		self.ra = ra
		if self.ra > 180:
			self.ra -= 360
		self.dec = dec
		self.mjd1 = mjd1
		self.mjd2 = mjd2
		self.band = band
		self.corners = {}

		self.m25 = m25 
		self.k1 = k1 
		self.k2 = k2 
		self.c = c

	def collectExposures(self, survey):
		self.exposures = {} 

		for i in self.exposure_list:
			try:
				self.exposures[i] = survey[i]
			except:
				print(f'EXPNUM {i} not in survey')

	def probDetection(self, m):
		'''
		Computes the detection probability of something with magnitude m in this pointing group
		
		Arguments:
		- m: magnitude (numpy array or float)
		
		Returns:
		- detection probability, same shape as m
		'''
		if self.m25 != None:
			term1 = (1. + np.exp(self.k1 * (m - self.m25))) 
			term2 = (1. + np.exp(self.k2 * (m - self.m25)))
			return self.c/(term1 * term2)
		else:
			return np.zeros_like(m)

	def calculateRate(self, population, expmin = 50, ccdmask = [2,61], compute_mag = False):
		'''
		Uses the positions of the population to determine the rates, and also checks whether the source would be
		inside a functional CCD for the duration of the stare

		Arguments:
		- population: Population object from tno/population containing the input orbits and detections
		- expmin (int): minimum number of exposures the source has to be inside the CCD
		- ccdmask (list): list of masked CCDs
		'''
		try:
			population.obs
		except:
			raise AttributeError("Population has no obs attribute. Perhaps you need to call survey.observePopulation first?")

		## first, reduce population to this PG

		pop_pg = population.obs[np.isin(population.obs['EXPNUM'], self.exposure_list)]
		
		if len(pop_pg) == 0:
			return tb.Table()
		## see which objects are actually inside a searched CCD
		unique_ccds, counts = np.unique(pop_pg['ORBITID', 'CCDNUM'], return_counts = True)
		unique_ccds = tb.Table(unique_ccds)
		unique_ccds['COUNTS'] = counts 
		unique_ccds = unique_ccds[(unique_ccds['COUNTS'] > expmin) & (unique_ccds['CCDNUM'] != 0)]
		unique_ccds = unique_ccds[np.isin(unique_ccds['CCDNUM'], ccdmask, invert = True)]

		## go to rates 
		pop_inccd = pop_pg[np.isin(pop_pg['ORBITID'], unique_ccds['ORBITID'])]

		# this will be in degrees
		pop_inccd['THETA_X'], pop_inccd['THETA_Y'] = self.gnomonicProjection(pop_inccd['RA'], pop_inccd['DEC'])

		# don't see a way to avoid this for loop yet
		pop_inccd.add_index('ORBITID')

		rates = []
		if compute_mag:
			mags = []
		for i in np.unique(pop_inccd['ORBITID']):
			obj = pop_inccd.loc[i]
			deltax = obj['THETA_X'][np.argmax(obj['EXPNUM'])] - obj['THETA_X'][np.argmin(obj['EXPNUM'])]
			deltay = obj['THETA_Y'][np.argmax(obj['EXPNUM'])] - obj['THETA_Y'][np.argmin(obj['EXPNUM'])]

			r = np.sqrt(deltax**2 + deltay**2)/(self.exposures[np.max(obj['EXPNUM'])].mjd - self.exposures[np.min(obj['EXPNUM'])].mjd)

			rates.append(r)
			if compute_mag:
				mags.append(np.mean(obj['MAG']))

		t = tb.Table()
		t['ORBITID'] = np.unique(pop_inccd['ORBITID'])
		t['RATE'] = np.array(rates) * (3600 / 0.263)
		if compute_mag:
			t['MAG'] = mags
		#t = tb.join(t, pop_pg)
		return t 




class DEEP(Survey):
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
		pointing_group = '{}/{}.pointing.fits'.format(orbdata, self.release)
		
		exp = tb.Table.read(track, 1)
		try:
			Survey.__init__(self, exp['EXPNUM'], exp['RA'], exp['DEC'], exp['MJD_MID'], exp['BAND'], track = track, corners = corners)
		except:
			Survey.__init__(self, exp['expnum'], exp['ra'], exp['dec'], exp['mjd_mid'], exp['filter'], track = track, corners = corners)
		self.exp = exp
		## ensure keys are capitalized
		for i in self.exp.keys():
			self.exp.rename_column(i, i.upper())
		if 'FILTER' in self.exp.keys():
			self.exp.rename_column('FILTER', 'BAND')

		self.pg = tb.Table.read(pointing_group)

		self.params = [self.pg.meta['R50_1'], self.pg.meta['R50_2'], self.pg.meta['KAPPA1'], self.pg.meta['KAPPA2'], self.pg.meta['C']] 
		self.split = self.pg.meta['SPLIT']


	def __getitem__(self, key):
		'''
		Allows DECamExposures to be accessed by indexing the Survey object
		'''
		try:
			return self.exposures[key]
		except AttributeError:
			try:
				return self.pointing_groups[key]
			except AttributeError:
				raise AttributeError("Survey does not have a list of DECamExposures or PointingGroups!")
		except KeyError:
			try:
				return self.pointing_groups[key]
			except:
				raise KeyError("Exposure or PointingGroup {} not in survey".format(key))


	def createPointingGroup(self):
		''' 
		Creates the dictionary of pointing groups for the survey
		'''

		try:
			self.exposures
		except:
			self.createExposures()

		self.pointing_groups = {} 
		for i in self.pg:
			exp_list = self.exp[(self.exp['EXPNUM'] >= i['EXPNUM1']) & (self.exp['EXPNUM'] < i['EXPNUM2'])]['EXPNUM']
			self.pointing_groups[i['POINTING_GROUP']] = PointingGroup(i['RA_PG'], i['DEC_PG'], exp_list, i['EXPNUM1'], i['EXPNUM2'],
																	  i['MJD1'], i['MJD2'], i['BAND'], i['m25'], i['k1'], i['k2'], i['c'])
		for i in self.pointing_groups:
			self.pointing_groups[i].collectExposures(self)

	def probRate(self, rate):
		'''
		Computes the probability that this object would be detected by kbmod in this pointing group 
		with the given rate

		Arguments:
		- rate: rate of motion in pixel/hour (numpy array or float)
		
		Returns:
		- detection probability, same shape as rate
		'''

		# r50_1, r50_2, kappa1, kappa2, c = params 
		return detprob_piecewise(rate, self.params, self.split)



	def observePopulation(self, population, lightcurve = None, keepall = False, expmin = 50, ccdmask = [2,61]):
		'''
		Uses the population's magnitudes and rates to check if a detection is observable given each pointing group's completeness

		Arguments:
		- population: Population object to be observed, requires magnitudes of each object
		- lightcurve: list (or single) of lightcurve objects to these can be applied for the objects. If a list is provided, make sure len(lightcurve) == len(population)
		- keepall: boolean, will split orbits between detections and non-detections if True
		- expmin (int): minimum number of exposures the source has to be inside the CCD
		- ccdmask (dictionary of list): dictionary with a  list of masked CCDs per PG
		'''
		try:
			population.observations
		except:
			raise AttributeError("Population has no observations")


		try:
			self.pointing_groups
		except:
			self.createPointingGroup()

		#exp = tb.join(self.exp, self.pg)

		#we don't need all exposures - should save memory
		#exp = self.exp[np.isin(self.exp['EXPNUM'], population.observations['EXPNUM'])]

		obs = tb.join(population.observations, self.exp['EXPNUM', 'BAND', 'MJD_MID'])
		obs.rename_column('MJD_MID', 'MJD')

		# remove what doesn't have observations before joining!
		mags = population.mag_obs[np.isin(population.mag_obs['ORBITID'], np.unique(obs['ORBITID']))]
		
		obs = tb.join(population.observations, mags)

		population.obs = obs

		if lightcurve is not None:
			population.generateLightCurve(lightcurve)

		del mags


		## now onto rates, the main change here 
		r = [] 
		for i in self.pg['POINTING_GROUP']:
			rate = self.pointing_groups[i].calculateRate(population, expmin, ccdmask, compute_mag = True)
			if len(rate) > 0:
				rate['POINTING_GROUP'] = i
				r.append(rate)

		r = tb.vstack(r)

		population.rates = r #tb.join(population.obs, r)
		population.rates = tb.join(population.rates, self.pg['POINTING_GROUP', 'EXPNUM1', 'EXPNUM2', 'MJD1', 'MJD2', 'm25', 'c', 'k1', 'k2'])

		### rate selection function

		denom  = (1. + np.exp(population.rates['k1'] * (population.rates['MAG'] - population.rates['m25'])))
		denom *= (1. + np.exp(population.rates['k2'] * (population.rates['MAG'] - population.rates['m25'])))
		population.rates['DETPROB'] = population.rates['c']/denom
														


		population.rates['RANDOM'] = np.random.rand(len(population.rates))
		del population.rates['m25', 'c', 'k1', 'k2']


		population.rates['DETPROB'] *= self.probRate(population.rates['RATE'])


		if not keepall:
			ndet = population.rates[population.rates['DETPROB'] < population.rates['RANDOM']]
			obs = population.rates[population.rates['DETPROB'] > population.rates['RANDOM']]
			population.ndet = ndet
		else:
			obs = population.rates


		p1 = obs['ORBITID', 'EXPNUM1', 'MJD1', 'POINTING_GROUP', 'MAG', 'RATE', 'DETPROB', 'RANDOM']
		p1.rename_column('EXPNUM1', 'EXPNUM')
		p1.rename_column('MJD1', 'MJD')

		p2 = obs['ORBITID', 'EXPNUM2', 'MJD2', 'POINTING_GROUP', 'MAG', 'RATE', 'DETPROB', 'RANDOM']
		p2.rename_column('EXPNUM2', 'EXPNUM')
		p2.rename_column('MJD2', 'MJD')
		p = tb.vstack([p1, p2])
		p.rename_column('MAG', 'MEAN_MAG')

		population.detections = tb.join(p, population.obs)
