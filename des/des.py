from decam import *
import os 
import astropy.table as tb

class DESExposure(DECamExposure):
	def __init__(self, expnum, ra, dec, mjd_mid, band, m50 = None, c = None, k = None):
		DECamExposure.__init__(self,expnum, ra, dec, mjd_mid, band)
		self.m50 = m50 
		self.c = c
		self.k = k

	def probDetection(self, m):
		'''
		Computes the detection probability of something with magnitude m in this exposure
		'''
		if self.m50 != None:
			return self.c/(1 + np.exp(self.k * (m - self.m50)))
		else:
			return np.zeros_like(m)

class DES(Survey):
	def __init__(self, release, m50 = None, c = None, k = None):
		orbdata = os.getenv('DESDATA')

		self.release = release
		track ='{}/{}.exposure.positions.fits'.format(orbdata, self.release)
		corners = '{}/{}.ccdcorners.fits'.format(orbdata, self.release)
		
		exp = tb.Table.read(track, 1)

		Survey.__init__(self, exp['expnum'], exp['ra'], exp['dec'], exp['mjd_mid'], exp['filter'], track = track, corners = corners)
		self.exp = exp

		if m50 == None:
			self.m50 = len(self.ra) * [None]
		else:
			self.m50 = m50
		if c == None:
			self.c = len(self.ra) * [None]
		else:
			self.c = c
		if k == None:
			self.k = len(self.ra) * [None]
		else:
			self.k = k

	def createExposures(self):
		self.exposures = {}
		for ra,dec,mjd,n,b, m50,c,k in zip(self.ra, self.dec, self.mjd, self.expnum, self.band, self.m50, self.c, self.k):
			self.exposures[n] = DESExposure(n, ra, dec, mjd, b, m50, c, k)
