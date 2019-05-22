from decam import *
import os 


class DESExposure(DECamExposure):
	def __init__(self, expnum, ra, dec, mjd_mid, band, m50 = None, c = None, k = None):
		DECamExposure.__init__(self,expnum, ra, dec, mjd_mid, band)

	def probDetection(self, m):
		return None

class DES(Survey):
	def __init__(self, expnum, ra, dec, mjd, band, release, m50 = None, c = None, k = None):
		self.release = release
		track ='{}.exposure.positions.fits'.format(self.release)
		corners = '{}.ccdcorners.fits'.format(self.release)
		Survey.__init__(self, expnum, ra, dec, mjd, band, track = track, corners = corners)

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
