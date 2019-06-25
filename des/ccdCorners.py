import numpy as np
import ccd
import decam
from astropy.table import Table
exposures = Table.read("alldes.exposure.positions.fits", format ='fits')

for exp in exposures:
    decam_exposure = decam.DECamExposure(band = exp['filter'], ra=exp['ra'], dec = exp['dec'], mjd_mid=exp['mjd_mid'], expnum=exp['expnum'])
    for j in ccd.ccdBounds:
        ra_ccd, dec_ccd = decam_exposure.inverseGnomonic([ccd.ccdBounds[j][0], ccd.ccdBounds[j][1]], [ccd.ccdBounds[j][2],ccd.ccdBounds[j][3]])
        ra_ccd = np.append(ra_ccd, ra_ccd[[1,0]])
        dec_ccd = np.append(dec_ccd, dec_ccd[[1,0]])
        ra_ccd = np.append(ra_ccd, (ra_ccd[0] + ra_ccd[1])/2)
        dec_ccd = np.append(dec_ccd, (dec_ccd[0] + dec_ccd[1])/2)
        decam_exposure.expnum.append(exp['expnum'])
        decam_exposure.mjd.append(exp['mjd_mid'])
        decam_exposure.band.append(exp['filter'])
        decam_exposure.detpos.append(j)
        decam_exposure.ra.append(ra_ccd)
        decam_exposure.dec.append(dec_ccd[[0,3,1,2,4]])

