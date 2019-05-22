# DESTNOSIM

#### Introduction
This repository has the basis of what will eventually become the DES TNO catalog search survey simulator. Currently, I'm implementing a bunch of tools that allows us to produce a synthetic population of TNOs and observe them with an arbitrary set of DECam exposures. I am also implementing DES-specific tools for the additional information we have for DES exposures (for example, astrometric covariance matrices and solutions, completeness functions for point sources). This is a work in progress.

#### Dependencies
Python:
- `numpy`
- `astropy`
- `scipy`
- (When astrometry is implemented): `pixmappy` (https://github.com/gbernstein/pixmappy)

External:
- `orbitspp` (https://github.com/gbernstein/orbitspp) and dependencies

#### Installation
Right now, just make sure your environment has a `ORBITSPP` variable pointing to the `bin` folder of your `orbitspp` installation. This is not yet a proper Python package, so you'll have to manually add the `des` and `tno` folders to your path.
