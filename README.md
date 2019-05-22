# DESTNOSIM

#### Introduction
This repository has the basis of what will eventually become the DES TNO catalog search survey simulator. Currently, I'm implementing a bunch of tools that allows us to reproduce a synthetic population of TNOs for an arbitrary set of DECam exposures. I am also implementing DES-specific tools for the additional information we have from DES exposures (for example, astrometric covariance matrices and solutions, completeness functions for point sources). This is a work in progress.

#### Dependencies
Python:
- `numpy`
- `astropy`
- `scipy`

External:
- `orbitspp` (https://github.com/gbernstein/orbitspp) and dependencies

#### Installation
Right now, just make sure your environment has a `ORBITSPP` variable pointing to the `bin` folder of your `orbitspp` installation. 