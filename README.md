# DESTNOSIM

#### Introduction
This repository has the basis of what will eventually become the DES TNO catalog search survey simulator. Currently, I'm implementing a bunch of tools that allows us to produce a synthetic population of TNOs and observe them with an arbitrary set of DECam exposures. I am also implementing DES-specific tools for the additional information we have for DES exposures (for example, astrometric covariance matrices and solutions, completeness functions for point sources). This is a work in progress.

#### Dependencies
Python:
- `numpy`
- `astropy`
- `scipy`
- `numba` 
- `pixmappy` (https://github.com/gbernstein/pixmappy)

External:
- `orbitspp` (https://github.com/gbernstein/orbitspp) and dependencies

#### Installation
Make sure your environment has a `ORBITSPP` variable pointing to the `bin` folder of your `orbitspp` installation. In order to use the `des`-specific functions, you'll need the correspondent exposure and ccd corner file, located in a folder defined by your `DESDATA` environment variable. The file `desdata.tar.gz` should be extracted inside the `DESDATA` folder and contains the required files. You may need to compile the `tno/popstat` program for the numba libraries.

This is not yet a proper Python package, so you'll have to manually add the `des` and `tno` folders to your path.
