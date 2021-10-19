# DESTNOSIM

#### Introduction
This repository contains the DES TNO survey simulator, as well as a bunch of tools for generating synthethic TNO populations, and for dealing with DES exposure information. This software has been described in detail in:
- [Bernardinelli et. al. 2020](https://ui.adsabs.harvard.edu/abs/2020ApJS..247...32B/abstract), the Y4 data release
- [Bernardinelli et. al. 2021](https://ui.adsabs.harvard.edu/abs/2021arXiv210903758B/abstract), the Y6 data release

While this software can handle both the Y4 and Y6 releases, we suggest that only Y6 is used for any statistical studies. 

#### Dependencies
Python:
- `numpy`
- `astropy`
- `scipy`
- `numba` 
- Optional: `pixmappy` (https://github.com/gbernstein/pixmappy)

External:
- `orbitspp` (https://github.com/gbernstein/orbitspp) and dependencies

#### Installation
Make sure your environment has a `ORBITSPP` variable pointing to the `bin` folder of your `orbitspp` installation. In order to use the `des`-specific functions, you'll need the correspondent exposure and ccd corner file, located in a folder defined by your `DESDATA` environment variable. The file `desdata.tar.gz` should be extracted inside the `DESDATA` folder and contains the required files. You may need to compile the `tno/popstat` program for the `numba` libraries.

Feel free to contact the author if you run into problems installing this package, or `orbitspp`.

This is not yet a proper Python package, so you'll have to manually add the `des` and `tno` folders to your path.

### Usage
A simple tutorial for the software included here is included [here] (Notebooks/DESTNOSIM Tutorial.ipynb). This tutorial projects the almost 70 thousand synthethic objects from the CFEPS-L7 model into the DES exposures and evaluates the detectability of all objects. For more advanced usage, the user is encouraged to delve into the files and the documentation. 