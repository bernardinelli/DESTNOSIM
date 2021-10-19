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
- Optional: `pixmappy` (https://github.com/gbernstein/pixmappy). This is only required if you wish to access some of the WCS functionalities of this package. If all you intend to do with this package is TNO simulations, you don't need `pixmappy`.

External:
- `orbitspp` (https://github.com/gbernstein/orbitspp) and dependencies

#### Installation
The `orbitspp` installation is detailed in that package's page. It can be a bit tricky, and requires many different pieces to be put together. 

The Python package can be installed in a standard way: 
```
    python3 setup.py install
```

Make sure your environment has a `ORBITSPP` variable pointing to the `bin` folder of your `orbitspp` installation. In order to use the `des`-specific functions, you'll need the correspondent exposure and ccd corner files, located in a folder defined by your `DESDATA` environment variable. The data folder is what should be inside your `DESDATA` folder and contains the required files.


### Usage
A simple tutorial for the software included here is included [here](Examples/DESTNOSIM Tutorial.ipynb). This tutorial projects the almost 70 thousand synthethic objects from the CFEPS-L7 model into the DES exposures and evaluates the detectability of all objects. For more advanced usage, the user is encouraged to delve into the files and the documentation. 