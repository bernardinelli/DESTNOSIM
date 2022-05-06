# DESTNOSIM

#### Introduction
This repository contains the DES TNO survey simulator, as well as a bunch of tools for generating synthethic TNO populations, and for dealing with DES exposure information. This software has been described in detail in:
- [Bernardinelli et. al. 2020](https://ui.adsabs.harvard.edu/abs/2020ApJS..247...32B/abstract), the Y4 data release
- [Bernardinelli et. al. 2022](https://ui.adsabs.harvard.edu/abs/2022ApJS..258...41B/abstract), the Y6 data release

While this software can handle both the Y4 and Y6 releases, we suggest that only Y6 is used for any statistical studies. 

#### Dependencies
Python:
- `numpy`
- `astropy`
- `scipy`
- `numba` 
- `spacerocks` (https://github.com/kjnapier/spacerocks)
- Optional: `pixmappy` (https://github.com/gbernstein/pixmappy). This is only required if you wish to access some of the WCS functionalities of this package. If all you intend to do with this package is TNO simulations, you don't need `pixmappy`.

External:
- Optional: `orbitspp` (https://github.com/gbernstein/orbitspp) and dependencies

#### Installation

The Python package can be installed in a standard way: 
```
    pip install destnosim
```

In order to use the `des`-specific functions, you'll need the correspondent exposure and ccd corner files (located in the [`data`](data/) directory). Their location should be specified by defining an environment variable called `DESDATA`. The [`data`](data/) folder needs to be downloaded separately from the cloning/download of the repository, as the files sizes are a bit too large for GitHub's standard file system. 


The `orbitspp` installation is detailed in that package's page. It can be a bit tricky, and requires many different pieces to be put together. Make sure your environment has a `ORBITSPP` variable pointing to the `bin` folder of your `orbitspp` installation. This will link the C++ software with some of the tools present here.

If you do *not* install `orbitspp`, you necessarily have to use `spacerocks` for ephemerides generation. See [here](Examples/spacerocks.ipynb) for a discussion.



### Usage
A simple tutorial for the software included here is included [here](Examples/DESTNOSIM_Tutorial.ipynb). This tutorial projects the almost 70 thousand synthethic objects from the [CFEPS-L7 model](http://www.cfeps.net/?page_id=105) into the DES exposures and evaluates the detectability of all objects. For more advanced usage, the user is encouraged to delve into the files and the documentation. 

**WARNING**

In some versions of MacOS, there is a safety feature that does not propagate environment variables to shells that are generated inside other processes. What this means for `destnosim` is that `orbitspp` calls from inside python shells do *not* load `orbitspp` correctly. This is unfortunate, and a fix is forthcoming. In the meanwhile, the code will print the correct command line calls to `orbitspp`, and you can run these in your own shell. We apologize for the somewhat messy problem!