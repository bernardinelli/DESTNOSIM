

### Following pixmappy's setup 
import glob, os  


try:
    from setuptools import setup
    import setuptools
    print("Using setuptools version",setuptools.__version__)
except ImportError:
    from distutils.core import setup
    import distutils
    print("Using distutils version",distutils.__version__)


dependencies = ['numpy', 'astropy', 'scipy', 'numba', 'spacerocks', 'rich']

with open('README.md') as file:
	long_description = file.read()


## ignoring version stuff for now 

data = glob.glob(os.path.join('data', '*'))

dist = setup(
	name = "DESTNOSIM",
	version = "1.3.3",
	author = "Pedro Bernardinelli",
	author_email = "pedrobe@sas.upenn.edu",
	description = "Python module for simulating DES TNO observations",
	long_description = long_description,
	long_description_content_type='text/markdown',
	license = "BSD License",
	url = "https://github.com/bernardinelli/DESTNOSIM",
	packages = ['destnosim', 'destnosim.des', 'destnosim.tno'],
	package_data = {'destnosim' : data},
	install_requires = dependencies,)
