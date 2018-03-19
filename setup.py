from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy

exts = [Extension("fibmeasure/measure", 
                  ["fibmeasure/measure.pyx"],
                  include_dirs=[numpy.get_include()]
                  )]

setup(
	name = 'fibmeasure',
	version = '1.0',
	packages = ['fibmeasure'],
    cmdclass = {'build_ext': build_ext},
    ext_modules = cythonize(exts),
)
