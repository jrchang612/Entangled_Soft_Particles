import numpy
import os, sys, os.path, tempfile, subprocess, shutil
from sys import platform
from distutils.core import setup
# from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

cymodule = 'angle'

setup(
	name='angle',
	version='1.0.0',
	author='Ray Chang',
	author_email='jrc612@stanford.edu',
	license='MIT',
	ext_modules=cythonize([ Extension(cymodule, [cymodule + '.pyx'],
		include_dirs=[numpy.get_include()],
		)]),
	libraries=[],
)