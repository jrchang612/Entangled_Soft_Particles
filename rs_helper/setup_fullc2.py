from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
cymodule = 'rock_string_helpers_fullc2'

setup(
  name='rock_string2',
  ext_modules=[Extension(cymodule, [cymodule + '.pyx'],)],
  cmdclass={'build_ext': build_ext},
  include_dirs=[numpy.get_include()]
)