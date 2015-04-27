
from distutils.core import setup, Extension
import numpy as np
import numpy.distutils.misc_util
from Cython.Distutils import build_ext

setup(
    name = 'parametric',
    version = '1.0',
    description = 'C wrapper for parametric function created by Haotian Pang et. al',
    author = 'John Pura',
    author_email = 'john.pura@duke.edu',
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("parametric",
                           sources=["_parametric.pyx", "parametric.c",
                                    "lu.c","tree.c","linalg.c",
                                    "heap.c"],
                 include_dirs=[np.get_include(), ])]
)