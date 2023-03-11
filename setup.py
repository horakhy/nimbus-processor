from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        "mainAPS",
        sources=["mainAPS.pyx"],
        extra_compile_args=["-Wl,--export-dynamic"],
        include_dirs=[np.get_include()],
        # define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        libraries=[],
        compiler="gcc",
        language="c",
    )
]

setup(
    name="mainAPS",
    ext_modules=cythonize(ext_modules, compiler_directives={'language_level': "3"}),
)