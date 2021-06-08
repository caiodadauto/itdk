import os

try:
    from Cython.Distutils import build_ext
except ImportError:
    raise ImportError("Cython is not installed")
from setuptools import Extension
from setuptools.dist import Distribution


def build(setup_kwargs):
    ext = [Extension(name="itdk.graph_utils.paths", sources=["itdk/graph_utils/_paths.cpp", "itdk/graph_utils/paths.pyx"], extra_compile_args=["-fopenmp"], extra_link_args=["-fopenmp"], language="c++")]

    setup_kwargs.update({
        'ext_modules': ext,
        'cmdclass': {'build_ext': build_ext}
    })
