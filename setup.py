from __future__ import division, absolute_import, with_statement, print_function
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob
import os
if not os.path.exists('qpu'):
    os.mkdir('./qpu')
try:
    import builtins
except:
    import __builtin__ as builtins

builtins.__POINTNET2_SETUP__ = True

_ext_src_root = os.path.abspath("_ext_qpu")
_ext_sources =  glob.glob("{}/src/*.cpp".format(_ext_src_root)) + \
                glob.glob("{}/src/*.cu".format(_ext_src_root))
_ext_headers = glob.glob("{}/include/*".format(_ext_src_root))

requirements = []


setup(
    name="qpu",
    version='1.0.0',
    author="Shaofei Qin",
    packages=find_packages(),
    install_requires=requirements,
    ext_modules=[
        CUDAExtension(
            name="qpu._ext",
            sources=_ext_sources,
            extra_compile_args={
                "cxx": ["-O2", "-I{}".format("{}/include".format(_ext_src_root))],
                "nvcc": ["-O2", "-I{}".format("{}/include".format(_ext_src_root))],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
# python setup.py build_ext --inplace