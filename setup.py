from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import os
import sys

# Output directory for the shared library
output_dir = os.path.abspath("atommover")  # put the shared lib inside the package
os.makedirs(output_dir, exist_ok=True)

# Compiler flags
extra_compile_args = []
if sys.platform == "win32":
    extra_compile_args = ['-Wno-unused-parameter', '-Wno-unused-variable', '-Wno-parentheses']
else:
    extra_compile_args = [
        "-fPIC",
        "-Wno-unused-but-set-variable",
        "-Wno-unused-variable",
        "-Wno-sign-compare",
        "-Wno-unreachable-code",
    ]

# Define the dummy Extension (needed to hook build_ext)
c_extension = Extension(
    name="libmatching_placeholder",  # not used for Python import
    sources=[
        "bottleneckBipartiteMatching.c",
        "matrixUtils.c",
        "mmio.c",
        "extern/cheap.c",
        "extern/matching.c",
    ],
    include_dirs=["extern"],
    extra_compile_args=extra_compile_args,
    libraries=["m"],
)

# Custom build_ext to produce a shared library
class BuildSharedLibrary(build_ext):
    def build_extensions(self):
        for ext in self.extensions:
            objects = self.compiler.compile(
                ext.sources,
                include_dirs=ext.include_dirs,
                extra_postargs=ext.extra_compile_args,
            )
            # Link into a shared object inside the package directory
            lib_name = "libmatching_for_PPSU"
            if sys.platform == "win32":
                lib_file = os.path.join(output_dir, lib_name + ".dll")
            elif sys.platform == "darwin":
                lib_file = os.path.join(output_dir, lib_name + ".dylib")
            else:
                lib_file = os.path.join(output_dir, lib_name + ".so")
            self.compiler.link_shared_object(objects, lib_file, libraries=ext.libraries)
            print(f"Built shared library: {lib_file}")

# Standard Python package setup
setup(
    name="atommover",
    version="0.1.0",
    packages=find_packages(),  # includes 'atommover'
    python_requires=">=3.8",
    install_requires=[
        # Add Python dependencies here (e.g., "numpy>=1.25")
    ],
    ext_modules=[c_extension],
    cmdclass={"build_ext": BuildSharedLibrary},
    include_package_data=True,  # ensure shared lib is included
)
