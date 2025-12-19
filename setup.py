import os
import sys
import sysconfig

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

# Output directory for the shared library
PPSU_DIR = os.path.join(os.path.dirname(__file__), "PPSU2023")
EXTERN_DIR = os.path.join(os.path.dirname(__file__), "atommover", "extern")

os.makedirs(EXTERN_DIR, exist_ok=True)

# Compiler flags
extra_compile_args = []
if sys.platform == "win32":
    extra_compile_args = [
        "-Wno-unused-parameter",
        "-Wno-unused-variable",
        "-Wno-parentheses",
    ]
else:
    extra_compile_args = [
        "-fPIC",
        "-Wno-unused-but-set-variable",
        "-Wno-unused-variable",
        "-Wno-sign-compare",
        "-Wno-unreachable-code",
    ]

lib_name = "libmatching_for_PPSU"
ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
lib_file = os.path.join(EXTERN_DIR, f"{lib_name}{ext_suffix}")

# Define the dummy Extension (needed to hook build_ext)
c_extension = Extension(
    name="atommover.extern.libmatching_for_PPSU",
    sources=[
        os.path.join("PPSU2023", "bottleneckBipartiteMatching.c"),
        os.path.join("PPSU2023", "matrixUtils.c"),
        os.path.join("PPSU2023", "mmio.c"),
        os.path.join("PPSU2023", "extern", "cheap.c"),
        os.path.join("PPSU2023", "extern", "matching.c"),
    ],
    include_dirs=[os.path.join("PPSU2023", "extern")],
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
            output_file = self.get_ext_fullpath(ext.name)
            self.compiler.link_shared_object(
                objects, output_file, libraries=ext.libraries
            )
            print(f"Built shared library: {lib_file}")


# Standard Python package setup
setup(
    name="atommover",
    version="0.1.0",
    ext_modules=[c_extension],
    cmdclass={"build_ext": BuildSharedLibrary},
    include_package_data=True,  # ensure shared lib is included
)
