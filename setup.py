# -*- coding: UTF-8 -*-
#! /usr/bin/python

from pathlib import Path
from setuptools import setup, find_packages
# added to run some files with pyccel : @bahari
from setuptools.command.install import install
import subprocess

# ...
# Read library version into '__version__' variable
path = Path(__file__).parent / 'simplines' / 'version.py'
exec(path.read_text())
# ...

NAME    = 'simplines'
VERSION = __version__
AUTHOR  = 'Ahmed RATNANI'
EMAIL   = 'ratnaniahmed@gmail.com'
URL     = 'https://github.com/ratnania/simplines'
DESCR   = 'TODO.'
KEYWORDS = ['math']
LICENSE = "LICENSE"

setup_args = dict(
    name                 = NAME,
    version              = VERSION,
    description          = DESCR,
    long_description     = open('README.md').read(),
    author               = AUTHOR,
    author_email         = EMAIL,
    license              = LICENSE,
    keywords             = KEYWORDS,
    url                  = URL,
)

# ...
packages = find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"])
# ...

# Dependencies
install_requires = [
    'numpy',
    ]

# List of files to process with pyccel @bahari
files_to_process = [
    'simplines/ad_mesh_core.py',
    'simplines/results_f90_core.py',
    'simplines/fast_diag_core.py',
]
# @bahari
class CustomInstallCommand(install):
    """Custom installation command to run pyccel on multiple files."""
    def run(self):
        # Run the standard install process
        install.run(self)
        # Process each file with pyccel
        for file in files_to_process:
            print(f"Running pyccel on {file}...")
            try:
                subprocess.check_call(['pyccel', file])
            except subprocess.CalledProcessError as e:
                print(f"Error occurred while processing {file}: {e}")
                raise


def setup_package():
    setup(packages=packages, \
          include_package_data=True, \
          install_requires=install_requires, \
         **setup_args)

if __name__ == "__main__":
    setup_package()
