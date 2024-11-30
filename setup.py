# -*- coding: UTF-8 -*-
#! /usr/bin/python

from pathlib import Path
from setuptools import setup, find_packages
#@bahari : added class to run pyccel on specific files
from setuptools.command.install import install
from setuptools import Command  # Import Command
import subprocess

# Read library version into '__version__' variable
path = Path(__file__).parent / 'simplines' / 'version.py'
exec(path.read_text())

NAME = 'simplines'
VERSION = __version__
AUTHOR = 'Ahmed RATNANI'
EMAIL = 'ratnaniahmed@gmail.com'
URL = 'https://github.com/ratnania/simplines'
DESCR = 'TODO.'
KEYWORDS = ['math']
LICENSE = "LICENSE"

setup_args = dict(
    name=NAME,
    version=VERSION,
    description=DESCR,
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    license=LICENSE,
    keywords=KEYWORDS,
    url=URL,
)

packages = find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"])

install_requires = [
    'numpy',
]

#To pyccelize specific files, enhancing the computation speed for results and mesh adaptation. @bahari
class RunPyccelCommand(Command):
    """Custom command to run pyccel on specific files."""
    description = "Run pyccel on Python files in the specified folder."
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        # List of files to process
        files_to_process = [
            'simplines/ad_mesh_core.py',
            'simplines/results_f90_core.py',
            'simplines/fast_diag_core.py',
        ]
        for file in files_to_process:
            print(f"Running pyccel on {file}...")
            try:
                subprocess.check_call(['pyccel', file])
            except subprocess.CalledProcessError as e:
                print(f"Error occurred while processing {file}: {e}")
                raise
        # Create a folder after running pyccel
        folder_name = "newFolder"
        Path(folder_name).mkdir(parents=True, exist_ok=True)
        print(f"Folder '{folder_name}' created successfully.")

def setup_package():
    setup(
        packages=packages,
        include_package_data=True,
        install_requires=install_requires,
        cmdclass={'run_pyccel': RunPyccelCommand},  # Register custom command
        **setup_args
    )

if __name__ == "__main__":
    setup_package()

