#!/usr/bin/env python
__usage__ = "setup.py command [--options]"
__description__ = "standard install script"
__author__ = "Reed Essick <reed.essick@gmail.com"

#-------------------------------------------------

from setuptools import (setup, find_packages)
import glob

setup(
    name = 'gw-detectors',
    version = '0.0.0',
    url = 'https://git.ligo.org/reed.essick/gw-detectors',
    author = __author__,
    author_email = 'reed.essick@gmail.com',
    description = __description__,
    license = 'MIT',
    scripts = glob.glob("bin/*"),
    packages = find_packages(),
    data_files = [],
    requires = [], ### FIXME: specify requirements for numpy, h5py, lalsuite, else?
)
