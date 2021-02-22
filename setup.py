#!/usr/bin/env python

from distutils.core import setup, Extension
from os.path import dirname
import os
import sys
import shutil
import fnmatch
import numpy
import site


def find_package_names (basedir='lib'):
    matches = []
    for path in (path for path, sub, files in os.walk(basedir) if '__init__.py' in files):
        components = path.split('/')
        name = '.'.join(components[1:])
        matches.append (name)
    return matches


def find_package_mapping (basedir='lib'):
    mapping = {}
    for path in (path for path, sub, files in os.walk(basedir) if '__init__.py' in files):
        components = path.split('/')
        name = '.'.join(components[1:])
        mapping[name] = path
    return mapping



# now do main setup
setup(name='gcforest',
      version = '1.1',
      description = 'gcForest implementation',
      author = 'Zhi-Hua Zhou ',
      author_email = 'zhouzh@lamda.nju.edu.cn',
      packages = find_package_names(),
      package_dir = find_package_mapping(),
      ext_modules = [])


