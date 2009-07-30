#!/usr/bin/env python

from distutils.core import setup

for line in open('audioprocessing/__init__.py'):
	if line.startswith('__version__'):
		version = line.split()[-1].strip("'").strip('"')
		break
else:
	raise ValueError, '"__version__" not found in "audioprocessing/__init__.py"'

setup(
	name = 'python-audioprocessing',
	version = version,
	description = 'A set of convenience tools to process audio',
	long_description = """The Python audio processing suite is a set of tools to process audio.""",
	author = 'Manuel Amador (Rudd-O)',
	author_email = 'rudd-o@rudd-o.com',
	license = "GPL",
	url = 'http://rudd-o.com/new-projects/python-audioprocessing',
	packages = ['audioprocessing'],
	scripts = ["butterscotch","butterscotch-batchanalyze"],
	keywords = "audio signal processing fft spectrum analyzer",
)
