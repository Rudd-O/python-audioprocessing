#!/usr/bin/env python

from distutils.core import setup

setup(
	name = 'python-audioprocessing',
	version = '0.0.1',
	description = 'A set of convenience tools to process audio',
	long_description = """The Python audio processing suite is a set of tools to process audio.""",
	author = 'Manuel Amador (Rudd-O)',
	author_email = 'rudd-o@rudd-o.com',
	license = "GPL",
	url = 'http://projects.rudd-o.com/python-audioprocessing',
	packages = ['audioprocessing'],
	scripts = ["butterscotch","butterscotch-correlate","butterscotch-batchanalyze"],
	keywords = "audio signal processing fft spectrum analyzer",
)
