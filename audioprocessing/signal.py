#!/usr/bin/env python

import numpy
import math
from audioprocessing.util import chunked,in_pairs
import scipy
import scipy.signal

def calculate_rms(chunk):
	"""Given a numpy array, return its RMS power level
	in floating point.  Returns the absolute of the same value (in FP)
	if given only one sample.
	"""
	try:
		len(chunk)
		chunk = pow(abs(chunk),2)
		return math.sqrt ( chunk.mean() )
	except TypeError: return chunk


def calculate_rms_dB(chunk,zero_dB_equiv=1.0):
	"""Given a numpy array with a signal, calculate its RMS power level
	and return it multiplied by ten.  The value is normalized by the
	maximum value, so an RMS power level equal to max should
	return 0 dB.  If a single sample is passed, the peak dB power level
	of that sample is calculated instead."""
	normalized = calculate_rms(chunk) / float(zero_dB_equiv)
	if normalized == 0.0: return -numpy.inf
	try: return 10 * math.log10 ( normalized )
	except OverflowError: return -numpy.inf

calculate_dB = calculate_rms_dB


def analyze_spectrum(signal,npoints):
	"""Computes FFT for the signal, discards the zero freq and the
	above-Nyquist freqs. Auto-pads signals nonmultple of npoints,
	auto-averages results from streams longer than npoints.
	Thus, npoints results in npoints/2 bands.

	Returns a numpy array, each element represents the raw amplitude
	of a frequency band.
	"""

	signal = signal.copy()
	if divmod(len(signal),npoints)[1] != 0:
		round_up = len(signal) / npoints * npoints + npoints
		signal.resize( round_up )

	window = scipy.signal.hanning(npoints)
	window_blocks = scipy.vstack( [ window for x in xrange(len(signal) / npoints) ] )

	signal_blocks = signal.reshape((-1,npoints))

	windowed_signals = signal_blocks * window_blocks

	ffts = numpy.fft.rfft(windowed_signals)[:,1:]

	result = pow(abs(ffts),2) / npoints
	result = result.mean(0)

	return result


def dB_to_char(dB_value):

	codes = "-zyxwvutsrqponmlkjihgfedcba012+"
	brackets = [-scipy.inf] + list(numpy.linspace(-81.+1.5,6.+1.5,len(codes)-1)) + [scipy.inf]
	for n,(lbound,ubound) in enumerate(in_pairs(brackets)):
		if lbound < dB_value <= ubound: return codes[n]
	assert False, "%s could not be quantized"%dB_value


def dB_to_string(dB_value_list):
	return "".join([ dB_to_char(val) for val in dB_value_list ])


__all__ = [
	'calculate_rms',
	'calculate_rms_dB',
	'calculate_dB',
	'analyze_spectrum',
	'dB_to_char',
	'dB_to_string',
]
