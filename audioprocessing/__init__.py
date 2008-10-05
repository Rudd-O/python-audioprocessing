#!/usr/bin/env python

import numpy
import array
import scipy
import scipy.signal
import math
import sys
import wave

"""
Python audio processing suite

This is a toolkit of convenience functions for audio processing.
Distributed under the GPL v3.  Copyright Manuel Amador (Rudd-O).
"""


def chunked(stream,samples):
	"""Given a sequence, returns it in chunks of len(samples) or less"""
	offset = 0
	while offset < len(stream):
		yield stream[offset:offset+samples]
		offset = offset + samples


def mix_down(stereosignal,channels):
	"""Given a 1D numpy array where samples from each channel are
	interspersed like this:
		LRLRLRLRLRLR (stereo)
	or
		123456123456 (5.1)
	returns the mono mixdown of the signal.
	
	Resulting stream is floating-point.  If channels=1, the stereo signal
	is converted from int to float but no other changes take place.
	"""
	if channels == 1: return stereosignal * 1.
	shape = (len(stereosignal)/channels,channels)
	return stereosignal.reshape(shape).sum(1) / float(channels)


def average(series):
	"""Given a series of numbers, return their average
	in floating point."""
	return sum(series) / float(len(series))


def calculate_rms(chunk):
	"""Given a numpy array, return its RMS power level
	in floating point.  Returns the absolute of the same value (in FP)
	if given only one sample.
	"""
	try: len(chunk)
	except TypeError: chunk = numpy.array([chunk])
	chunk = pow(abs(chunk),2)
	return math.sqrt ( average(chunk) )


def calculate_rms_dB(chunk,zero_dB_equiv=1.0):
	"""Given a numpy array with a signal, calculate its RMS power level
	and return it multiplied by ten.  The value is normalized by the
	maximum value, so an RMS power level equal to max should
	return 0 dB.  If a single sample is passed, the peak dB power level
	of that sample is calculated instead."""
	return 10 * math.log10 ( calculate_rms(chunk) / float(zero_dB_equiv) )

calculate_dB = calculate_rms_dB


def find_signal_onset(samples,sampling_rate):
	"""Chunks the samples in 11.337 ms chunks (give or take a few samples)
	calculates RMS power levels of each chunk, and compares each power
	level to the previous one.
	The first comparison that yields a greater-than-2-dB increase stops
	the process, and returns the sample offset at which it happened.
	Calculations are performed lazily for performance reasons.
	This works on a 1-channel signal only.
	"""

	def calcdb(samples,chunksize):
		for chunk in chunked(samples,chunksize):
			yield calculate_rms_dB(chunk)

	def diffs(lst):
		olddatum = None
		for n,datum in enumerate(lst):
			if n != 0: yield datum - olddatum
			else: yield None
			olddatum = datum

	chunksize = sampling_rate/500
	threshold = 2

	for chunknum,dB_diff in enumerate(diffs(calcdb(samples,chunksize))):
		if dB_diff > threshold: return chunknum * chunksize


def analyze_spectrum(signal,numbands):
	"""Computes amplitudes, discarding the zero freq and the
	above-Nyquist freqs. Auto-pads signals nonmultple of numbands * 2,
	auto-averages results from streams longer than numbands * 2.

	Returns a list of amplitudes, one for each band.  Values represent
	the raw sample value / strength of the signal in each band.
	"""

	npoints = numbands * 2

	def dofft(chunk):
		if len(chunk) < npoints:
			# pad chunk with zeros in case it is too short
			chunk = list(chunk) + [0.] * ( npoints  - len(chunk) )
			chunk = scipy.array(chunk)
		# ham the chunk
		hammed = chunk * scipy.signal.hamming(len(chunk))
		# compute amplitude for each band
		fftresult = pow(abs(scipy.fft(hammed)),2) / npoints
		# return, discarding above-nyquist frequency amplitudes
		return fftresult[ 1 : 1 + numbands ]

	specs = numpy.array([ dofft(chunk) for chunk in chunked(signal,npoints) ])
	spectrum = specs.sum(0) / specs.shape[0] # average along the 0th axis
	return spectrum


def analyze_spectrum_dB(data,numbands,zero_dB_equiv=1.0):
	"""Same as analyze_spectrum() but transforms each strength into dB.
	Equivalent to [ calculate_dB(m) for m in analyze_spectrum(...) ]
	"""
	return [ calculate_dB(m,zero_dB_equiv) for m in analyze_spectrum(data,numbands) ]


def dB_to_char(dB_value):
	def diddlydum(lst):
		for n,y in enumerate(lst):
			if n==0: continue
			yield (lst[n-1],y)

	codes = "-zyxwvutsrqponmlkjihgfedcba012+"
	brackets = [-scipy.inf] + list(numpy.linspace(-81.+1.5,6.+1.5,len(codes)-1)) + [scipy.inf]
	for n,(lbound,ubound) in enumerate(diddlydum(brackets)):
		if lbound < dB_value <= ubound: return codes[n]
	assert False

def dB_to_string(dB_value_list):
	return "".join([ dB_to_char(val) for val in dB_value_list ])


# === signatures ===

def wav_butterscotch(filename,use_dB=False,full_spectrum=False):

	f = wave.open(filename,"r")
	if f.getcomptype() != "NONE":
		raise NotImplementedError, "compression type %r not supported"%f.getcomptype()
	if f.getsampwidth() != 2:
		raise NotImplementedError, "sample width %d bits not supported"%f.getsampwidth()*8
	if f.getframerate() != 44100:
		raise NotImplementedError, "sampling rate %d not supported"%f.getframerate()
	if f.getnframes() == 0:
		raise wave.Error, "zero-length files not supported"
	sampling_rate = f.getframerate()
	sample_width = f.getsampwidth()
	channels = f.getnchannels()
	numframes = f.getnframes()

	buf = ""
	while f.tell() < numframes:
		buf += f.readframes(sampling_rate)
		samples = mix_down(numpy.array(array.array("h",buf)),channels)
		pos = find_signal_onset(samples,sampling_rate)
		if pos is not None: break

	if pos is None: pos = 0
	f.setpos(pos) # go to onset of signal

	bands =128 # below-nyquist bands, and we will trim the high end later
	# FIXME accept other sampling freqs, we need to scale the number of bands
	# or compute it proportionally to the sampling rate, so that the bandwidth
	# of each band stays constant and we can detect duplicates with
	# different sampling rate
	spectrums = []
	expected_bytes = sampling_rate*10*sample_width*channels
	for x in xrange(6):
		# read ten seconds
		buf = f.readframes(sampling_rate*10)
		# break on less than ten seconds of data
		if len(buf) < expected_bytes: break
		# mix those ten seconds down
		stream = mix_down(scipy.array(array.array("h",buf)),channels)
		# make the stream signed floating point, value 1.0 is 0 dB
		stream /= 32768.
		# get the spectrum
		if use_dB: spectrum = analyze_spectrum_dB(stream,bands)
		else: spectrum = analyze_spectrum(stream,bands)
		# trim the high end
		if not full_spectrum: spectrum = spectrum[:len(spectrum) / 2]
		# accumulate the spectrum
		spectrum = numpy.array(spectrum)
		spectrums.append(spectrum)

	spectrums = numpy.array(spectrums)
	return (pos,spectrums)

def butterscotch_correlate_by_spectrum(signature1,signature2):
	correls = []
	for n in range( min( [ len(signature1), len(signature2) ] ) ):
		correls.append(numpy.corrcoef(signature1[n],signature2[n])[1][0])
	return correls

def butterscotch_correlate_by_band(signature1,signature2):
	correls = []
	signature1,signature2 = [ s.transpose() for s in (signature1,signature2) ]
	for n in range( min( [ len(signature1), len(signature2) ] ) ):
		correls.append(numpy.corrcoef(signature1[n],signature2[n])[1][0])
	return correls