#!/usr/bin/env python

import array
import numpy
import scipy
import scipy.signal
import math
import sys
import wave
import tempfile
import subprocess
import os

"""
Python audio processing suite

This is a toolkit of convenience functions for audio processing.
Distributed under the GPL v3.  Copyright Manuel Amador (Rudd-O).
"""

__version__ = '0.0.2'


# === subprocess utilities ===


class CapturedProcessError(subprocess.CalledProcessError):
	def __init__(self,returncode,cmd,output):
		subprocess.CalledProcessError.__init__(self,returncode,cmd)
		self.output = output
	def __str__(self):
		parent_text = subprocess.CalledProcessError.__str__(self)
		this_text = "Process output follows:"
		return "\n".join([parent_text,this_text,self.output])


def capture_call(*args,**kwargs):
	"""Same parameters of subprocess.check_call, except for stdout and stderr,
	which are not allowed.

	In case of failure, returns CapturedProcessError (descendant of
	CalledProcessError) to callee, with an extra member "output" containing
	the stderr/stdout output of the command, merged.
	"""

	if "stderr" in kwargs or "stdout" in kwargs:
		raise ValueError, "Neither stderr nor stdout allowed as arguments"

	output_fd,output = tempfile.mkstemp()
	kwargs["stdout"] = output_fd
	kwargs["stderr"] = subprocess.STDOUT

	try: subprocess.check_call(*args,**kwargs)
	except subprocess.CalledProcessError, e:
		raise CapturedProcessError(e.returncode,e.cmd,file(output).read(-1))
	finally:
		try: unlink(output)
		except: pass
		try: os.close(output_fd)
		except: pass


# === iterables processing primitives ===


def chunked(stream,samples):
	"""Given a sequence, returns it in chunks of len(samples) or less"""
	offset = 0
	length = len(stream)
	while offset < length:
		yield stream[offset:offset+samples]
		offset = offset + samples


def in_pairs(lst):
	"""Given an iterable, it returns each element with its next in a tuple."""
	for n,y in enumerate(lst):
		if n==0: continue
		yield (lst[n-1],y)


def deltas(lst):
	"""Given an iterable, it returns the difference of each element and its preceding one."""
	for before,after in in_pairs(lst): yield after - before


# === audio processing pipelines


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
	hamming = scipy.signal.hamming(npoints)

	def dofft(chunk):
		if len(chunk) < npoints:
			# pad chunk with zeros in case it is too short
			chunk = list(chunk) + [0.] * ( npoints  - len(chunk) )
			chunk = scipy.array(chunk)
		# compute amplitude for each band, using hamming window
		fftresult = scipy.fft(chunk * hamming)
		# return, discarding above-nyquist frequency amplitudes
		return pow(abs(fftresult[ 1 : 1 + numbands ]),2) / npoints

	specs = numpy.vstack([ dofft(chunk) for chunk in chunked(signal,npoints) ])
	spectrum = specs.mean(0) # average along the 0th axis
	return spectrum


def analyze_spectrum_dB(data,numbands,zero_dB_equiv=1.0):
	"""Same as analyze_spectrum() but transforms each strength into dB.
	Equivalent to [ calculate_dB(m) for m in analyze_spectrum(...) ]
	"""
	return [ calculate_dB(m,zero_dB_equiv) for m in analyze_spectrum(data,numbands) ]


def dB_to_char(dB_value):

	codes = "-zyxwvutsrqponmlkjihgfedcba012+"
	brackets = [-scipy.inf] + list(numpy.linspace(-81.+1.5,6.+1.5,len(codes)-1)) + [scipy.inf]
	for n,(lbound,ubound) in enumerate(in_pairs(brackets)):
		if lbound < dB_value <= ubound: return codes[n]
	assert False, "%s could not be quantized"%dB_value

def dB_to_string(dB_value_list):
	return "".join([ dB_to_char(val) for val in dB_value_list ])


# === decoders, readers and other stuff that operates with on-disk files ===


def find_audio_onset(f):
	"""Seeks a wave object to position 0, then returns the sample number where it took place, or None if it could not be found.  Position of wave object is not changed by this function."""

	oldpos = f.tell()
	try:
		f.setpos(0) # go to first sample
		buf = ""
		while f.tell() < f.getnframes():
			frames = f.readframes(f.getframerate())
			buf += frames
			samples = mix_down(numpy.core.fromstring(buf,dtype=numpy.dtype("int16")),f.getnchannels())
			pos = find_signal_onset(samples,f.getframerate())
			if pos is not None: break
		if pos is None: f.setpos(0)
		return pos
	finally:
		f.setpos(oldpos)


def read_wave(f,secs):
	"""Given a wave object, read ( object's sampling rate * secs ) frames,
	then return a numpy array with `secs` seconds of sample data;
	this data has been mixed down and each sample has been converted
	to floating point where 0 dB = 1.0. If there wasn't enough data, all the
	data that could be read is returned.
	"""

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

	frames = f.readframes(sampling_rate*secs)
	buf = numpy.core.fromstring(frames,dtype=numpy.dtype("int16"))
	stream = mix_down(buf,channels)
	stream /= 32768.
	return stream


def decode_mp3(mp3file,numframes=None,firstframe=None):
	"""Decodes an MP3 file using mpg321, returns a wave object to a temporary file with the WAV data (file is unlinked before returning)"""

	if numframes is not None and ( type(numframes) is not int or numframes < 1):
		raise TypeError, "numframes must be a positive int, not %s"%numframes
	if firstframe is not None and ( type(firstframe) is not int or numframes < 1):
		raise TypeError, "firstframe must be a positive int, not %s"%firstframe
	test = file(mp3file,"r")
	test.close()

	command = ["mpg321","--quiet"]
	if numframes is not None: command += ["-n",str(numframes)]
	if firstframe is not None: command += ["-k",str(firstframe)]

	try:
		wavfile_fd,wavfile = tempfile.mkstemp()
		command += ["-w",wavfile,mp3file]
		capture_call(command)
		return wave.open(os.fdopen(os.dup(wavfile_fd)))
	finally:
		try: os.unlink(wavfile)
		except: pass
		try: os.close(wavfile_fd)
		except: pass


# === signatures ===


class NotEnoughAudio(Exception): pass # raised by Butterscotch signature generator
def butterscotch(stream,sampling_rate,secs_per_block,use_dB=False,full_spectrum=False):
	# This function expects the first sample of stream to be the onset
	# of signal, and expects the signal to be 60 seconds long

	if sampling_rate != 44100: # FIXME
		raise NotImplementedError, "Butterscotching for sample rate %d not implemented"%sampling_rate

	if len(stream) < sampling_rate*secs_per_block:
		raise NotEnoughAudio,"%d samples received, at least %d samples needed"%(len(stream),sampling_rate*secs_per_block)

	# FIXME accept other sampling freqs, we need to scale
	# the number of bands
	# or compute it proportionally to the sampling rate,
	# so that the bandwidth
	# of each band stays constant and we can detect duplicates with
	# different sampling rate
	bands = 64 # below-nyquist bands - we cut the high half later
	spectrums = []
	for block in chunked(stream,sampling_rate*secs_per_block):
		# if chunk is incomplete, we cannot compute the FFT
		if len(block) < sampling_rate*secs_per_block: break
		# do frequency analysis
		if use_dB: spectrum = analyze_spectrum_dB(block,bands)
		else: spectrum = analyze_spectrum(block,bands)
		# trim the high end
		if not full_spectrum: spectrum = spectrum[:len(spectrum) / 2]
		# accumulate the spectrum
		spectrums.append(spectrum)

	try: return numpy.vstack(spectrums)
	except ValueError: return None


class NoAudioOnset(Exception): pass # raised by wav_butterscotch
def wav_butterscotch(f, blocks = 12, secs_per_block = 10,
					use_dB=False,full_spectrum=False):

	if type(blocks) is not int or blocks < 1:
		raise ValueError, "blocks must be a positive integer"
	if secs_per_block <= 0:
		raise ValueError, "secs_per_block must be > 0"

	if type(f) in (str,unicode): f = wave.open(f)

	seconds = blocks * secs_per_block

	pos = find_audio_onset(f)
	if pos is None:
		raise NoAudioOnset, "could not determine onset of audio"
	f.setpos(pos)

	stream = read_wave(f,seconds)
	return (pos,butterscotch(stream,f.getframerate(),
		secs_per_block,use_dB,full_spectrum))


def mp3_butterscotch(filename, blocks = 12, secs_per_block = 10,
					use_dB=False,full_spectrum=False):
	waveobject = decode_mp3(filename)
	return wav_butterscotch(waveobject,blocks,
		secs_per_block,use_dB,full_spectrum)


# correlators


def butterscotch_correlate_by_spectrum(signature1,signature2):
	"""Correlates two signatures.  Normally the signatures's rows are
	the frequency bands, but if you're correlating by time instead of
	by spectrum, a row would be the series of intensities in a particular
	band.

	If the two signatures don't match in frequency band count
	or number of samples, the signatures are automatically compared
	by the minimum number of frequencies and samples in all of the
	signatures.  Silently."""

	if signature1.shape != signature2.shape:
		h = min((signature1.shape[0],signature2.shape[0]))
		w = min((signature1.shape[1],signature2.shape[1]))
		signature1 = signature1[0:h,0:w]
		signature2 = signature2[0:h,0:w]

	c = 0.
	for s1row, s2row in zip(signature1,signature2):
		c += numpy.corrcoef(s1row,s2row)[1][0]
	return c / len(signature1)


def butterscotch_correlate_by_band(signature1,signature2):
	signature1,signature2 = [ s.transpose() for s in (signature1,signature2) ]
	return butterscotch_correlate_by_spectrum(signature1,signature2)
