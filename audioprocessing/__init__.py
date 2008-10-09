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


def chunked(stream,count):
	"""Given a sequence, returns it in chunks of len(samples) or less"""
	offset = 0
	length = len(stream)
	while offset < length:
		yield stream[offset:offset+count]
		offset = offset + count


def in_pairs(lst):
	"""Given an iterable, it returns each element with its next in a tuple."""
	for n,y in enumerate(lst):
		if n==0: continue
		yield (lst[n-1],y)


def deltas(lst):
	"""Given an iterable, it returns the difference of each element and its preceding one."""
	for before,after in in_pairs(lst): yield after - before


def log2_average(arr):
	if math.log(len(arr),2) != int(math.log(len(arr),2)):
		raise ValueError, "array length must be a power of 2"

	avgs = []
	while len(arr) > 1:
		lows = arr[:len(arr)/2]
		highs = arr[len(arr)/2:]
		avgs.append(highs.mean())
		arr = lows
	avgs.append(arr[0])
	avgs.reverse()
	return numpy.array(avgs)


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


def find_signal_onset(stream,sampling_rate):
	"""Chunks the stream in 11.337 ms chunks (give or take a few samples)
	calculates RMS power levels of each chunk, and compares each power
	level to the previous one.
	The first comparison that yields a greater-than-2-dB increase stops
	the process, and returns the sample offset at which it happened.
	Calculations are performed lazily for performance reasons.
	This works on a 1-channel signal only.
	"""

	def calcdb(stream,chunksize):
		for chunk in chunked(stream,chunksize):
			yield calculate_rms_dB(chunk)

	def diffs(lst):
		olddatum = None
		for n,datum in enumerate(lst):
			if n != 0: yield datum - olddatum
			else: yield None
			olddatum = datum

	chunksize = sampling_rate/500
	threshold = 2

	for chunknum,dB_diff in enumerate(diffs(calcdb(stream,chunksize))):
		if dB_diff > threshold: return chunknum * chunksize


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
	signal = signal.reshape((-1,npoints))

	hamming = scipy.signal.hamming(npoints)
	hamming = scipy.hstack( [ hamming for x in xrange(len(signal)) ] )
	hamming = hamming.reshape((-1,npoints))

	hammed = signal * hamming

	fft = numpy.fft.rfft(hammed)[:,1:]
	result = pow(abs(fft),2) / npoints
	result = result.mean(0)

	return result


def analyze_spectrum_by_blocks(signal,npoints,samples_per_block):
	"""Computes analyze_spectrum(...) for each block of
	block_len samples in signal."""

	return numpy.vstack(
		map(
			lambda chunk: analyze_spectrum(chunk,npoints),
			chunked(
				signal,
				samples_per_block,
			)
		)
	)


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


def read_wave(f,frames):
	"""Given a wave object, read the number of frames,
	then return a numpy array with those frames.
	this data has been mixed down and each sample has been converted
	to floating point where 0 dB = 1.0. If there wasn't enough data, all the
	data that could be read is returned.
	"""

	if f.getcomptype() != "NONE":
		raise NotImplementedError, "compression type %r not supported"%f.getcomptype()
	if f.getsampwidth() != 2:
		raise NotImplementedError, "sample width %d bits not supported"%f.getsampwidth()*8
	if f.getnframes() == 0:
		raise wave.Error, "zero-length files not supported"
	sampling_rate = f.getframerate()
	sample_width = f.getsampwidth()
	channels = f.getnchannels()

	frames = f.readframes(frames)
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

class ButterscotchSignature:
	"""A Butterscotch signature:

	self.bands: self.bands[N] = list of intensities for band N, one per block
	self.blocks: self.blocks[N]: list of intensities for frequency N, one per band"""

	def __init__(self,data,secs_per_block,audio_onset_sample,highest_freq):
		self.bands = data
		self.blocks = data.transpose()
		self.audio_onset_sample = audio_onset_sample
		self.secs_per_block = secs_per_block
		self.highest_freq = highest_freq
		self.linear_bands = True
		self.linear_intensities = True

	def freq_centerpoints(self):
		if self.linear_bands:
			return numpy.linspace(0,self.highest_freq,len(self.bands)+1)[1:]
		else:
			return log2_average(numpy.linspace(0,self.highest_freq,2**(len(self.bands)-1)+1)[1:])

	def __str__(self):
		f = self.freq_centerpoints()
		startfreq = f[0]
		endfreq = f[-1]
		if self.linear_intensities: u = "raw: 0 dB = value 1.0"
		else: u = "measured in dB"
		if self.linear_bands: l = "linear"
		else: l = "logarithmically averaged"
		return "Butterscotch signature containing %d %d-second blocks of %d %s bands (%s), from %d to %d Hz, starting at sample %d in the original audio data."%(len(self.blocks), self.secs_per_block, len(self.bands), l, u, startfreq, endfreq, self.audio_onset_sample)

	def __repr__(self):
		if self.linear_intensities: u = "raw"
		else: u = "dB"
		if self.linear_bands: l = "linear"
		else: l = "log"
		return "<Butterscotch: %s blocks, %d s/block, %d %s bands, %s, %d Hz, onset %d>"%(len(self.blocks), self.secs_per_block, len(self.bands), l, u, self.highest_freq, self.audio_onset_sample)

	def as_dB(self):
		bands = self.bands.copy()
		for n,val in enumerate(bands.flat): bands.flat[n] = calculate_dB(val)
		b = ButterscotchSignature(bands,self.secs_per_block,self.audio_onset_sample,self.highest_freq)
		b.linear_intensities = False
		return b

	def as_log_bands(self):
		logblocks = numpy.vstack( [ log2_average( b ) for b in self.blocks ] )
		logbands = logblocks.transpose()
		b = ButterscotchSignature(logbands,self.secs_per_block,self.audio_onset_sample,self.highest_freq)
		b.linear_bands = False
		return b

	def halve_highest_freq(self):
		if self.linear_bands:
			if divmod(len(self.bands),2)[1] != 0: raise ValueError, "Number of bands in original signature is odd, cannot halve odd numbers."
			bands = self.bands[:len(self.bands)/2].copy()
		else:
			if len(self.bands) == 1: raise ValueError, "Number of bands == 1, cannot halve 1"
			bands = self.bands[:len(self.bands)-1].copy()
		b = ButterscotchSignature(bands,self.secs_per_block,self.audio_onset_sample,self.highest_freq / 2)
		return b

	def display(self,title=None,async = False):
		try:
			import pylab as p
			import matplotlib.axes3d as p3 
		except ImportError: raise ImportError, "you need to install the Matplotlib and PyLab libraries to use this feature"
		fig=p.figure()
		ax = p3.Axes3D(fig)
		if title: ax.text(0,0,title)
		xs = (numpy.arange(len(self.blocks)) * self.secs_per_block ).repeat(len(self.bands))
		ys = list (self.freq_centerpoints()) * len(self.blocks)
		zs = numpy.ravel(self.blocks)
		#print self.bands.shape,len(ys),ys
		assert len(xs) == len(zs) == len(ys)
		ax.scatter3D(xs, ys , zs)
		ax.set_xlabel('Time (s)')
		if self.linear_bands: ax.set_ylabel('Frequency (linear)')
		else: ax.set_ylabel('Frequency (log)')
		if self.linear_intensities: ax.set_zlabel('Intensity (raw)')
		else: ax.set_zlabel('Intensity (dB)')
		if not async: p.show()

	def correlate(self,other):
		"""Correlates two signatures.  For performance reasons, they
		are correlated blindly."""

		corrcoef = scipy.corrcoef
		c = 0.
		for r1, r2 in zip(self.bands,other.bands):
			c += corrcoef(r1,r2)[1][0]
		return c / len(self.bands)


class NoAudioOnset(Exception): pass
class NotEnoughAudio(Exception): pass


# === the function that generates butterscotch signatures ===


def wav_butterscotch(f,
	num_blocks=12,
	secs_per_block=10,
	num_bands=64,
	force_audio_onset=None):

	if type(num_blocks) is not int or num_blocks < 1: raise ValueError, "blocks must be a positive integer"
	if secs_per_block <= 0: raise ValueError, "secs_per_ must be > 0"

	if type(f) in (str,unicode): f = wave.open(f)
	
	sampling_rate = f.getframerate()
	highest_freq = sampling_rate / 2

	if force_audio_onset is None: audio_onset = find_audio_onset(f)
	else: audio_onset = force_audio_onset
	if audio_onset is None: raise NoAudioOnset, "could not determine onset of audio"
	f.setpos(audio_onset)

	block_samples = sampling_rate * secs_per_block
	total_samples = block_samples * num_blocks

	signal = read_wave(f,total_samples) # read the wave file
	signal = signal[:len(signal)/block_samples*block_samples] # discard partial blocks

	if len(signal) < block_samples: raise NotEnoughAudio, "%d samples received, at least %d samples needed"%(len(signal),block_samples)


	analysis = analyze_spectrum_by_blocks(
		signal,
		num_bands*2,
		block_samples).transpose().copy()

	return ButterscotchSignature(
		analysis,
		secs_per_block,
		audio_onset,
		highest_freq)


# === convenience functions ===


def mp3_butterscotch(filename,*args,**kwargs):
	return wav_butterscotch(decode_mp3(filename),*args,**kwargs)



def parser(cmdline,description):
	from optparse import OptionParser

	usage = """usage: %s %s

%s"""%("%prog",cmdline,description)

	epilog = """This program only knows how to process 16-bit little-endian linear-encoded mono / stereo / multichannel RIFF WAVE files (MP3 files may work if mpg321 is installed).  It also doesn't know how to read from standard input or named pipes, since it needs to perform a seek in the audio file."""

	parser = OptionParser(usage=usage,epilog=epilog)

	parser.add_option("-d","--decibel",help="Compute decibel (dB) levels instead of raw signal values.  Useful for plotting spectrums.", action="store_true",dest="use_dB")
	parser.add_option("-l","--log-bands",help="Logarithmically average frequency bands.  Useful for emphasis in the low- and mid-frequency ranges.", action="store_true",dest="use_log_bands")
	parser.add_option("-s","--full-spectrum",help="Compute the full spectrum below the Nyquist frequency instead of discarding the high half.  Useful to plot the full spectrum, and to compare high frequency loss in lossly encoded files.  You can combine it with --fingerprint to get a quick ASCII representation of frequency bands.", action="store_true",dest="full_spectrum")

	parser.add_option("-e","--seconds-per-block",help="Change the number of seconds per block from the default.", dest="spb", type="int", default= 10)
	parser.add_option("-n","--num-blocks",help="Change the number of blocks from the default.", dest="blocks", type="int", default = 12)
	parser.add_option("-b","--num-bands",help="Change the number of frequency bands from the default.  This always refer to linear bands, even when --log-bands is specified, and if --log-bands is specified, --num-bands must be a power of two.", dest="bands", type="int", default = 64)

	return parser