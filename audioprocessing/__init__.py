#!/usr/bin/env python

import array
import numpy
import scipy
import audioprocessing.stream
import audioprocessing.signal
import audioprocessing.util

"""
Python audio processing suite

This is a toolkit of convenience functions for audio processing.
Distributed under the GPL v3.  Copyright Manuel Amador (Rudd-O).
"""

__version__ = '0.0.7'


# === signatures ===

class ButterscotchSignature:
	"""A Butterscotch signature:

	self.bands: self.bands[N] = list of intensities for band N, one per block
	self.blocks: self.blocks[N]: list of intensities for frequency N, one per band"""

	def __init__(self,data,secs_per_block,audio_onset_sample,highest_freq):
		if data.dtype != numpy.dtype("float32"):
			 data = data.astype("float32")
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
			return audioprocessing.util.log2_average(numpy.linspace(0,self.highest_freq,2**(len(self.bands)-1)+1)[1:])

	def __str__(self):
		f = self.freq_centerpoints()
		startfreq = f[0]
		endfreq = f[-1]
		if self.linear_intensities: u = "raw: 0 dB = value 1.0"
		else: u = "measured in dB"
		if self.linear_bands: l = "linear"
		else: l = "logarithmically averaged"
		return "Butterscotch signature containing %d %d-second blocks of %d %s bands (%s), whose centers range from %d to %d Hz, starting at sample %d in the original audio data."%(len(self.blocks), self.secs_per_block, len(self.bands), l, u, startfreq, endfreq, self.audio_onset_sample)

	def __repr__(self):
		if self.linear_intensities: u = "raw"
		else: u = "dB"
		if self.linear_bands: l = "linear"
		else: l = "log"
		return "<Butterscotch: %s blocks, %d s/block, %d %s bands, %s, %d Hz, onset %d>"%(len(self.blocks), self.secs_per_block, len(self.bands), l, u, self.highest_freq, self.audio_onset_sample)

	def as_dB(self):
		if not self.linear_intensities:
			raise ValueError, "cannot convert to dB scale - already converted"
		if not self.linear_bands:
			raise ValueError, "cannot convert to dB scale if bands are logarithmic - try converting the other way around"
		bands = self.bands.copy()
		for n,val in enumerate(bands.flat): bands.flat[n] = audioprocessing.signal.calculate_dB(val)
		b = ButterscotchSignature(bands,self.secs_per_block,self.audio_onset_sample,self.highest_freq)
		b.linear_intensities = False
		b.linear_bands = self.linear_bands
		return b

	def as_log_bands(self):
		if not self.linear_bands:
			raise ValueError, "cannot convert to logarithmic bands - already converted"
		logblocks = numpy.vstack( [ audioprocessing.util.log2_average( b ) for b in self.blocks ] )
		logbands = logblocks.transpose()
		b = ButterscotchSignature(logbands,self.secs_per_block,self.audio_onset_sample,self.highest_freq)
		b.linear_bands = False
		b.linear_intensities = self.linear_intensities
		return b

	def halve_highest_freq(self):
		if self.linear_bands:
			if divmod(len(self.bands),2)[1] != 0: raise ValueError, "Number of bands in original signature is odd, cannot halve odd numbers."
			bands = self.bands[:len(self.bands)/2].copy()
		else:
			if len(self.bands) == 1: raise ValueError, "Number of bands == 1, cannot halve 1"
			bands = self.bands[:len(self.bands)-1].copy()
		b = ButterscotchSignature(bands,self.secs_per_block,self.audio_onset_sample,self.highest_freq / 2)
		b.linear_bands = self.linear_bands
		b.linear_intensities = self.linear_intensities
		return b

	def halve_block_count(self):
		if divmod(len(self.blocks),2)[1] != 0: raise ValueError, "Number of blocks in original signature is odd, cannot halve odd numbers."
		blocks = self.blocks[:len(self.blocks)/2]
		bands = blocks.transpose().copy()
		b = ButterscotchSignature(bands,self.secs_per_block,self.audio_onset_sample,self.highest_freq)
		b.linear_bands = self.linear_bands
		b.linear_intensities = self.linear_intensities
		return b

	def correlate(self,other):
		"""Correlates two signatures.  For performance reasons, they
		are correlated blindly."""

		corrcoef = scipy.corrcoef
		c = 0.
		minimum = min( [ len(self.bands[0]) , len(other.bands[0]) ] )
		for r1, r2 in zip(self.bands,other.bands):
			c += corrcoef(r1[:minimum],r2[:minimum])[1][0]
		return c / len(self.bands)


# === the function that generates butterscotch signatures ===


class NotEnoughAudio(Exception):
	"""butterscotch() raises this exception if the supplied audio file
	does not contain enough audio data for the first block."""
	pass


def butterscotch(f,
	num_blocks=30,
	secs_per_block=4,
	num_bands=256,
	force_audio_onset=None):
	"""Compute a Butterscotch signature for the given path name."""

	if type(num_blocks) is not int or num_blocks < 1: raise ValueError, "blocks must be a positive integer"
	if secs_per_block <= 0: raise ValueError, "secs_per_ must be > 0"

	f =  audioprocessing.stream.AudioOnsetDiscarder(
		audioprocessing.stream.MonoStream(
			audioprocessing.stream.FloatStream(
				audioprocessing.stream.NumPyStream(
					audioprocessing.stream.decode(f)
				)
			)
		)
	)

	if f.getframerate() != 44100:
		raise NotImplementedError, "Reading from files not 44100 Hz not implemented, because we would need to adjust the number of points in the FFT to scale, and we are too lazy to program that yet!"

	sampling_rate = f.getframerate()
	highest_freq = sampling_rate / 2

	block_samples = sampling_rate * secs_per_block

	analysis = []
	for block in range(num_blocks):
		signal = f.readframes(block_samples)
		if len(signal) < block_samples:
			if block == 0: # not enough audio
				raise NotEnoughAudio, "%d samples received, at least %d samples needed"%(len(signal),block_samples)
			else: # end of wave file
				break
		result = audioprocessing.signal.analyze_spectrum(signal,num_bands*2)
		analysis.append(result)

	analysis = numpy.vstack(analysis).transpose().copy()

	return ButterscotchSignature(
		analysis,
		secs_per_block,
		f.audio_onset_sample,
		highest_freq)


# === convenience functions ===


def wav_butterscotch(filename,*args,**kwargs):
	"""Compatibility function.  Just runs butterscotch()."""
	return butterscotch(filename,*args,**kwargs)


def mp3_butterscotch(filename,*args,**kwargs):
	"""Compatibility function.  Just runs butterscotch()."""
	return butterscotch(filename,*args,**kwargs)


def parser(cmdline,description):
	"""Convenient command line parser for common options in all console
	programs shipped on this suite."""
	from optparse import OptionParser

	usage = """usage: %s %s

%s"""%("%prog",cmdline,description)

	epilog = """This program only knows how to process 16-bit little-endian linear-encoded mono / stereo / multichannel RIFF WAVE files (MP3 files may work if mpg321 is installed).  It also doesn't know how to read from standard input or named pipes, since it needs to perform a seek in the audio file."""

	parser = OptionParser(usage=usage,epilog=epilog)

	parser.add_option("-d","--decibel",help="Compute decibel (dB) levels instead of raw signal levels (the default).  Useful for plotting spectrums.", action="store_true",dest="use_dB")
	parser.add_option("-l","--linear-bands",help="Leave the frequency bands linear instead of logarithmically averaging them (the default).  Useful for plotting spectrums.", action="store_true",dest="use_linear_bands")
	parser.add_option("-s","--full-spectrum",help="Compute the full spectrum below the Nyquist frequency instead of discarding the high half (the default).  Useful to plot the full spectrum, and to compare high frequency loss in lossly encoded files.", action="store_true",dest="use_full_spectrum")

	parser.add_option("-e","--seconds-per-block",help="Change the number of seconds per block from the default 4.", dest="spb", type="int", default= 4)
	parser.add_option("-n","--num-blocks",help="Change the number of blocks from the default 30.", dest="blocks", type="int", default = 30)
	parser.add_option("-b","--num-bands",help="Change the number of frequency bands from the default 256.  This always refer to linear bands, even when --log-bands is specified, and if --log-bands is specified, --num-bands must be a power of two.  Unless --full-spectrum is also specified, the resulting bands will be the lower half and the number of bands will also be half.", dest="bands", type="int", default = 256)

	return parser


__all__ = [
 	'butterscotch',
	'mp3_butterscotch',
	'wav_butterscotch',
	'signal',
	'stream',
	'util',
	'parser',
	'ButterscotchSignature',
	'NotEnoughAudio',
]
