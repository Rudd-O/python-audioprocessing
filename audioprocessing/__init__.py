#!/usr/bin/env python

import numpy
import wave
import array
import alsaaudio
import scipy
import scipy.signal
import math
import sys
import time
import matplotlib
import pickle

def chunk_stream(stream,samples):
	offset = 0
	while offset < len(stream):
		yield stream[offset:offset+samples]
		offset = offset + samples


def mixdown(stereosignal,channels):
	shape = (len(stereosignal)/channels,channels)
	return stereosignal.reshape(shape).sum(1) / float(channels)

def average(series): return sum(series) / float(len(series))

def calculate_rms(chunk):
	try: len(chunk)
	except TypeError: chunk = numpy.array([chunk])
	chunk = pow(abs(chunk),2)
	return math.sqrt ( average(chunk) )

def calculate_rms_dB(chunk,max=32768):
	return 10 * math.log10 ( calculate_rms(chunk) / float(max) )

def find_first_nonsilence(samples,sampling_rate):
	"""Finds the first 6 dB jump in the signal, then explores
	the 20 milliseconds nearby and gets the sample offset
	of the highest-powered 64-sample block."""

	def calcdb(samples,chunksize):
		for chunk in chunk_stream(samples,chunksize):
			yield calculate_rms_dB(chunk)

	def diffs(lst):
		olddatum = None
		for n,datum in enumerate(lst):
			if n != 0: yield datum - olddatum
			else: yield None
			olddatum = datum

	chunksize = sampling_rate/500 # 10 ms
	threshold = 2

	for chunknum,dB_diff in enumerate(diffs(calcdb(samples,chunksize))):
		if dB_diff > threshold: return chunknum * chunksize

def get_amplitudes(chunk,npoints):
	"""Generates amplitudes, discarding the zero freq and the
	above-Nyquist freqs. Auto-pads chunks shorter than npoints,
	auto-averages chunks longer than npoints.
	Returns a list of amplitudes, half the length of npoints.
	"""
	if len(chunk) > npoints:
		amplitudes = numpy.array( [
			get_amplitudes(chunk,npoints) for chunk in
			[ c for c in chunk_stream(chunk,npoints) ]
		] )
		return average(amplitudes)
	if len(chunk) < npoints:
		 # pad array with zeros in case it mismatches
		chunk = list(chunk) + [0.] * ( npoints  - len(chunk) )
		chunk = scipy.array(chunk)
	hammed = chunk * scipy.signal.hamming(len(chunk))
	# compute magnitudes for each frequency band
	norm_amplitudes = pow(abs(scipy.fft(hammed)),2) / npoints
	return norm_amplitudes[ 1 : 1 + len(norm_amplitudes) / 2 ]


def get_amplitudes_dB(data,npoints):
	return [ calculate_rms_dB(m,max=1.0) for m in get_amplitudes(data,npoints) ]



fft_window_size = 256 # 128 below-nyquist bands, 64 after trimming the high end


lowpass_freq = 200
lowpass_order = 3
#b,a = scipy.signal.butter(lowpass_order,float(lowpass_freq)/(sampling_rate/2))
#firstminute = scipy.signal.lfilter(b,a,firstminute)


def get_first_frequency_analysis(filename):
	f = wave.open(filename,"r")
	sampling_rate = f.getframerate()
	channels = f.getnchannels()
	print "Frames: ",f.getnframes()

	firstminute = f.readframes(10*sampling_rate)
	firstminute = scipy.array(array.array("h",firstminute))
	firstminute = mixdown(firstminute,channels)
	first_audible_frame = find_first_nonsilence(firstminute,sampling_rate)
	if first_audible_frame is None:
		print "Failed to find frame with audible data, seeking to zero"
		f.setpos(0)
	else:
		print "Found first frame with audible data: %d, seeking to it"%first_audible_frame
		f.setpos(first_audible_frame)

	pointlist = []
	for x in xrange(4):
		offset = f.tell()
		audio_data = f.readframes(sampling_rate*10)
		stream = scipy.array(array.array("h",audio_data)) / 32768.
		stream = mixdown(stream,channels)
		points = get_amplitudes(stream,fft_window_size)
		# here I have up to the nyquist frequencies
		# I'll make sure to trim the upper half
		# since MP3s lose them
		# FIXME make me work with 22050 hz and other sampling freqs
		# for this we would need to scale the FFT points depending on the
		# source sampling frequency
		points = points[:len(points) / 2]
		pointlist.append(points)
		#points = [ (band,v) for band,v in enumerate(points[:len(points) / 2]) ]
		#pointlist.extend( [ (offset,band,v) for band,v in points ] )
	return pointlist


ps1 = get_first_frequency_analysis(sys.argv[1])
ps2 = get_first_frequency_analysis(sys.argv[2])

for n,row in enumerate(ps1):
	p1 = ps1[n]
	p2 = ps2[n]
	print n, numpy.corrcoef(p1,p2)[1][0]

#audiodevice = alsaaudio.PCM()
#audiodevice.setchannels(2)
#audiodevice.setformat(alsaaudio.PCM_FORMAT_S16_LE)
#audiodevice.setperiodsize(sampling_rate)

#cla()
pickle.dump([ps1,ps2],file("/tmp/data1","w"))

	##freq = numpy.fft.fftfreq(5)
	##print freq



#write an algorithm that combines the phase beat detector and a time-domain beat detector for best results, and write a program that uses it and writes the TBPM tag on mp3
#now that we know the fft code is solid, we can apply the bark scale to get the appropriate buckets from the FFT, so we can compare songs, thkn about the databse structure that will contain these 1 second 60 buckets of spectrum amplitude