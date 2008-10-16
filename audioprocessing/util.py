#!/usr/bin/env python

import math
import numpy

# List processing primitives


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


# Audio processing primitives


def play(signal,rate=44100):
	import alsaaudio
	d = alsaaudio.PCM()
	d.setchannels(1)
	d.setformat(alsaaudio.PCM_FORMAT_S16_LE)
	#signal = signal[:rate]
	d.setperiodsize(rate*2)
	d.setrate(rate)
	for s in chunked(signal,rate*2):
		print "writing %s"%len(s)
		d.write(s)


__all__ = [
	'chunked',
	'in_pairs',
	'deltas',
	'log2_average',
	'play',
]