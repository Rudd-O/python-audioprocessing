#!/usr/bin/env python

import wave
import numpy
import subprocess
from audioprocessing.signal import calculate_rms_dB
import threading

# Stream filters


class WaveStream:
	def __init__(self,wave):
		self.wave = wave

	def close(self):
		try: self.wave.close()
		except Exception: pass
		del self.wave

	def readframes(self,n):
		#import time
		#start = time.time()
		#assert n > 0
		#ret = self.wave.readframes(n)
		#endtime = time.time()
		#if n > 100: print "Readframes %s %s: %s"%(n,str(self),endtime-start)
		return self.wave.readframes(n)

	def getcomptype(self): return self.wave.getcomptype()
	def getsampwidth(self): return self.wave.getsampwidth()
	def getnchannels(self): return self.wave.getnchannels()
	def getframerate(self): return self.wave.getframerate()
	def getnframes(self): return self.wave.getnframes()
	def tell(self): return self.wave.tell()
	def __del__(self):
		try: self.close()
		except Exception: pass


class NumPyStream(WaveStream):
	"""Given a bytestream wave object, transforms the data into NumPy."""

	def __init__(self,wave):
		if wave.getcomptype() != "NONE":
			raise NotImplementedError, "compression type %r not supported"%wave.getcomptype()
		if wave.getsampwidth() != 2:
			raise NotImplementedError, "sample width %d bits not supported"%wave.getsampwidth()*8
		WaveStream.__init__(self,wave)

	def readframes(self,n):
		return numpy.core.fromstring(
			self.wave.readframes(n),
			dtype=numpy.dtype("int16")
		)


class FloatStream(WaveStream):
	"""Given a NumPy stream of integers, transform the data into floating point
	audio."""

	def readframes(self,n):
		return self.wave.readframes(n) / 32768.


class MonoStream(WaveStream):
	"""Given a multichannel NumPy stream, transform the data into mono"""

	def __init__(self,wave):
		WaveStream.__init__(self,wave)
		self.nchannels = self.wave.getnchannels()

	def readframes(self,n):
		shape = (-1,self.nchannels)
		stereosignal = self.wave.readframes(n)
		monosignal = stereosignal.reshape(shape).mean(1)
		return monosignal

	def getnchannels(self):
		return 1


class NoAudioOnset(Exception): pass
class AudioOnsetDiscarder(WaveStream):

	def __init__(self,wave):

		assert wave.getnchannels() == 1
		WaveStream.__init__(self,wave)
		
		chunksize = self.getframerate() / 2000
		threshold = 5

		chunk = self.wave.readframes(chunksize)
		position = len(chunk)
		rms = calculate_rms_dB(chunk)

		while True:
			newchunk = self.wave.readframes(chunksize)
			position = position + len(newchunk)
			newrms = calculate_rms_dB(newchunk)
			if newrms <= -96.0: continue
			if newrms - rms > threshold:
				self.audio_onset_sample = position - len(chunk) - len(newchunk)
				self.buf = numpy.hstack( [ chunk, newchunk ] )
				return
			chunk = newchunk
			rms = newrms
		
		raise NoAudioOnset, "No audio onset could be found"

	def readframes(self,n):
		if self.buf is None or len(self.buf) == 0:
			return self.wave.readframes(n)
		if len(self.buf) >= n:
			portion,self.buf = self.buf[:n],self.buf[n:]
			return portion
		remainder = self.wave.readframes(n-len(self.buf))
		data = numpy.hstack( [ self.buf , remainder ] )
		self.buf = None
		return data


# Decoders


class MP3Decoder(WaveStream):

	def __init__(self,mp3file,numframes=None,firstframe=None):
		if numframes is not None and ( type(numframes) is not int or numframes < 1):
			raise TypeError, "numframes must be a positive int, not %s"%numframes
		if firstframe is not None and ( type(firstframe) is not int or numframes < 1):
			raise TypeError, "firstframe must be a positive int, not %s"%firstframe

		command = ["mpg321","--quiet"]
		if firstframe is not None: command += ["-k",str(firstframe)]
		if numframes is not None:
			if firstframe is None: command += ["-n",str(numframes)]
			else: command += ["-n",str(numframes+firstframe)]

		command += ["-w","/dev/stdout",mp3file]
		bufsize = 0
		self.popen = subprocess.Popen(command,stdout=subprocess.PIPE,bufsize=bufsize)
		self.pipe = self.popen.stdout
		WaveStream.__init__(self,wave.open(self.pipe))

	def close(self):
		WaveStream.close(self)
		try: self.pipe.close()
		except Exception: pass
		threading.Thread(target=self.popen.wait).start()
		del self.pipe
		del self.popen


class FLACDecoder(WaveStream):

	def __init__(self,mp3file):
		command = ["flac","-d","-c","--silent"]
		command += [mp3file]
		bufsize = 0
		self.popen = subprocess.Popen(command,stdout=subprocess.PIPE,bufsize=bufsize)
		self.pipe = self.popen.stdout
		WaveStream.__init__(self,wave.open(self.pipe))

	def close(self):
		WaveStream.close(self)
		try: self.pipe.close()
		except Exception: pass
		threading.Thread(target=self.popen.wait).start()
		del self.pipe
		del self.popen

def decode(filename):
	"""Auto-chooses the appropriate decoder for the file name passed,
	returns a WaveStream object with it."""
	# FIXME: make it detect based on content, not on extension
	if filename.lower().endswith(".mp3"): return MP3Decoder(filename)
	if filename.lower().endswith(".flac"): return FLACDecoder(filename)
	return WaveStream(wave.open(file(filename)))


if __name__ == "__main__":

	import sys

	print "\nMP3Decoder"
	print sys.argv[1]
	o = AudioOnsetDiscarder(MonoStream(FloatStream(NumPyStream(decode(sys.argv[1])))))

	print o.audio_onset_sample
	data = o.readframes(44100*3)
	print len(data)
	rms = calculate_rms_dB(data,1.0)
	print rms
	#print data[512:768]*32768

	from audioprocessing.util import play
	data = (data * 32768).astype("int16").tostring()
	play(data)


__all__ = [
	'WaveStream',
	'NumPyStream',
	'FloatStream',
	'MonoStream',
	'AudioOnsetDiscarder',
	'NoAudioOnset',
	'MP3Decoder',
	'FLACDecoder',
	'decode',
]
