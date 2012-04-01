#!/usr/bin/env python

import audioprocessing
import numpy
import cPickle as pickle
import time


class Database:

	def __init__(self,filename):
		self.filename = filename
		try: self.fingerprints = pickle.load(file(self.filename+".fingerprints"))
		except (IOError,OSError),e: self.fingerprints = {}
		try: self.correlations = pickle.load(file(self.filename+".correlations"))
		except (IOError,OSError),e: self.correlations = {}

	def get_fingerprints(self):
		""" Returns:
		a list with (filename,Butterscotch fingerprint) as each row
		"""
		return self.fingerprints.items()

	def is_already_fingerprinted(self,filename):
		return filename in self.fingerprints

	def add_fingerprint(self,filename,fingerprint):
		""" Arguments:
		filename: string with the full, absolute path to the file
		fingerprint: Butterscotch fingerprint
		FIXME: make contentious for the insert IDs!
		"""
		self.fingerprints[filename] = fingerprint
		pickle.dump(self.fingerprints,file(self.filename+".fingerprints","w"),-1)

	def add_correlations(self,filename,*correlations):
		""" Arguments:
		filename: string with the full, absolute path to the file
		correlations: a list of (otherfilename,corrcoef)
		"""
		for otherfilename,corrstrength in correlations:
			self.correlations[filename,otherfilename] = corrstrength

		pickle.dump(self.correlations,file(self.filename+".correlations","w"),-1)


def get_next_song():
	playlist = "playlist.m3u"
	files = [ s.strip() for s in file(playlist).readlines() if s.strip() ]
	files = [ f for f in files if not f.startswith("#") ]
	for f in files: yield f

def obtain_fingerprint(filename):
	signature = audioprocessing.butterscotch(filename)
	signature = signature.halve_highest_freq()
	signature = signature.as_log_bands()
	return signature

import time
def timed(f):
	def x(*args,**kwargs):
		start = time.time()
		retval = f(*args,**kwargs)
		diff = time.time()  - start
		print "Function took %f seconds"%(diff)
		return retval
	return x

def run():
	database = Database("wasabitest")
	threshold = 0.7
	for filename in get_next_song():
		def loop():
			if database.is_already_fingerprinted(filename):
				print "Skipping %s"%filename
				return
			print "Working with file %s"%filename
			try: fingerprint = obtain_fingerprint(filename)
			except NotImplementedError,e:
				print "Skipping %s because: %s"%(filename,e)
				return
			print fingerprint
			all_fingerprints = database.get_fingerprints()
			print "Correlating against %d fingerprints..."%len(all_fingerprints),
			starttime = time.time()
			corrs = []
			for otherfn,other in all_fingerprints:
				corr = fingerprint.correlate(other)
				if corr > threshold: corrs.append( (otherfn,corr) )
			print "process yielded %d correlations in %f seconds"%(len(corrs),time.time() - starttime)
			database.add_fingerprint(filename,fingerprint)
			database.add_correlations(filename,*corrs)
			print ""
		#loop = timed(loop)
		loop()

if __name__ == "__main__":
	run()

"""

nuestra base de datos debe contener series de esta entidad:

[fingerprint                               ][  sample onset         ]
[960 bytes - 32 bit fp * 30 * 8][2 bytes unsigned int]

tambien debe tener un mapping filename -> ID

finalmente una lista de correlaciones (id1,id2)

This program starts,and it connects the following actors in a pipeline:

we need something that:
	- produces songs to scan
	- consumes songs to scan and produces fingerprints
	- consumes fingerprints and produces correlation lists

and also another thing that gathers both results and puts them into the database





def getDuplicates(filename):
	id = database.getId(filename)
	d <- (id,corrcoef,...) = database.getCorrelations(id)
	f = [ (database.getFilename(id),c) for id,c in d ]

"""



#a.bands = a.bands.astype("float16")
#b.bands = b.bands.astype("float16")
#print "Correlation float16: %f"%(a.correlate(b))
#a.bands = a.bands.astype("float8")
#b.bands = b.bands.astype("float8")
#print "Correlation float8: %f"%(a.correlate(b))

"""

algoritmo
leer playlist
descartar las que ya fueron analizadas
para cada una de las que quedan:
	analizar
	correlacionarla completamente con todas las analizadas
	almacenar las correlaciones positivas > 0.8 en la super pytables
	almacenar la signature con el path name (y el numero serial) en la DB
listo!

antes de hacer esto
2. guardar los analisis como numeros flotantes de 32 bits
3. verificar que precision pierdo si guardo la correlacion como un entero de 1 byte


y de ahi si escribir el core

maybe I can have three databases,
	one where positive corrs are stored
	one where negative corrs are stored
	one storing the strength of the corrs
	the goal is to quickly ascertain which ones correlate with a simple query
	maybe if I can index corr strength we don't need that
	but this database would actually make my day as I can store only low amounts of correlations and use that to analyze
	and if this works out I can use very few bits to store corrs
algoritmo:
song = get the next song ()
compute butterscotch signature
save (path,signature) to database
correlate that signature with all other songs' signatures, save the results as (signature1,signature2,correlation)

use pytables to store the data

amarok plasmoid for new amarok to plot the frequency response of a song, the first 30 seconds after the first minute.

idea for smear algorithm
if a track is bleeped or kikikjed (ripped from a scratched disc) we can maybe average the blocks before and after the mistake to "fix" it without affecting the signature so that we can use even damaged songs to discover duplicates, and maybe even flag the song as damaged!


I store only the positive correlations.  I can also store them as 1 byte integers representing reduced ranges from 0.8 to 1.0 which are the significant correlations.

I can vectorize the correlations that I need to do, try and make them fit into L1 cache so they take little time.  If I have a buncha signatures, I may be able to perform the operations using SIMD instructions if I row them all up.



"""
