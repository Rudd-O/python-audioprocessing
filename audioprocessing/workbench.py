#!/usr/bin/env python

import pylab as p
import mpl_toolkits.mplot3d.axes3d as p3
import gdbm
import numpy
import cPickle as pickle
import os
import time
from math import exp
import subprocess
import glob


def plot_butterscotch_signature(self,title=None,show = True):

		fig=p.figure()
		ax = p3.Axes3D(fig,azim=0,elev=15)

		if title: ax.text(0,0,0,title)

		ax.set_xlabel('Time (s)')
		if self.linear_bands: ax.set_ylabel('Frequency (%d linear bands)'%len(self.bands))
		else: ax.set_ylabel('Frequency (%d logarithmic bands)'%len(self.bands))
		if self.linear_intensities: ax.set_zlabel('Intensity (raw)')
		else: ax.set_zlabel('Intensity (dB)')

		zmin = self.bands.min()
		fbands = list(self.freq_centerpoints())
		for n,block in enumerate(self.blocks):
			zs = numpy.array(   list(block)  )     # + [zmin,zmin,block[0]]   )
			xs = numpy.array(   [n * self.secs_per_block ] * len(block)   )     #* ( len(block) + 3 )   )
			ys = numpy.array(   fbands   )         # +  [fbands[-1],fbands[0],fbands[0]]   )
			ax.plot3D(xs, ys , zs)

		if show: p.show()
		return fig


def just_filename(m):
	return os.path.splitext(os.path.basename(m))[0].lower()


databasename = None
signatures = None
corrcache = None
corrs = None
def load_batch(dbn):
	global databasename
	global signatures
	global corrcache
	global corrs

	databasename = dbn
	signatures = {}
	corrcache = {}
	corrs = []

	d = gdbm.open(dbn)
	signatures = dict([ (k, pickle.loads(d[k])) for k in d.keys() ])
	print "Loaded %d signatures from disk"%len(signatures)
	d.close()

	try:
		corrs = pickle.load(file(dbn+".corrcache"))
		corrs = [ (x,y,f1.lower(),f2.lower(),c) for x,y,f1,f2,c in corrs ]
		print "Loaded %d correlations from disk"%len(corrs)
	except Exception:
		print "Correlations were not loaded.  Use batchcorrelate() to generate them."


def corr_two(one,two):
	global corrcache
	if (one,two) in corrcache: result = corrcache[(one,two)]
	elif (two,one) in corrcache: result = corrcache[(two,one)]
	else:
		result = one.correlate(two)
		corrcache[(one,two)] = result
	return result


def correlate_batch(printsweep=True):
	global signatures
	global corrs
	global databasename
	corrs = []

	def corrall():
		xtime = time.time()
		blah = [ ( n, just_filename(k), v ) for n,(k,v) in enumerate(signatures.iteritems()) ]
		for x,f1,v1 in blah:
			if printsweep: starttime = time.time()
			for y,f2,v2 in blah:
				c = corr_two(v1,v2)
				yield (x,y,f1,f2,c)
			if printsweep: print "Time for sweep %d: %f"%(x,time.time() - starttime)
		alltime = time.time() - xtime
		avg = alltime / len(signatures) / len(signatures)
		print "Total time: %f -- average %f"%(alltime,avg)

	for t in corrall(): corrs.append(t)

	pickle.dump(corrs,file(databasename+".corrcache","w"),protocol=-1)


def vlc(args=None):
	global toexcept
	if args is None: args = toexcept
	subprocess.check_call(["vlc"] + list(args),stdout=file("/dev/null","w"),stderr=subprocess.STDOUT)


def mpg321(args,skip=0,frames=40):
	frames = skip + frames
	for a in args:
		print a
		subprocess.check_call(["mpg321","-k",str(skip),"-n",str(frames),"-o","alsa",a],stdout=file("/dev/null","w"),stderr=subprocess.STDOUT)


def plot_correlations(plot=True):
	global corrs
	global signatures
	global databasename

	# filter points to remove mirror image and to remove very low correlations
	def size(corr):
		s = int(exp(corr*5)/3)
		if s < 1: s = 1
		return s
	toplot = [ (x,y,z,size(z),f1==f2,f1,f2) for x,y,f1,f2,z in corrs if z >= 0.2 and y >= x and y != x ]

	signkeys = signatures.keys()
	soundalikes = pickle.load(file("soundalikes.pickle"))

	triangulitos = []
	bolitas = []
	for c in toplot:
		f1 = signkeys[c[0]] ; f2 = signkeys[c[1]]
		triangulito = False
		for soundalike in soundalikes:
			if f1 in soundalike and f2 in soundalike:
				triangulito = True
				break
		if triangulito: triangulitos.append(c)
		else: bolitas.append(c)

	assert len(triangulitos) + len(bolitas) == len(toplot)

	strength = lambda m:m[2]
	triangulitos.sort(key=strength)
	bolitas.sort(key=strength)
	triangulitos.reverse()
	bolitas.reverse()

	print "Lowest correlated soundalikes:"
	for x in triangulitos[-5:]: print x

	print "\nHighest correlated nonsoundalikes:"
	for x in bolitas [:5]: print x
	print ""

	print "Gap: %f\n"% ( triangulitos[-1][2] - bolitas[0][2] )

	# trim many low correlations!
	bolitas = bolitas[:2000]

	if plot:
		print "Plotting %d soundalikes and %d nonsoundalikes"%(len(triangulitos),len(bolitas))
		
		fig=p.figure()
		ax = p3.Axes3D(fig)
		ax.set_xlabel('X')
		ax.set_ylabel('Y')
		ax.set_zlabel('Correlation strength\n%s'%databasename)
		xs,ys,zs,sizes,fms,fn1s,fn2s = zip(*bolitas)
		ax.scatter3D(xs, ys , zs, s=sizes)
		xs,ys,zs,sizes,fms,fn1s,fn2s = zip(*triangulitos)
		ax.scatter3D(xs, ys , zs, s=100,marker='^',c='r')


def plot_databases(globspec="*.gdbm",*args,**kwargs):
	global corrs
	for x in glob.glob(globspec):
		try: load_batch(x)
		except Exception,e:
			print "Skipping %s: %s"%(x,str(e))
			continue
		if not corrs: correlate_batch(False)
		print  "\nShowing X: %s"%x
		plot_correlations(*args,**kwargs)

