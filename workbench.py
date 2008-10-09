#!/usr/bin/env python

import gdbm
import numpy
import audioprocessing
import cPickle as pickle
import sys
import os
import sets
import pylab as p
import matplotlib.axes3d as p3
import time
import threading
import cProfile


def justfn(m): return os.path.splitext(os.path.basename(m)[0])


signatures = None
corrcache = {}
def loadsigns(f):
	global signatures
	global corrcache
	d = gdbm.open(f)
	signatures = dict([ (k, pickle.loads(d[k])) for k in d.keys() ])
	corrcache = {}
	d.close()
	del d


def corr(one,two):
	global corrcache
	if (one,two) in corrcache: result = corrcache[(one,two)]
	elif (two,one) in corrcache: result = corrcache[(two,one)]
	else:
		result = one.correlate(two)
		corrcache[(one,two)] = result
	return result


def loadorgencorrs(filename):
	global signatures
	loadsigns(filename)
	try: corrs = pickle.load(file("%s.corrs"%filename))
	except (IOError,OSError):
		corrs = list ( corrall (signatures) )
		pickle.dump(corrs,file("%s.corrs"%filename,"w"))
	return corrs


def corrall(signatures):
	for x in range(len(signatures)):
		print "X is now %d"%x
		starttime = time.time()
		for y in range(len(signatures)):
			if x == y: continue # do not plot that 1.0 correlation!
			k1 = signatures.keys()[x] ; k2 = signatures.keys()[y]
			v1 = signatures[k1] ; v2 = signatures[k2]
			c = corr(v1,v2)
			yield (k1,k2,x,y,c)
		print "Time for sweep %d: %f"%(x,time.time() - starttime)



#should_stop = False
#def graphcorrs(async = False):
	#global signatures
	#ss = signatures

	#fig=p.figure()
	#ax = p3.Axes3D(fig)
	#ax.set_xlabel('Index')
	#ax.set_ylabel('Index')
	#ax.set_zlabel('Correlation')
	#if not async: p.show()
	#def inc_corr():
		#for x in range(len(ss)):
			#for y in range(len(ss)):
				#if x == y: continue # do not plot that 1.0 correlation!
				#k1 = ss.keys()[x] ; k2 = ss.keys()[y]
				#v1 = ss[k1] ; v2 = ss[k2]
				#c = corr(v1,v2)
				#if c < 0.8: continue
				#sys.stdout.write("\n%s\n%s\n%s\n\n"%(
								#k1[-64:],k2[-64:],c))
				#sys.stdout.flush()
				#def justfn(m):
					#return os.path.splitext(
						#os.path.basename(m)
					#)[0].split("(")[0]
				#fn1 = justfn(k1) ; fn2 = justfn(k2)
				#yield (x,y,c,fn1,fn2)
	
	#def plotinc():
		#global should_stop
		#toplot = []
		#for x,y,z,fn1,fn2 in inc_corr():
			#if should_stop:
				#should_stop = False
				#return
			#if fn1 == fn2:
				#ax.scatter3D([x],[y],[z],s=12,c="b")
			#else:
				#ax.scatter3D([x],[y],[z],s=3,c="g")
				#ax.annotate("%s\n%s"%(fn1,fn2),(x,y))
			#time.sleep(0.2)
	
	#t = threading.Thread(target=plotinc)
	#t.start()

#def stopgraph():
	#global should_stop
	#should_stop = True


#cache = {}

#def corr(k1,k2,s1,s2):
	#global cache
	#k1k2 = k1+k2
	#k2k1 = k2+k1
	#if k1k2 in cache:
		#return cache[k1k2]
	#if k2k1 in cache:
		#return cache[k2k1]
	#result = a.butterscotch_interference(s1,s2)
	#cache[k1k2] = result
	#return result
	

#func = a.butterscotch_correlate_by_band

#def corrs(key):
	#keysign = signatures[key]
	#filenames = signatures.keys()
	#signs = signatures.values()
	#profile = cProfile.Profile()
	#profile.runctx('corrs = [ a.butterscotch_correlate_by_band(keysign,s) for s in signs ]',globals(),locals())
	#corrs = [ func(keysign,s) for s in signs ]
	#result = zip(filenames,corrs)
	#return (profile,result)

#def output(pr,threshold=0.9):
	#l = pr[1]
	#l.sort(reverse=True,key=lambda x:x[1])
	##m = l
	#m = [ e for e in l if e[1] > threshold ]
	#m = [ ( os.path.splitext(os.path.basename(e[0]))[0].split("(")[0] , e[1] ) for e in m ]
	#if len(m) != len(l): m.append(l[len(m)])
	##filenames = sets.Set([ e[0].lower() for e in m])
	##diditwork = len(filenames) == 2
	##if not diditwork:
	##for e in m: print "%s: %.05f"%(e[0][-55:],e[1])
	##print "Correlation worked? %s"%diditwork
	##print ""
	##return diditwork
	##return pr[0]

