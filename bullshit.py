#!/usr/bin/env python

import gdbm
import numpy
import audioprocessing as a
import cPickle as pickle
import sys
import array
import os
import sets
import cProfile

d = gdbm.open("data.gdbm")

signatures = dict([ (k, pickle.loads(d[k])[1]) for k in d.keys() ])

print "Cached.  Analyzing..."

cache = {}

def corr(k1,k2,s1,s2):
	global cache
	k1k2 = k1+k2
	k2k1 = k2+k1
	if k1k2 in cache:
		return cache[k1k2]
	if k2k1 in cache:
		return cache[k2k1]
	result = a.butterscotch_interference(s1,s2)
	cache[k1k2] = result
	return result
	

func = a.butterscotch_correlate_by_band

def corrs(key):
	keysign = signatures[key]
	filenames = signatures.keys()
	signs = signatures.values()
	profile = cProfile.Profile()
	profile.runctx('corrs = [ a.butterscotch_correlate_by_band(keysign,s) for s in signs ]',globals(),locals())
	corrs = [ func(keysign,s) for s in signs ]
	result = zip(filenames,corrs)
	return (profile,result)

def output(pr,threshold=0.9):
	l = pr[1]
	l.sort(reverse=True,key=lambda x:x[1])
	#m = l
	m = [ e for e in l if e[1] > threshold ]
	m = [ ( os.path.splitext(os.path.basename(e[0]))[0].split("(")[0] , e[1] ) for e in m ]
	if len(m) != len(l): m.append(l[len(m)])
	#filenames = sets.Set([ e[0].lower() for e in m])
	#diditwork = len(filenames) == 2
	#if not diditwork:
	for e in m: print "%s: %.05f"%(e[0][-55:],e[1])
	#print "Correlation worked? %s"%diditwork
	#print ""
	#return diditwork
	return pr[0]

