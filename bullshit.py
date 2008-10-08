#!/usr/bin/env python

import gdbm
import numpy
import audioprocessing as a
import cPickle as pickle
import sys
import array
import os
import sets

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
	result = a.butterscotch_correlate_by_band(s1,s2)
	cache[k1k2] = result
	return result
	

def correlate_with_dict(d,key):
	s1 = d[key]
	result = [ (k,corr(key,k,s1,s2)) for k,s2 in d.items() ]
	result.sort(reverse=True,key=lambda x:x[1])
	return result

def output(l,threshold):
	m = [ e for e in l if e[1] > threshold ]
	m.append(l[len(m)])
	m = [ ( os.path.splitext(os.path.basename(e[0]))[0].split("(")[0] , e[1] ) for e in m ]
	filenames = sets.Set([ e[0].lower() for e in m])
	diditwork = len(filenames) == 2
	if not diditwork:
		for e in m: print "%s: %.05f"%(e[0][-55:],e[1])
		print "Correlation worked? %s"%diditwork
		print ""
	return diditwork

#f = a.butterscotch_correlate_by_band
#l = []
#for k2,corr in produce_correlations(sys.argv[1],f):
	#if not len(corr): continue
	#l.append((k2,numpy.average(corr),max(corr),min(corr)))

#print "By average:"
#l.sort(key=lambda x:x[1])
#l.reverse()
#print "\n".join([ "%s: %.04f"%(t[0][-55:],t[1]) for t in l[1:6] ])
#print ""

#print "By max:"
#l.sort(key=lambda x:x[2])
#l.reverse()
#print "\n".join([ "%s: %.04f"%(t[0][-55:],t[2]) for t in l[1:6] ])
#print ""

#print "By min:"
#l.sort(key=lambda x:x[3])
#l.reverse()
#print "\n".join([ "%s: %.04f"%(t[0][-55:],t[3]) for t in l[1:6] ])
#print ""

