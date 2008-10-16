#!/usr/bin/env python

import subprocess

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
