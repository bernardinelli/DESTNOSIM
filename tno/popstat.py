#import numpy as np 
import numba

from numba.pycc import CC

cc = CC('popstat')
# Uncomment the following line to print out the compilation steps
#cc.verbose = True

@cc.export('compute_arccut', 'f8(f8[:])')

@numba.jit(nopython=True)
def compute_arccut(times):
	'''
	Computes ARCCUT, the time between the first and last detection dropping one night of detection
	times must be in DAYS
	'''
	arccut = 0.
	t1 = min(times)
	t2 = max(times)
	t1a = t2
	t2a = t1

	n = len(times)

	for i in range(n):
	  if times[i] - t1 > 0.7 and times[i] < t1a:
	  	t1a = times[i]
	  if t2 - times[i] > 0.7 and times[i] > t2a:
	  	t2a = times[i]

	arccut = min(t2 - t1a, t2a - t1)

	return arccut

@cc.export('compute_nunique', 'i8(f8[:])')

@numba.jit(nopython=True)
def compute_nunique(times):
	'''
	Computes NUNIQUE, the number of unique nights in which we have observations from an object
	times must be in DAYS
	'''
	nunique = 0

	n = len(times)

	unique = True 

	for i in range(n):
		unique = True
		for j in range(0, i):
			if abs(times[j] - times[i]) < 0.1:
				unique = False 
		if unique:
			nunique += 1

	return nunique
	
if __name__ == "__main__":
    cc.compile()
