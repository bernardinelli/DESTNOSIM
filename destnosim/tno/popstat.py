import numpy as np 
import numba

from numba.typed import List

# Uncomment the following line to print out the compilation steps
#cc.verbose = True

#@cc.export('compute_arccut', 'f8(f8[:])')

@numba.jit('f8(f8[:])', nopython=True,)
def compute_arccut(times):
	'''
	Computes ARCCUT, the time between the first and last detection dropping one night of detection
	Arguments:
	- times: list of times, must be in DAYS
	'''
	arccut = 0.
	t1 = np.min(times)
	t2 = np.max(times)
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

#@cc.export('compute_nunique', 'i8(f8[:])')

@numba.jit('i8(f8[:])',nopython=True,)
def compute_nunique(times):
	'''
	Computes NUNIQUE, the number of unique nights in which we have observations from an object

	Arguments:
	- times: list of times, must be in DAYS
	'''
	nunique = 0

	n = len(times)

	unique = True 

	for i in range(n):
		unique = True
		for j in range(0, i):
			if abs(times[j] - times[i]) < 0.4:
				unique = False 
		if unique:
			nunique += 1

	return nunique

#@cc.export('compute_triplet', 'b1(f8[:], f8)')

@numba.jit('b1(f8[:], f8)',nopython=True,)
def compute_triplet(times, thresh):
	'''
	Computes the time difference between triplets, returns if a triplet is formed

	Arguments:
	- times: list of times, must be in DAYS
	- thresh: threshold time for pairs
	'''
	first_pair =  999.
	second_pair = 999.
	det = List()

	det.append(times[0])

	n = len(times)

	for i in range(n-1):
		if times[i + 1] - times[i] > 0.4:
			det.append(times[i+1])
	
	n = len(det)
	for i in range(1,n-1):
		if det[i + 1] - det[i] < first_pair and det[i] - det[i-1] < second_pair:
			first_pair = det[i + 1] - det[i]
			second_pair = det[i] - det[i-1]

	if first_pair < thresh and second_pair < thresh:
		return True
	else:
		return False

#@cc.export('find_triplet_time', 'f8[:](f8[:])')

@numba.jit('f8[:](f8[:])',nopython=True,)
def find_triplet_time(times):
	'''
	Computes the minimum triplet time for both pairs of an object

	Arguments:
	- times: list of times, must be in DAYS

	'''
	first_pair =  999.
	second_pair = 999.
	det = List()

	det.append(times[0])

	n = len(times)

	for i in range(n-1):
		if times[i + 1] - times[i] > 0.4:
			det.append(times[i+1])
	
	n = len(det)
	for i in range(1,n-1):
		if det[i + 1] - det[i] < first_pair and det[i] - det[i-1] < second_pair:
			first_pair = det[i + 1] - det[i]
			second_pair = det[i] - det[i-1]

	return np.array([first_pair, second_pair])

