import numpy as np 

'''
Solar system constants
Information from https://github.com/gbernstein/gbutil/blob/master/include/AstronomicalConstants.h
'''
# ----------------------------------------------------------------------------------------------------------------
SunGM 		=	 4.*np.pi*np.pi/1.000037773533  #solar gravitation
MercuryGM 	= 	6.55371264e-06
VenusGM 	=   9.66331433e-05
EarthMoonGM =	1.20026937e-04
MarsGM 		=   1.27397978e-05;
JupiterGM   =	3.76844407e-02 + 7.80e-6
SaturnGM 	=  1.12830982e-02 + 2.79e-6
UranusGM 	=  1.72348553e-03 + 0.18e-6
NeptuneGM 	= 2.03318556e-03 + 0.43e-6

SolarSystemGM = SunGM + MercuryGM + VenusGM + EarthMoonGM + MarsGM + JupiterGM + SaturnGM + UranusGM + NeptuneGM

EclipticInclination = 23.43928 * np.pi/180
# ----------------------------------------------------------------------------------------------------------------

def R(omega, Omega, i):
	'''
	Computes the rotation matrix :math:`R = R(\\Omega, \\omega, i)` that maps from the plane of the ellipse to 3D space aligned with the ecliptic plane
	'''
	cO = np.cos(np.pi*Omega/180)
	sO = np.sin(np.pi*Omega/180)
	co = np.cos(np.pi*omega/180)
	so = np.sin(np.pi*omega/180)
	ci = np.cos(np.pi*i/180)
	si = np.sin(np.pi*i/180)

	R = np.matrix([[cO * co - sO * so * ci, - cO * so - sO * co * ci, sO * si],[sO * co + cO * so * ci, -sO * so + cO * co * ci, -cO * si],[si * so, si*co, ci]])

	return R

def cartesian_to_keplerian(xv, epoch, helio = False, ecliptic = False):
	'''
	Goes from ecliptic oriented state vector (i.e. cartesian representation) to orbital elements.
	See appendix B of Bernstein & Khushalani 2000, for example
	'''
	mu = SunGM if helio else SolarSystemGM

	if not ecliptic:
		cosEcl = np.cos(EclipticInclination)
		sinEcl = np.sin(EclipticInclination)
		xv[:,1], xv[:,2] = cosEcl * xv[:,1] + sinEcl * xv[:,2], -sinEcl * xv[:,1] + cosEcl * xv[:,2]
		xv[:,4], xv[:,5] = cosEcl * xv[:,4] + sinEcl * xv[:,5], -sinEcl * xv[:,4] + cosEcl * xv[:,5]

	x = np.sqrt(xv[:,0]*xv[:,0] + xv[:,1]*xv[:,1]+ xv[:,2]*xv[:,2])
	vsq_mu = (xv[:,3]**2 + xv[:,4]**2 + xv[:,5]**2)/mu

	inv_a = 2./x - vsq_mu
	a = 1./inv_a

	x_dot_v = (xv[:,0] * xv[:,3] + xv[:,1] * xv[:,4] + xv[:,2] * xv[:,5])
	pref = (vsq_mu - 1./x)
	e_vec = (pref * xv[:,0:3].T -  x_dot_v * xv[:,3:6].T/mu).T
	e = np.sqrt(e_vec[:,0] * e_vec[:,0] + e_vec[:,1] * e_vec[:,1] + e_vec[:,2] * e_vec[:,2])
	h_vec = np.cross(xv[:,0:3], xv[:,3:6])
	n_vec = np.cross(np.array([0,0,1]), h_vec)
	h = np.sqrt(h_vec[:,0] * h_vec[:,0] + h_vec[:,1] * h_vec[:,1] + h_vec[:,2] * h_vec[:,2])
	n = np.sqrt(n_vec[:,0] * n_vec[:,0] + n_vec[:,1] * n_vec[:,1] + n_vec[:,2] * n_vec[:,2])
	cos_i = h_vec[:,2]/h 
	cosOmega = n_vec[:,0]/n 
	cosomega = (n_vec[:,0] * e_vec[:,0] + n_vec[:,1] * e_vec[:,1] + n_vec[:,2] * e_vec[:,2])/(n*e)

	i = np.arccos(cos_i) * 180./np.pi
	Omega = np.arccos(cosOmega) * 180./np.pi
	omega = np.arccos(cosomega) * 180./np.pi
	omega[np.where(e_vec[:,2] < 0)] = 360 - omega[np.where(e_vec[:,2] < 0)]

	p = h*h/mu
	b = a * np.sqrt(1 - e*e)

	xbar = (p - x)/e 
	ybar = x_dot_v/e * np.sqrt(p/mu)

	E = np.arctan2(ybar/b, xbar/a + e)

	M = E - e*np.sin(E)

	T_p = epoch - M * np.sqrt(a**3/mu)

	aei = np.zeros_like(xv)

	aei[:,0] = a
	aei[:,1] = e 
	aei[:,2] = i 
	aei[:,3] = Omega
	aei[:,4] = omega
	aei[:,5] = T_p

	return aei




