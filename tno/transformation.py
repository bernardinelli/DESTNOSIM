from __future__ import print_function
import numpy as np 
from numba import vectorize

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

	R = np.array([[cO * co - sO * so * ci, - cO * so - sO * co * ci, sO * si],
				[sO * co + cO * so * ci, -sO * so + cO * co * ci, -cO * si],
				[si * so, si*co, ci]])


	if np.isscalar(omega):
		return R
	else:
		return np.transpose(R,(2,0,1))

def cartesian_to_keplerian(cartesian, epoch, helio = False, ecliptic = False):
	'''
	Goes from ecliptic oriented state vector (i.e. cartesian representation) to orbital elements.
	See appendix B of Bernstein & Khushalani 2000, for example
	'''
	mu = SunGM if helio else SolarSystemGM

	xv = np.zeros_like(cartesian)
	if not ecliptic:
		cosEcl = np.cos(EclipticInclination)
		sinEcl = np.sin(EclipticInclination)
		xv[:,0] = cartesian[:,0]
		xv[:,3] = cartesian[:,3]
		xv[:,1], xv[:,2] = cosEcl * cartesian[:,1] + sinEcl * cartesian[:,2], -sinEcl * cartesian[:,1] + cosEcl * cartesian[:,2]
		xv[:,4], xv[:,5] = cosEcl * cartesian[:,4] + sinEcl * cartesian[:,5], -sinEcl * cartesian[:,4] + cosEcl * cartesian[:,5]
	else:
		xv = cartesian


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
	Omega[np.where(n_vec[:,1] < 0)] = 360 - Omega[np.where(n_vec[:,1] < 0)]
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

def q(E, a, e):
        '''
        Computes

        :math:`\\mathbf{q} = \(a(\\cos(E) - e), a\\sqrt{1-e^2}\\sin(E),0 \)`,

        the coordinate vector on the plane of the ellipse
        '''     
        q1 = a*(np.cos(E)-e)
        q2 = a*np.sqrt(1-e**2)*np.sin(E)

        if np.isscalar(a):
        	return np.array([q1,q2,0])
        else:
        	return np.array([q1,q2,np.zeros_like(a)])

def dqdt(E, a, e, mu):
        '''
        Computes

        :math:`\\frac{\\mathrm{d}\\mathbf{q}}{\\mathrm{d} t} = \(- \\frac{n a \\sin E}{1 - e \\cos E}, \\frac{n a \\sqrt{1-e^2} \\cos E}{1 - e \\cos E},0 \)`,

        the velocity on the plane of the ellipse, where :math:`n = \\sqrt{G (M + m)}a^{-3\2} \\aprox \\sqrt{GM}a^{-3/2} = 2 \\pi a^{-3/2}` in our units and heliocentric coordinates.
        '''     
        n = np.sqrt(mu)/np.power(a,3./2)
        den = 1 - e*np.cos(E)
        q1 = -n*a*np.sin(E)/den
        q2 = n*a*np.sqrt(1-e**2)*np.cos(E)/den
        if np.isscalar(a):
        	return np.array([q1,q2,0])
        else:
        	return np.array([q1,q2,np.zeros_like(a)])


@vectorize(["float32(float32,float32)", "float64(float64,float64)"],nopython=True)
def solve_anomaly(e, M0):
	'''
	Super fast way of computing the true anomaly. Uses numba vectorization
	'''
	sol = M0
	delta = 0.0
	ones = 1.0
	for i in range(1000):
		delta = (M0 - (sol - e * np.sin(sol)))/(ones - e*np.cos(sol))
		sol += delta
	return sol

def rotate_to_ecliptic(xv, inverse = False):
	'''
	Rotates a 6D element vector between the ecliptic and equatorial frames
	'''
	cosEcl = np.cos(EclipticInclination)
	sinEcl = -np.sin(EclipticInclination)
	if inverse:
		sinEcl = - sinEcl
	xv[:,1], xv[:,2] = cosEcl * xv[:,1] + sinEcl * xv[:,2], -sinEcl * xv[:,1] + cosEcl * xv[:,2]
	xv[:,4], xv[:,5] = cosEcl * xv[:,4] + sinEcl * xv[:,5], -sinEcl * xv[:,4] + cosEcl * xv[:,5]

	return xv


def keplerian_to_cartesian(keplerian, epoch, helio = False, ecliptic = False):
	'''
	Goes from Keplerian elements to ecliptic or equatorial state vectors
	See chapter 1 of Modern Celestial Mechanics - Morbidelli for example
	'''
	mu = SunGM if helio else SolarSystemGM


	M0 = np.array((epoch - keplerian[:,5])*(np.sqrt(mu)/np.power(keplerian[:,0],3./2)))
	#print(M0)


	E = solve_anomaly(np.array(keplerian[:,1]), M0)
	#return M0, E, keplerian[:,1]

	q_vec = q(E, keplerian[:,0], keplerian[:,1])

	Rot = R(keplerian[:,4], keplerian[:,3], keplerian[:,2])

	x = np.einsum('...ij,j...', Rot, q_vec)

	dqdt_vec = dqdt(E, keplerian[:,0], keplerian[:,1], mu)

	v = np.einsum('...ij,j...', Rot, dqdt_vec)

	xv = np.zeros_like(keplerian)

	xv[:,0:3] = x
	xv[:,3:] = v

	if not ecliptic:
		xv = rotate_to_ecliptic(xv)
	return xv



def dist_to_point(elements, epoch, element_type, point, helio = False, ecliptic = False):
	'''
	Computes the distance for all objects from the 3d vector point. Computes coordinate changes as needed
	'''
	if element_type == 'keplerian':
		xv = keplerian_to_cartesian(elements, epoch, helio, ecliptic)
	elif element_type == 'cartesian':
		xv = elements
	else:
		raise ValueError("Element type must either be keplerian or cartesian!")

	r = np.sqrt((xv[:,0] - point[0])**2 + (xv[:,1] - point[1])**2 + (xv[:,2] - point[2])**2)

	return r

def bary_to_helio(elements, element_type, epoch, sun_coordinates, ecliptic = False):
	'''
	Converts a set of barycentric elements (either Keplerian or cartesian) to heliocentric, returning them in the same fashion
	sun_coordinates must be a 6D vector in (AU,AU/yr)
	'''
	if element_type == 'keplerian':
		xv = keplerian_to_cartesian(elements, epoch, False, ecliptic)
	elif element_type == 'cartesian':
		xv = elements
	else:
		raise ValueError("Element type must either be keplerian or cartesian!")

	
	new_elements = xv[:,] - sun_coordinates

	if element_type == 'keplerian':
		new_elements = cartesian_to_keplerian(new_elements, epoch, True, ecliptic)

	return new_elements

def helio_to_bary(elements, element_type, epoch, bary_coordinates, ecliptic = False):
	'''
	Converts a set of heliocentric elements (either Keplerian or cartesian) to barycentric, returning them in the same fashion
	bary_coordinates must be a 6D vector in (AU,AU/yr)
	'''
	if element_type == 'keplerian':
		xv = keplerian_to_cartesian(elements, epoch, True, ecliptic)
	elif element_type == 'cartesian':
		xv = elements
	else:
		raise ValueError("Element type must either be keplerian or cartesian!")

	
	new_elements = xv[:,] - bary_coordinates

	if element_type == 'keplerian':
		new_elements = cartesian_to_keplerian(new_elements, epoch, False, ecliptic)

	return new_elements


def table_to_matrix(table):
	'''
	Using a table of covariance matrix elements, in cartesian space, constructs the 6x6 covariance matrix
	'''
	cov = np.array([[table['Sigma_x_x'], table['Sigma_x_y'], table['Sigma_x_z'], table['Sigma_x_vx'], table['Sigma_x_vy'], table['Sigma_x_vz']],
					[table['Sigma_x_y'], table['Sigma_y_y'], table['Sigma_y_z'], table['Sigma_y_vx'], table['Sigma_y_vy'], table['Sigma_y_vz']],
					[table['Sigma_x_z'], table['Sigma_y_z'], table['Sigma_z_z'], table['Sigma_z_vx'], table['Sigma_z_vy'], table['Sigma_z_vz']],
					[table['Sigma_x_vx'],table['Sigma_y_vx'], table['Sigma_z_vx'], table['Sigma_vx_vx'], table['Sigma_vx_vy'], table['Sigma_vx_vz']],
					[table['Sigma_x_vy'], table['Sigma_y_vy'], table['Sigma_z_vy'], table['Sigma_vx_vy'], table['Sigma_vy_vy'], table['Sigma_vy_vz']],
					[table['Sigma_x_vz'], table['Sigma_y_vz'], table['Sigma_z_vz'], table['Sigma_vx_vz'], table['Sigma_vy_vz'], table['Sigma_vz_vz']]])
	return cov.T


