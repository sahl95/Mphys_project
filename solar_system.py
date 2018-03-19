"""
Simulates the Solar System.
"""
import pandas as pd
import time
import numpy as np
import numpy.ma as ma
from scipy import integrate
import scipy.linalg
from scipy.optimize import fsolve
from sympy import symbols, Matrix, linsolve, diag
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation
import matplotlib
from planet import planet

# plt.rcParams['grid.color'] = (0.3, 0.3, 0.3)
# plt.rcParams['xtick.color'] = 'white'
# plt.rcParams['ytick.color'] = 'white'
# plt.rcParams['text.color'] = 'white'
# plt.rcParams['axes.labelcolor'] = 'white'
# # plt.rcParams['savefig.facecolor']='black'
# matplotlib.rc('axes', edgecolor='white')
SMALL_SIZE = 10
MEDIUM_SIZE = 13
BIGGER_SIZE = 15
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title 
# plt.rcParams['xtick.color'] = 'white'
# plt.rcParams['ytick.color'] = 'white'
# plt.rcParams['text.color'] = 'white'
# plt.rcParams['axes.labelcolor'] = 'white'

# red = (255/255, 81/255, 81/255)
# orange = 'orange'
# blue = (0, 191/255, 255/255)
# green = (86/255, 255/255, 71/255)
# purple = (157/255, 136/255, 239/255)


M_SUN = 1.9885*10**30
R_SUN = 6.9551*10**8
M_EARTH = 5.9726*10**24
AU = 149597870700
SECS_IN_YEAR = 365.25*24*3600
G_CONST = 6.6738*10**(-11)
LIGHT_SPD = 63241.0774

def calculate_laplace_coeff(alpha, j, s):
    r"""
    Calculates the Laplace coeffecient given by

    .. math::
        b^{(j)}_{s} (\alpha) = \frac{1}{\pi}\int_{0}^{2\pi}\left [ \frac{cos(j \psi)}{(1+\alpha^{2} - 2\alpha \cos\, \psi)^{s}} \right]d\psi.

    Args:
        alpha (float):
        j (float):
        s (float):
    
    Returns:
        The value of the Laplace coefficient.
    """
    return integrate.quad(lambda psi, alpha, j, s: np.cos(j*psi)/(1-2*alpha*np.cos(psi)+alpha**2)**s,
                          0, 2*np.pi, args=(alpha, j, s,))[0]/np.pi

def scaling_factor_and_phase(p, *boundaries):
    s, phase = p
    return (s*np.sin(phase)-boundaries[0], s*np.cos(phase)-boundaries[1])

def f(x, *boundaries):
    s_factor, phase = x
    return [s_factor*np.sin(phase)-boundaries[0], s_factor*np.cos(phase)-boundaries[1]]

def real_scaling_factor_and_phase(x1, *boundaries):
    s_factor, phase = x1[0]+1j*x1[1], x1[2]+1j*x1[3] 
    x = [s_factor, phase]
    actual_f = f(x, *boundaries)
    return [np.real(actual_f[0]), np.imag(actual_f[0]), np.real(actual_f[1]), np.imag(actual_f[1])]

class solar_System():
    """
    Stores the mass and radius of the Sun. Also stores the properties of the orbiting planets.

    Args:
        starMass (float): The mass of the Sun in solar masses.
        starRadius (float): The radius of the Sun in solar radii.
        planet_data_file (str): A data file containing planet orbital properties.
    """
    def __init__(self, star_data_file, planet_data_file):
        star_data = pd.read_csv(star_data_file)
        self.star_mass = star_data['star_mass'][0]
        self.star_radius = star_data['star_radius'][0]
        self.planets = self.add_planets(planet_data_file)

    def add_planets(self, planet_data_file):
        """
        Create planet objects using :func:`~planet.planet` and adds each planet to a list.

        Args:
            planet_data_file (str): A data file containing planet orbital properties.

        Returns:
            A list of planet objects.
        """
        planets = pd.read_csv(planet_data_file)
        planet_list = []
        for p in range(len(planets)):
            planet_list.append(planet(**planets.ix[p]))
        return planet_list

    def __str__(self):
        output = ''
        for p in self.planets:
            output += p.__str__()+'\n'
        return output

    def get_property_all_planets(self, property_name):
        """
        Returns an array containing a specific property of all planets. The possible property names can be
        found in :func:`~planet.planet`.

        Args:
            property_name (str): The name of the property.

        Returns:
            Array containing the specified property of each planet.
        """
        return np.array([p.__dict__[property_name] for p in self.planets])

    def set_n(self):
        for p in self.planets:
            p.n = np.sqrt(G_CONST*self.star_mass*M_SUN/(p.a*AU)**3)*365*24*3600*180/np.pi

    def frequency_matrix(self, matrix_id, J2=0, J4=0):
        r"""
        Calculates the frequency matrix with the following correction terms included:

        * Oblateness of the central body.
        * General Relativity (For 'A' only).
        * Eccentricity damping (For 'A' only).

        Args:        
            J2 (float): First zonal gravity coefficient.
            J4 (float): Second zonal gravity coefficient.
            matrix_id (str): Can be 'A' or 'B'.
                
        Returns:
            The frequency matrix, A or B as defined below.
        
        Note:
            If 'A'

            .. math::
                A_{jj} = n_{j} & \left [ \frac{3}{2}J_{2} \left ( \frac{R_{\star}}{a_{j}} \right)^{2} -  \frac{9}{8}J_{2}^{2} \left ( \frac{R_{\star}}{a_{j}} \right)^{4} -  \frac{15}{4}J_{4}^{2} \left ( \frac{R_{\star}}{a_{j}} \right)^{4} \vphantom{\sum _{k\neq}} \right. \\
                & \left. + 3\frac{a_{j}^{2}n_{j}^{2}}{c^{2}} + \frac{63}{4} \frac{1}{Q{}'_{p}}\frac{m_{\star}}{m_{p}} \left (\frac{R_{p}}{a_{p}} \right)^{5} \right.\\
                & \left. + \frac{1}{4} \sum _{k\neq j} \frac{m_{k}}{m_{\star}+m_{j}} \alpha_{jk} \bar{\alpha}_{jk} b_{3/2}^{(1)}(\alpha_{jk}) \vphantom{\sum _{k\neq}} \right]
        
            .. math::
                A_{jk} = -\frac{n_{j}}{4} \frac{m_{k}}{m_{\star}+m_{j}} \alpha_{jk} \bar{\alpha}_{jk} b_{3/2}^{(2)}(\alpha_{jk}) \hspace*{1.8cm} (j \neq k)
            
            If 'B'

            .. math::
                B_{jj} = -n_{j} & \left [ \frac{3}{2}J_{2} \left ( \frac{R_{\star}}{a_{j}} \right)^{2} -  \frac{27}{8}J_{2}^{2} \left ( \frac{R_{\star}}{a_{j}} \right)^{4} -  \frac{15}{4}J_{4}^{2} \left ( \frac{R_{\star}}{a_{j}} \right)^{4} \vphantom{\sum _{k\neq}} \right. \\
                & \left. + \frac{1}{4} \sum _{k\neq j} \frac{m_{k}}{m_{\star}+m_{j}} \alpha_{jk} \bar{\alpha}_{jk} b_{3/2}^{(1)}(\alpha_{jk}) \vphantom{\sum _{k\neq}} \right]
            
            .. math::
                B_{jk} = \frac{n_{j}}{4} \frac{m_{k}}{m_{\star}+m_{j}} \alpha_{jk} \bar{\alpha}_{jk} b_{3/2}^{(1)}(\alpha_{jk}) \hspace*{2.65cm} (j \neq k)
            
        See also:
            :func:`calculate_laplace_coeff`
        """
        M = M_SUN*self.star_mass
        R = R_SUN*self.star_radius/AU
        m = M_EARTH*self.get_property_all_planets('Mass')
        n = self.get_property_all_planets('n')*np.pi/180
        a = self.get_property_all_planets('a')
        e = self.get_property_all_planets('e')

        do_ecc_damp = all(x is not None for x in self.get_property_all_planets('r'))
        if do_ecc_damp:
            k = self.get_property_all_planets('k')
            Q = self.get_property_all_planets('Q')
            r = self.get_property_all_planets('r')/AU
            spin = self.get_property_all_planets('spin')
        n_planets = len(self.planets)
        # f_mat = np.zeros([n_planets, n_planets])
        f_mat = np.zeros([n_planets, n_planets], dtype='complex128')

        if matrix_id == 'A':
            j_laplace_coeff_jk, j_laplace_coeff_jj = 2, 1
            front_factor = -1
            J2_correction = (((3/2)*J2*(R/a)**2)-((9/8)*(J2**2)*(R/a)**4)-((15/4)*J4*(R/a)**4))
            if np.isnan(R): J2_correction = np.zeros(len(self.planets), dtype=complex)
            if do_ecc_damp:
                ecc_damp = (63/4)*(k/(1.5*Q))*(M/m)*(r/a)**5
                tide = (15/2)*k*((1+3/2*e**2+1/8*e**4)/(1-e**2)**5)*M/m*(r/a)**5
                # print(tide)
                if all(x is not None for x in self.get_property_all_planets('spin')):
                    rot = 0.5*(k/(1-e**2))*(spin/n)**2*M/m*(r/a)**5
                    # print(rot)
                # print(ecc_damp)
            gr_correction = 3*(a)**2*(n)**2/(LIGHT_SPD**2*(1+m/M))*1/(1-e**2)
            # print(gr_correction)
            # gr_correction[0] = 3*(a[0]/AU)**2*(n[0])**3/(LIGHT_SPD**2*(1+m[0]/M))*1/(1-e[0]**2)

        if matrix_id == 'B':
            j_laplace_coeff_jk = j_laplace_coeff_jj = 1
            front_factor = 1
            J2_correction = (((3/2)*J2*(R/a)**2)-((27/8)*(J2**2)*(R/a)**4)-((15/4)*J4*(R/a)**4))
            if np.isnan(R): J2_correction = np.zeros(len(self.planets), dtype=complex)
            gr_correction = np.zeros(n_planets)
            if do_ecc_damp: 
                ecc_damp = np.zeros(n_planets)
                tide = np.zeros(n_planets)
            if all(x is not None for x in self.get_property_all_planets('spin')): rot = np.zeros(n_planets)
            # n_merc = np.sqrt(G_CONST*M/a[0]**3)

        for j in range(n_planets):
            for k in range(n_planets):
                if j != k:
                    alpha_jk = a[j]/a[k]
                    if alpha_jk > 1:
                        alpha_jk = alpha_jk**(-1)
                    laplace_coeff = calculate_laplace_coeff(alpha_jk, j_laplace_coeff_jk, 3./2.)
                    # print(alpha_jk, j_laplace_coeff_jk, 3./2., laplace_coeff)
                    alpha_jk_bar = np.where(a[k] < a[j], 1., alpha_jk)
                    f_mat[j, k] = front_factor*(n[j]/4.)*(m[k]/(M+m[j]))*alpha_jk*alpha_jk_bar*laplace_coeff
            
                else:
                    for kk in range(n_planets):
                        if kk != j:
                            alpha_jj = a[j]/a[kk]
                            if alpha_jj > 1:
                                alpha_jj = alpha_jj**(-1)
                            laplace_coeff = calculate_laplace_coeff(alpha_jj, j_laplace_coeff_jj, 3./2.)
                            # print(alpha_jj, j_laplace_coeff_jj, 3./2., laplace_coeff)
                            alpha_jj_bar = np.where(a[kk] < a[j], 1., alpha_jj)
                            f_mat[j, k] += (1./4.)*(m[kk]/(M+m[j]))*alpha_jj*alpha_jj_bar*laplace_coeff
                    # f_mat[j, k] += J2_correction[j]
                    # if do_ecc_damp: f_mat[j, k] += 1j*ecc_damp[j]+1j*tide[j]
                    # if all(x is not None for x in self.get_property_all_planets('spin')):
                        # f_mat[j, k] += 1j*rot[j]
                    # f_mat[j, k] += gr_correction[j]
                    f_mat[j, k] *= -front_factor*(n[j])
                    # print(J2_correction[j], ecc_damp[j], gr_correction[j])
                # print(f_mat[j, k])
        # print(f_mat[0, 0], gr_correction[0])
        return f_mat

    def A_octupole(self):
        if len(self.planets) != 2:
            print('here')
            return self.frequency_matrix('A')
        else:
            M = M_SUN*self.star_mass
            R = R_SUN*self.star_radius/AU
            m = M_EARTH*self.get_property_all_planets('Mass')
            n = self.get_property_all_planets('n')*np.pi/180
            a = self.get_property_all_planets('a')

            A = np.zeros((2, 2))
            alpha = np.min(a)/np.max(a)
            A[0, 0] = 3/4*n[0]*(m[1]/(M+m[0]))*alpha**3
            A[0, 1] = -15/16*n[0]*(m[1]/(M+m[0]))*((M-m[0])/(M+m[0]))*alpha**4
            A[1, 0] = -15/16*n[1]*(M*m[0]/(M+m[0])**2)*((M-m[0])/(M+m[0]))*alpha**3
            A[1, 1] = 3/4*n[1]*(M*m[0])/((M+m[0])**2)*alpha**2
            # print(A)
            return A

    def initial_conditions(self):
        """
        Calculates the initial values for the 4 equations of motion (h, k, p, and q) for each planet.

        Returns:
            * :math:`h` (*numpy array*)
            * :math:`k` (*numpy array*)
            * :math:`p` (*numpy array*)
            * :math:`q` (*numpy array*)
        """
        e = self.get_property_all_planets('e')
        pi = self.get_property_all_planets('pi')*np.pi/180
        i = self.get_property_all_planets('i')*np.pi/180
        omega = self.get_property_all_planets('Omega')*np.pi/180

        h = np.array(e*np.sin(pi), dtype='complex128')
        k = np.array(e*np.cos(pi), dtype='complex128')
        p = np.array(i*np.sin(omega), dtype='complex128')  # WHY DIVIDE BY 2 TO MATCH LASKAR 1986
        q = np.array(i*np.cos(omega), dtype='complex128')

        return h, k, p, q

    def solve_property(self, eigenvectors, initial_conditions):
        answers = np.linalg.solve(eigenvectors, initial_conditions)
        return answers

    def find_all_scaling_factor_and_phase(self, eigenvectors_of_A, eigenvectors_of_B):
        r"""
        Calculates the scale factors :math:`S, T` of the eigenvectors and the phases, :math:`\beta, \gamma`.

        Args:
            eigenvectors_of_A (numpy array): The normalised eigenvectors of the frequency matrix A.
            eigenvectors_of_B (numpy array): The normalised eigenvectors of the frequency matrix B.
        
        Returns:
            * :math:`S` (*numpy array*) - Scaling factor for the eigenvectors of A.
            * :math:`\beta` (*numpy array*) - Phase shift for :math:`h, k`
            * :math:`T` (*numpy array*) - Scaling factor for the eigenvectors of B.
            * :math:`\gamma` (*numpy array*) - Phase shift for :math:`p, q`
        """
        x, y = eigenvectors_of_A, eigenvectors_of_B

        init_conditions = np.array(self.initial_conditions())
        # print(init_conditions[:, 3])
        h_solved = self.solve_property(x, init_conditions[0, :])
        k_solved = self.solve_property(x, init_conditions[1, :])
        p_solved = self.solve_property(y, init_conditions[2, :])
        q_solved = self.solve_property(y, init_conditions[3, :])

        n = len(self.planets)
        S, beta = np.zeros(n, dtype='complex128'), np.zeros(n, dtype='complex128')
        T, gamma = np.zeros(n, dtype='complex128'), np.zeros(n, dtype='complex128')

        for i in range(n):
            # S[i], beta[i] = fsolve(scaling_factor_and_phase, (1, -1), args=(h_solved[i], k_solved[i],))
            # T[i], gamma[i] = fsolve(scaling_factor_and_phase, (-1, 1), args=(p_solved[i], q_solved[i],))
            S_real, S_img, beta_real, beta_img = fsolve(real_scaling_factor_and_phase, (1, 1, -1, -1), args=(h_solved[i], k_solved[i],))
            T_real, T_img, gamma_real, gamma_img = fsolve(real_scaling_factor_and_phase, (1, 1, -1, -1), args=(p_solved[i], q_solved[i],))

            S[i], beta[i] = S_real+1j*S_img, beta_real+1j*beta_img
            T[i], gamma[i] = T_real+1j*T_img, gamma_real+1j*gamma_img
            
        return S, beta, T, gamma

    def components_of_ecc_inc(self, scaled_eigenvector, eigenvalue, phase, t, eq_id):
        r"""
        Calculates the vertical and horizontal components of the eccentricity and inclination for all times.

        Args:
            scaled_eigenvector (numpy array): The scaled eigenvector of the frequency matrix.
            eigenvalue (numpy array): The eigenvalues of the frequency matrix.
            phase (numpy array): The phase shift.
            t (numpy array): The times to calculate the ertical and horizontal components of the eccentricity
                and inclination at.
            eq_id (str): Can be 'h', 'k', 'p', or 'q'

        Returns:
            An array containing the values of the equation of motion at all time.

        Note:
            The vertical and horizontal components of the eccentricity and inclination are given by

            .. math::
                h_{j} = \sum_{i=0}^{N-1} e_{ji} \sin (g_{i}t+\beta_{i})\\
                k_{j} = \sum_{i=0}^{N-1} e_{ji} \cos (g_{i}t+\beta_{i})\\
                p_{j} = \sum_{i=0}^{N-1} I_{ji} \sin (f_{i}t+\gamma_{i})\\
                q_{j} = \sum_{i=0}^{N-1} I_{ji} \cos (f_{i}t+\gamma_{i})

            where :math:`e, I` are the scaled eigenvectors and :math:`f, g` are the eigenvalues of A and B
            respectively.

        See also:
            :func:`frequency_matrix`
        """
        # eq_id = 'h', 'k', 'p', 'q'
        kwargs = {'scaled_eigenvector' : scaled_eigenvector, 'eigenvalue' : eigenvalue, 'phase' : phase,
                  't' : t}
        if eq_id == 'h' or eq_id == 'p':
            return self.get_h_or_p(scaled_eigenvector=scaled_eigenvector, eigenvalue=eigenvalue, phase=phase, t=t)
        if eq_id == 'k' or eq_id == 'q':
            return self.get_k_or_q(scaled_eigenvector=scaled_eigenvector, eigenvalue=eigenvalue, phase=phase, t=t)
    
    def get_h_or_p(self, scaled_eigenvector, eigenvalue, phase, t):
        n = len(self.planets)
        h_list = []
        for j in range(n):
            h = np.zeros_like(t)
            for i in range(n):
                h += scaled_eigenvector[j, i]*np.sin((eigenvalue[i]*t+phase[i]))
            h_list.append(h)
        return np.array(h_list)

    def get_k_or_q(self, scaled_eigenvector, eigenvalue, phase, t):
        n = len(self.planets)
        k_list = []
        for j in range(n):
            k = np.zeros_like(t)
            for i in range(n):
                k += scaled_eigenvector[j, i]*np.cos((eigenvalue[i]*t+phase[i]))
            k_list.append(k)
        return np.array(k_list)

    def get_eccentricity(self, h_arr, k_arr):
        """
        Calculates the eccentricity of all planets.

        Args:
            h_arr (numpy array): The vertical component of the eccentricity of each planet.
            k_arr (numpy array): The horizontal component of the eccentricity of each planet.

        Returns:
            An array containing the values of the eccentricity at all times.

        See also:
            :func:`components_of_ecc_inc`
        """
        n = len(self.planets)
        h, k = h_arr, k_arr
        eccentricities = []
        for j in range(n):  
            # eccentricities.append(np.sqrt(h[j]**2+k[j]**2))
            eccentricities.append(np.real(np.sqrt(h[j]*np.conjugate(h[j])+k[j]*np.conjugate(k[j]))))
        return np.array(eccentricities)

    def get_inclination(self, p_arr, q_arr):
        """
        Calculates the inclination of all planets.

        Args:
            p_arr (numpy array): The vertical component of the inclination of each planet.
            q_arr (numpy array): The horizontal component of the inclination of each planet.

        Returns:
            An array containing the values of the inclination at all times.

        See also:
            :func:`components_of_ecc_inc`
        """
        n = len(self.planets)
        inclinations = []
        p, q = p_arr, q_arr
        for j in range(n):
            inclinations.append(np.real(np.sqrt(p[j]*np.conjugate(p[j])+q[j]*np.conjugate(q[j]))))
        return np.array(inclinations)

    def get_perihelion_precession_rates(self, A, eccentricities, h_arr, k_arr):
        """
        Calculates the rate of precession of each planet.

        Args:
            A (numpy array): The frequency matrix, A.
            eccentricities (numpy array): The eccentricity of each planet.
            h_arr (numpy array): The vertical component of the eccentricity of each planet.
            k_arr (numpy array): The horizontal component of the eccentricity of each planet.

        Returns:
            A list of arrays containing the precession rate of each planet.
        """
        n = len(self.planets)
        d_pidot_dt_list = []

        for j in range(n):
            h_dot_j, k_dot_j = 0, 0
            for i in range(n):
                h_dot_j += A[j, i]*k_arr[i]
                k_dot_j -= A[j, i]*h_arr[i]
            pidot_j = (k_arr[j]*h_dot_j - h_arr[j]*k_dot_j)/(eccentricities[j])**2
            # pidot_j = ma.masked_array(pidot_j, np.abs(pidot_j)>10).filled(0)

            d_pidot_dt_list.append((pidot_j))
        return d_pidot_dt_list

    def get_ascending_node_precession_rates(self, B, inclinations, p_list, q_list):
        n = len(self.planets)
        d_Omega_dt_list = []
        masks = []

        for j in range(n):
            p_dot_j, q_dot_j = 0, 0
            for i in range(n):
                p_dot_j += B[j, i]*q_list[i]
                q_dot_j -= B[j, i]*p_list[i]
            pidot_j = 3600*(q_list[j]*p_dot_j - p_list[j]*q_dot_j)/(inclinations[j])**2
            # pidot_j = ma.masked_array(pidot_j, np.abs(pidot_j)>10).filled(0)

            d_Omega_dt_list.append(np.real(pidot_j))
        return d_Omega_dt_list

    def get_pi_or_omega(self, hp, kq):
        r"""
        Calculates :math:`\varpi` (the longitude of periapsis) of each planet if given :math:`h` and :math:`k`
        as arguments. Calculates :math:`\Omega` (the longitude of the ascending node) if given :math:`p` and
        :math:`q` as arguments.

        Args:
            hp (numpy array): Values of :math:`h` or :math:`p` of each planet.
            kq (numpy array): Values of :math:`k` or :math:`q` of each planet.

        Returns:
            An array containing the values of :math:`\varpi` or :math:`\Omega`.
        """
        pi_om = []
        for i in range(len(self.planets)):
            pi_om.append(np.arctan2(np.real(hp[i]), np.real(kq[i])))
        return np.array(pi_om)

    def kep2cart(self, ecc, inc, h_arr, k_arr, p_arr, q_arr, time, t0, idx):
        """
        Converts from Keplerian coordinates to Cartesian coordinates.

        Args:
            ecc (numpy array): Array of eccentricities of all planets.
            inc (numpy array): Array of inclinations of all planets.
            h_arr (numpy array): The vertical component of the eccentricity of each planet.
            k_arr (numpy array): The horizontal component of the eccentricity of each planet.
            p_arr (numpy array): The vertical component of the inclination of each planet.
            q_arr (numpy array): The horizontal component of the inclination of each planet.
            time (numpy array): Array of the times to calculate at.
            t0 (numpy array): The time at which the initial conditions are defined (normally 0).
            idx (int): Chooses which planet to convert to Cartesian coordinates.

        Returns:
            An array containing the x, y, z coordinates of the planet at all times.
        """
        time = np.real(time)
        
        O_list = self.get_pi_or_omega(p_arr, q_arr)
        w_list = self.get_pi_or_omega(h_arr, k_arr)-O_list

        n = self.get_property_all_planets('n')#/(SECS_IN_YEAR)*np.pi/180
        a = self.get_property_all_planets('a')

        Mt = n[idx]*np.pi/180*(time-t0)

        EA = []
        e, w, O, i = ecc[idx], w_list[idx], O_list[idx], inc[idx]
        # print('e = {:.4f}, i = {:.4f}, w = {:.4f}, O = {:.4f}'.format(e[0], i[0]*180/np.pi, w[0]*180/np.pi, O[0]*180/np.pi))
        # print('h = {:.4f}, k = {:.4f}, p = {:.4f}, q = {:.4f}'.format(np.real(h_arr[idx][0]), np.real(k_arr[idx][0]), np.real(p_arr[idx][0]), np.real(q_arr[idx][0])))
        
        # e0, i0 = e[0], i[0]*180/np.pi
        # pi0 = np.arctan(np.real(h_arr[idx][0])/np.real(k_arr[idx][0]))*180/np.pi
        # Om0 = np.arctan(np.real(p_arr[idx][0])/np.real(q_arr[idx][0]))*180/np.pi
        # print(e0, i0, pi0, Om0)

        for t in range(len(time)):
            E = Mt[t]
            f_by_dfdE = (E-e[t]*np.sin(E)-Mt[t])/(1-e[t]*np.cos(E))
            j, maxIter, delta = 0, 30, 0.0000000001
            while (j < maxIter)*(np.abs(f_by_dfdE) > delta):
                E = E-f_by_dfdE
                f_by_dfdE = (E-e[t]*np.sin(E)-Mt[t])/(1-e[t]*np.cos(E))
                j += 1
            EA.append(E)
        EA = np.array(EA)
        # print(EA)
        nu = 2*np.arctan2(np.sqrt(1+e)*np.sin(EA/2), np.sqrt(1-e)*np.cos(EA/2))

        rc = a[idx]*(1-e*np.cos(EA))
        # print('a: {}, {}\nr_max: {}, {}\nr_min {}, {}\n'.format(a[idx], np.mean(rc), a[idx]*(1+np.max(e)), np.max(rc), a[idx]*(1-np.min(e)), np.min(rc)))

        o_vec = np.array([rc*np.cos(nu), rc*np.sin(nu), 0])

        rx = (o_vec[0]*(np.cos(w)*np.cos(O)-np.sin(w)*np.cos(i)*np.sin(O))-
              o_vec[1]*(np.sin(w)*np.cos(O)+np.cos(w)*np.cos(i)*np.sin(O)))
        ry = (o_vec[0]*(np.cos(w)*np.sin(O)+np.sin(w)*np.cos(i)*np.cos(O))+
              o_vec[1]*(np.cos(w)*np.cos(i)*np.cos(O)-np.sin(w)*np.sin(O)))
        rz = (o_vec[0]*(np.sin(w)*np.sin(i))+o_vec[1]*(np.cos(w)*np.sin(i)))
        
        # cos(relative angle) = (r_inner dot r_outer)/|r_inner||r_outer|
        return np.array([rx, ry, rz])

    def kep2cart_2(self, ecc, inc, h_arr, k_arr, p_arr, q_arr, time, t0, idx):
        time = np.real(time)
        
        O_list = self.get_pi_or_omega(p_arr, q_arr)
        w_list = self.get_pi_or_omega(h_arr, k_arr)-O_list
        n = self.get_property_all_planets('n')#/(SECS_IN_YEAR)*np.pi/180
        a = self.get_property_all_planets('a')
        Mt = n[idx]*np.pi/180*(time-t0)

        e, w, O, i = ecc[idx], w_list[idx], O_list[idx], inc[idx]

        E = Mt
        f_by_dfdE = (E-e*np.sin(E)-Mt)/(1-e*np.cos(E))
        j, maxIter, delta = 0, 30, 0.0000001
        while (j < maxIter):
            if np.sum((np.abs(f_by_dfdE) > delta)) == 0:
                break
            E = E-f_by_dfdE
            f_by_dfdE = (E-e*np.sin(E)-Mt)/(1-e*np.cos(E))
            j += 1
        EA = E

        nu = 2*np.arctan2(np.sqrt(1+e)*np.sin(EA/2), np.sqrt(1-e)*np.cos(EA/2))

        rc = a[idx]*(1-e*np.cos(EA))
        # print('a: {}, {}\nr_max: {}, {}\nr_min {}, {}\n'.format(a[idx], np.mean(rc), a[idx]*(1+np.max(e)), np.max(rc), a[idx]*(1-np.min(e)), np.min(rc)))

        o_vec = np.array([rc*np.cos(nu), rc*np.sin(nu), 0])

        rx = (o_vec[0]*(np.cos(w)*np.cos(O)-np.sin(w)*np.cos(i)*np.sin(O))-
              o_vec[1]*(np.sin(w)*np.cos(O)+np.cos(w)*np.cos(i)*np.sin(O)))
        ry = (o_vec[0]*(np.cos(w)*np.sin(O)+np.sin(w)*np.cos(i)*np.cos(O))+
              o_vec[1]*(np.cos(w)*np.cos(i)*np.cos(O)-np.sin(w)*np.sin(O)))
        rz = (o_vec[0]*(np.sin(w)*np.sin(i))+o_vec[1]*(np.cos(w)*np.sin(i)))
        
        # cos(relative angle) = (r_inner dot r_outer)/|r_inner||r_outer|
        return np.array([rx, ry, rz])

    def initial_pos_vel(self, idx):
        n = self.get_property_all_planets('n')
        a = self.get_property_all_planets('a')
        ecc = self.get_property_all_planets('e')
        inc = self.get_property_all_planets('i')*np.pi/180
        O_list = self.get_property_all_planets('Omega')*np.pi/180
        w_list = self.get_property_all_planets('pi')*np.pi/180-O_list

        M = 0
        e, w, O, i = ecc[idx], w_list[idx], O_list[idx], inc[idx]
        # print('e = {:.4f}, i = {:.4f}, w = {:.4f}, O = {:.4f}'.format(e, i*180/np.pi, w*180/np.pi, O*180/np.pi))
        # print(e, i, w, O)
        E = M
        f_by_dfdE = (E-e*np.sin(E)-M)/(1-e*np.cos(E))
        j, maxIter, delta = 0, 30, 0.0000000001
        while (j < maxIter)*(np.abs(f_by_dfdE) > delta):
            E = E-f_by_dfdE
            f_by_dfdE = (E-e*np.sin(E)-M)/(1-e*np.cos(E))
            j += 1
        EA = E
        # print(EA)
        nu = 2*np.arctan2(np.sqrt(1+e)*np.sin(EA/2), np.sqrt(1-e)*np.cos(EA/2))

        rc = a[idx]*(1-e*np.cos(EA))
        # print('a: {}, {}\nr_max: {}, {}\nr_min {}, {}\n'.format(a[idx], np.mean(rc), a[idx]*(1+np.max(e)), np.max(rc), a[idx]*(1-np.min(e)), np.min(rc)))

        o_vec = np.array([rc*np.cos(nu), rc*np.sin(nu), 0])

        mu = G_CONST*self.star_mass*M_SUN
        o_dot_vec = (np.sqrt(mu*a[idx]*AU)/(rc*AU))*np.array([-np.sin(EA), np.sqrt(1-e**2)*np.cos(EA), 0])
        # print(o_dot_vec)
        # print('mu = {}'.format(G_CONST*self.star_mass*M_SUN))

        rx = (o_vec[0]*(np.cos(w)*np.cos(O)-np.sin(w)*np.cos(i)*np.sin(O))-
              o_vec[1]*(np.sin(w)*np.cos(O)+np.cos(w)*np.cos(i)*np.sin(O)))
        ry = (o_vec[0]*(np.cos(w)*np.sin(O)+np.sin(w)*np.cos(i)*np.cos(O))+
              o_vec[1]*(np.cos(w)*np.cos(i)*np.cos(O)-np.sin(w)*np.sin(O)))
        rz = (o_vec[0]*(np.sin(w)*np.sin(i))+o_vec[1]*(np.cos(w)*np.sin(i)))

        vx = (o_dot_vec[0]*(np.cos(w)*np.cos(O)-np.sin(w)*np.cos(i)*np.sin(O))-
              o_dot_vec[1]*(np.sin(w)*np.cos(O)+np.cos(w)*np.cos(i)*np.sin(O)))
        vy = (o_dot_vec[0]*(np.cos(w)*np.sin(O)+np.sin(w)*np.cos(i)*np.cos(O))+
              o_dot_vec[1]*(np.cos(w)*np.cos(i)*np.cos(O)-np.sin(w)*np.sin(O)))
        vz = (o_dot_vec[0]*(np.sin(w)*np.sin(i))+o_dot_vec[1]*(np.cos(w)*np.sin(i)))

        return np.array([rx, ry, rz]), np.array([vx, vy, vz])*(86400/AU)

    def initial_pos_vel_all(self):

        xyz_s, uvw_s = [], []
        n = len(self.planets)
        for idx in range(n):
            xyz, uvw = self.initial_pos_vel(idx)
            xyz_s.append(xyz); uvw_s.append(uvw)
        
        return np.array(xyz_s), np.array(uvw_s)

    def centre_of_mass(self):
        n = len(self.planets)
        m = self.get_property_all_planets('Mass')
        m_total = np.sum(m)+(self.star_mass*M_SUN/M_EARTH)

        xyz_s, uvw_s = self.initial_pos_vel_all()
        xyz_center, uvw_center = np.array([0., 0., 0.]), np.array([0., 0., 0.])

        for idx in range(n):
            xyz_center += m[idx]*xyz_s[idx]
            uvw_center += m[idx]*uvw_s[idx]
        xyz_center *= 1/m_total
        uvw_center *= 1/m_total

        # print(xyz_center)
        # for idx in range(n):
        #     print(xyz_s[idx], xyz_s[idx]-xyz_center) #, np.sqrt(np.sum(xyz_s[idx]**2))
        # print()
        # for idx in range(n):
        #     print(uvw_s[idx], uvw_s[idx]-uvw_center)
        return xyz_center, uvw_center

    def simulate(self, t, plot_orbit=False, plot=False, separate=True):
        self.set_n()

        A, B = [self.frequency_matrix(matrix_id=mat_id, J2=2.2*10**(-7), J4=-4.45*10**(-9)) for mat_id in ['A', 'B']]
        g, x = np.linalg.eig(A)
        f, y = np.linalg.eig(B)

        S, beta, T, gamma = self.find_all_scaling_factor_and_phase(x, y)

        h_list = self.components_of_ecc_inc(scaled_eigenvector=S*x, eigenvalue=g, phase=beta, t=t, eq_id='h')
        k_list = self.components_of_ecc_inc(scaled_eigenvector=S*x, eigenvalue=g, phase=beta, t=t, eq_id='k')
        
        p_list = self.components_of_ecc_inc(scaled_eigenvector=T*y, eigenvalue=f, phase=gamma, t=t, eq_id='p')#*180/np.pi
        q_list = self.components_of_ecc_inc(scaled_eigenvector=T*y, eigenvalue=f, phase=gamma, t=t, eq_id='q')#*180/np.pi

        eccentricities = self.get_eccentricity(h_list, k_list)
        inclinations = self.get_inclination(p_list, q_list)
        names = self.get_property_all_planets('Name')

        t = np.real(t)
        if plot:
            if separate:
                plot_simulation_separate(t/10**0, eccentricities, 'Time (years)', 'Eccentricity', names)
                plot_simulation_separate(t/10**0, inclinations*180/np.pi, 'Time (years)', 'Inclination', names)
            else:
                plot_simulation_all(t/10**0, eccentricities, 'Time (years)', 'Eccentricity', names, save_name='js_ecc')
                plot_simulation_all(t/10**0, inclinations*180/np.pi, 'Time (years)', 'Inclination', names, save_name='js_inc')

        if plot_orbit:
            x, y, z = np.zeros(2), np.zeros(2), np.zeros(2)
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.view_init(25, 45)
            ax.plot(x, y, z, 'b*', markersize=3, zorder=-999)
            # ax = fig.add_subplot(111)
            # ax.plot(0, 0, 'b*', markersize=3)
            for idx in range(len(self.planets)):
                # t1 = time.clock()
                # xyz = self.kep2cart(eccentricities, inclinations, h_list, k_list, p_list, q_list, t, 0, idx)
                # t2 = time.clock()
                xyz = self.kep2cart_2(eccentricities, inclinations, h_list, k_list, p_list, q_list, t, 0, idx)
                # t3 = time.clock()
                # print(t3-t2, t2-t1)

                # ax.plot(X, Y, '.', markersize=2, label=names[idx])
                # if idx == 2:
                #     ax.plot(xyz[0][80:], xyz[1][80:], xyz[2][80:], '--', markersize=2, label=names[idx], zorder=-idx)
                # else:
                #     ax.plot(*xyz, '--', markersize=2, label=names[idx], zorder=-idx)
                
                ax.plot(xyz, '.', markersize=2, label=names[idx], zorder=-idx)
                max_axis = np.max([np.abs(np.min(xyz)), np.max(xyz)])
                ax.set_zlim(-max_axis, max_axis)
                ax.set_ylim(-max_axis, max_axis)
                ax.set_xlim(-max_axis, max_axis)
                ax.set_zlabel('z (AU)')
                plt.xlabel('x (AU)')
                plt.ylabel('y (AU)')

                df = pd.DataFrame({"time": t ,"x" : xyz[0], "y" : xyz[1], "z" : xyz[2]})
                df.to_csv('Animate_solar_system/'+str(idx+1)+'_'+names[idx]+'_xyz.csv', index=False)

        precession_rates, ylabel = self.get_perihelion_precession_rates(A, eccentricities, h_list, k_list), 'Pericenter'
        # precession_rates, xlabel = self.get_ascending_node_precession_rates(B, inclinations*180/np.pi, p_list, q_list), 'Ascending node'

        # inclinations *= 180/np.pi
        idx = 0
        # print(np.real(np.mean(precession_rates[idx]*180/np.pi*3600*100)))
        # plot_precession_rate(t, precession_rates[idx]*180/np.pi*3600*100, ylabel+r" precession rate [${}'{}'\ 100y^{-1}$]", names[idx])
        # plot_eccentricity(t, eccentricities[idx], names[idx])
        # print(np.max(eccentricities[idx]), np.min(eccentricities[idx]), np.mean(eccentricities[idx]), '\n')

        # for idx in range(len(precession_rates)):
        #     print('Precession rate of {} = {:.4f} arcseconds per century'.format(names[idx],
        #           np.mean(precession_rates[idx])*180/np.pi*3600*100))
            # print('Eccentricity of {} = {:.4f}'.format(names[idx],
            #       np.mean(eccentricities[idx])))
            # print('Inclination of {} = {:.2f} degrees'.format(names[idx],
            #       np.mean(inclinations[idx])*180/np.pi))

        return eccentricities, inclinations

def plot_simulation_separate(t_data=None, y_data=None, xlabel="", ylabel="", data_labels=None):
    
    for idx in range(len(data_labels)):
        plt.figure()
        plt.plot(t_data, y_data[idx], 'b-', label=data_labels[idx])
        plt.axvline(linestyle='--', color='r')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.axis(xmin=t_data[0], xmax=t_data[-1])
        l = plt.legend(loc='upper left', bbox_to_anchor=(0., 1.0),
                ncol=3, fancybox=True, shadow=False, facecolor='black', prop={'size' : 12})
        for text in l.get_texts():
            text.set_color("white")
        plt.subplots_adjust(top=0.95, right=0.95, left=0.11, bottom=0.11)
        # plt.savefig('Logbook/'+ylabel+'_'+data_labels[idx])

def plot_simulation_all(t_data=None, y_data=None, xlabel="", ylabel="", data_labels=None, save_name=None):
    plt.figure()
    for idx in range(len(star_system.planets)):
        plt.plot(t_data, y_data[idx], '-', label=data_labels[idx])
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.axis(xmin=t_data[0], xmax=t_data[-1], ymin=0, ymax=1.3*np.max(y_data))
    l = plt.legend(loc='upper left', bbox_to_anchor=(0., 1.0),
            ncol=3, fancybox=True, shadow=False, facecolor='black', fontsize=11)
    for text in l.get_texts():
        text.set_color("white")
    plt.subplots_adjust(top=0.95, right=0.95)
    # print(save_name)
    # plt.savefig('Report/{}.png'.format(save_name))

def plot_precession_rate(t, precession_rate, ylabel, label):
    bbox_props=dict(facecolor='black', edgecolor='white', boxstyle='round,pad=0.5')
    # precession_rates = self.get_ascending_node_precession_rates(B, inclinations, p_list, q_list)
    fig, ax = plt.subplots(1)
    # ax.patch.set_facecolor('white')
    # ax.patch.set_alpha(0)
    plt.plot((t/10**6), precession_rate, color='b', label=label)
    # plt.axhline(5.462, 0, 1, color='k', linestyle='--')
    plt.axhline(np.mean(precession_rate), 0, 1, color='k', linestyle='--')
    ax.text(0.75, 0.94, 'Mean precession rate = {:.2f}'.format(np.real(np.mean(precession_rate))), ha='center', va='center', transform=ax.transAxes, bbox=bbox_props)
    ax.text(0.085, 0.94, 'Mercury', ha='center', va='center', transform=ax.transAxes, bbox=bbox_props)
    plt.xlabel('Time (Myr)')
    plt.ylabel(ylabel, labelpad=15)
    plt.axis(xmin=t[0]/10**6, xmax=t[-1]/10**6, ymin=470, ymax=660)
    # l = plt.legend(loc='upper left', bbox_to_anchor=(0., 1.0),
    #         ncol=3, fancybox=True, shadow=False, facecolor='black',
    #         handlelength=0, handletextpad=0)
    # for text in l.get_texts():
    #     text.set_color("white")
    # fig.savefig('Presentation/Mercury_no_GR.pdf', facecolor='none')

def plot_eccentricity(t, eccentricity, label):
    plt.figure()
    plt.plot((t/10**6), eccentricity, 'b', linewidth=1, label=label)
    plt.xlabel('Time (Myr)')
    plt.ylabel(r"Eccentricity")
    plt.axis(xmin=t[0]/10**6, xmax=t[-1]/10**6)
    l = plt.legend(loc='upper left', bbox_to_anchor=(0., 1.0),
            ncol=3, fancybox=True, shadow=False, facecolor='black',
            handlelength=0, handletextpad=0)
    for text in l.get_texts():
        text.set_color("white")

if __name__ == "__main__":
    names = pd.read_csv('SolarSystemData/solar_system.csv')['Name']
    # for n in names:
    # print(n)
    star_system = solar_System('SolarSystemData/Sun.csv', 'SolarSystemData/js_system.csv')
    # star_system = solar_System(1., 1., 'SolarSystemData/'+n+'.csv')
    # t = np.linspace(-10*10**6, 10*10**6, 10000)+0j
    # t = np.linspace(0, 5*10**6, 10000)+0j
    # t = np.linspace(0., 1001, 1001)+0j
    t = np.linspace(-100000, 100000, 3508)+0j
    # t = np.linspace(-2, 3, 608)+0j
    eccentricities, inclinations = star_system.simulate(t=t, plot_orbit=False, plot=True, separate=False)
    # print(star_system)
    plt.show()

# WHY DOES VENUS AFFECT EARTHS ORBIT AT THE BEGINNING??????