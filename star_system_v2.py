import pandas as pd
import numpy as np
import numpy.ma as ma
from scipy import integrate
import scipy.linalg
from scipy.optimize import fsolve
from scipy.special import ellipe
from sympy import symbols, Matrix, linsolve, diag
import matplotlib.pyplot as plt
from planet import planet

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

def f(x, *boundaries):
    s_factor, phase = x
    return [s_factor*np.sin(phase)-boundaries[0], s_factor*np.cos(phase)-boundaries[1]]

def real_scaling_factor_and_phase(x1, *boundaries):
    s_factor, phase = x1[0]+1j*x1[1], x1[2]+1j*x1[3] 
    x = [s_factor, phase]
    actual_f = f(x, *boundaries)
    return [np.real(actual_f[0]), np.imag(actual_f[0]), np.real(actual_f[1]), np.imag(actual_f[1])]

class star_System():
    """
    Stores the mass and radius of the star. Also stores the properties of the orbiting planets.

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
        output = 'Star Properties\n'
        output += '---------------\n'
        output += 'Mass : {} Msun\n'.format(self.star_mass)
        output += 'Radius : {} Rsun\n\n'.format(self.star_radius)
        output += 'Planets\n'
        output += '-------\n'
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



    def frequency_matrix(self):
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
        n_planets = len(self.planets)
        # f_mat = np.zeros([n_planets, n_planets])
        f_mat = np.zeros([n_planets, n_planets], dtype='complex128')

        j_laplace_coeff_jk, j_laplace_coeff_jj = 2, 1
        front_factor = -1
 
        gr_correction = 3*(a)**2*(n)**2/(LIGHT_SPD**2*(1+m/M))*1/(1-e**2)
        # print(M)
        # print(3*(a)**2*(n)**2/(LIGHT_SPD**2))

        for j in range(n_planets):
            for k in range(n_planets):
                if j != k:
                    alpha_jk = a[j]/a[k]
                    if alpha_jk > 1:
                        alpha_jk = alpha_jk**(-1)
                    laplace_coeff = calculate_laplace_coeff(alpha_jk, j_laplace_coeff_jk, 3/2)
                    alpha_jk_bar = np.where(a[k] < a[j], 1, alpha_jk)
                    f_mat[j, k] = front_factor*(n[j]/4)*(m[k]/(M+m[j]))*alpha_jk*alpha_jk_bar*laplace_coeff

                else:
                    for kk in range(n_planets):
                        if kk != j:
                            alpha_jj = a[j]/a[kk]
                            if alpha_jj > 1:
                                alpha_jj = alpha_jj**(-1)
                            laplace_coeff = calculate_laplace_coeff(alpha_jj, j_laplace_coeff_jj, 3/2)
                            alpha_jj_bar = np.where(a[kk] < a[j], 1, alpha_jj)
                            f_mat[j, k] += (1/4)*(m[kk]/(M+m[j]))*alpha_jj*alpha_jj_bar*laplace_coeff
                    # f_mat[j, k] += J2_correction[j]
                    # f_mat[j, k] += 1j*ecc_damp[j]
                    f_mat[j, k] += gr_correction[j]
                    f_mat[j, k] *= -front_factor*(n[j])
                    # print(J2_correction[j], ecc_damp[j], gr_correction[j])
        return f_mat#*180/np.pi

    
    def initial_conditions(self):
        """
        Calculates the initial values for the 4 equations of motion (h, k, p, and q) for each planet.

        Returns:
            * :math:`h` (*numpy array*)
            * :math:`k` (*numpy array*)
        """
        e = self.get_property_all_planets('e')
        pi = self.get_property_all_planets('pi')*np.pi/180

        h = np.array(e*np.sin(pi), dtype='complex128')
        k = np.array(e*np.cos(pi), dtype='complex128')

        return h, k

    def solve_property(self, eigenvectors, initial_conditions):
        answers = np.linalg.solve(eigenvectors, initial_conditions)
        return answers

    def find_all_scaling_factor_and_phase(self, eigenvectors_of_A):
        r"""
        Calculates the scale factors :math:`S, T` of the eigenvectors and the phases, :math:`\beta, \gamma`.

        Args:
            eigenvectors_of_A (numpy array): The normalised eigenvectors of the frequency matrix A.
        
        Returns:
            * :math:`S` (*numpy array*) - Scaling factor for the eigenvectors of A.
            * :math:`\beta` (*numpy array*) - Phase shift for :math:`h, k`
        """
        x = eigenvectors_of_A

        init_conditions = np.array(self.initial_conditions())
        h_solved = self.solve_property(x, init_conditions[0, :])
        k_solved = self.solve_property(x, init_conditions[1, :])

        n = len(self.planets)
        S, beta = np.zeros(n, dtype='complex128'), np.zeros(n, dtype='complex128')

        for i in range(n):
            # S[i], beta[i] = fsolve(scaling_factor_and_phase, (1, -1), args=(h_solved[i], k_solved[i],))
            # T[i], gamma[i] = fsolve(scaling_factor_and_phase, (-1, 1), args=(p_solved[i], q_solved[i],))
            S_real, S_img, beta_real, beta_img = fsolve(real_scaling_factor_and_phase, (1, 1, -1, -1), args=(h_solved[i], k_solved[i],))
            S[i], beta[i] = S_real+1j*S_img, beta_real+1j*beta_img
            
        return S, beta

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
            return self.get_h(**kwargs)
        if eq_id == 'k' or eq_id == 'q':
            return self.get_k(**kwargs)
    
    def get_h(self, scaled_eigenvector, eigenvalue, phase, t):
        n = len(self.planets)
        h_list = []
        for j in range(n):
            h = np.zeros_like(t)
            for i in range(n):
                h += scaled_eigenvector[j, i]*np.sin((eigenvalue[i]*t+phase[i]))
            h_list.append(h)
        return np.array(h_list)

    def get_k(self, scaled_eigenvector, eigenvalue, phase, t):
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
            #pylint: disable=maybe-no-member
            eccentricities.append(np.real(np.sqrt(h[j]*np.conjugate(h[j])+k[j]*np.conjugate(k[j]))))
        return np.array(eccentricities)

    def mean_eccentricities(self, eigenvectors):
        mean_ecc = np.zeros(len(eigenvectors))
        e_vec = np.real(eigenvectors)
        for j in range(len(eigenvectors)):
            m_squiggle = 4*(e_vec[j, 0]*e_vec[j, 1])/(e_vec[j, 0]+e_vec[j, 1])**2
            # print(m_squiggle)
            # m_squiggle = np.abs(m_squiggle)*-1
            # m_squiggle = 0.31
            e_m_squiggle = ellipe(m_squiggle)
            mean_ecc[j] = (2/np.pi)*(e_vec[j, 0]*e_vec[j, 1])*e_m_squiggle
        print(mean_ecc)


    def simulate(self, t):
        A = self.frequency_matrix() 
        g, x = np.linalg.eig(A)
        S, beta = self.find_all_scaling_factor_and_phase(x)

        kwargs = {'scaled_eigenvector' : S*x, 'eigenvalue' : g, 'phase' : beta,
                  't' : t}
        h_list = self.components_of_ecc_inc(**kwargs, eq_id='h')
        k_list = self.components_of_ecc_inc(**kwargs, eq_id='k')

        eccentricities = self.get_eccentricity(h_list, k_list)

        names = self.get_property_all_planets('Name')
        plot_simulation_separate(t/10**6, eccentricities, 'Time (Myr)', 'Eccentricity', names)

        mean_ecc = self.mean_eccentricities(x)

        for e, ecc in enumerate(eccentricities):
            print(self.planets[e].Name, np.mean(ecc), np.max(ecc), np.min(ecc))
        print()
        Sx = np.abs(S*x)
        # print(Sx)
        # for e, ecc in enumerate(eccentricities):
        #     max_e = np.max([np.abs(Sx[e, 0]+Sx[e, 1]), np.abs(Sx[e, 0]-Sx[e, 1])])
        #     min_e = np.min([np.abs(Sx[e, 0]+Sx[e, 1]), np.abs(Sx[e, 0]-Sx[e, 1])])
        #     print(np.mean(ecc), max_e, min_e)

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

if __name__ == "__main__":
    star_name = 'HD_217107'
    star_sys = star_System('StarSystemData/'+star_name+'/star.csv', 'StarSystemData/'+star_name+'/planets.csv')
    # A, B = [star_sys.frequency_matrix(matrix_id=mat_id, J2=-6.84*10**(-7), J4=2.8*10**(-12)) for mat_id in ['A', 'B']]
    t = np.linspace(0, 10*10**4, 5000)+0j
    star_sys.simulate(t)
    # print(star_sys)
    # plt.show()
