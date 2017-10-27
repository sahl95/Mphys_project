import pandas as pd
import numpy as np
import numpy.ma as ma
from scipy import integrate
import scipy.linalg
from scipy.optimize import fsolve
from sympy import symbols, Matrix, linsolve, diag
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from planet import planet
from solar_system import solar_System, real_scaling_factor_and_phase

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

class test_particle():

    def __init__(self, star_data_file, planet_data_file):
        star_data = pd.read_csv(star_data_file)
        self.star_mass = star_data['star_mass'][0]
        self.star_radius = star_data['star_radius'][0]
        self.planets = self.add_planets(planet_data_file)

        self.a_particle = 0
        self.n_particle = 0
        self.e_particle = 0
        self.pi_particle = 0

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

    def freq_vector(self, matrix_id):
        M = M_SUN*self.star_mass
        R = R_SUN*self.star_radius/AU
        m = M_EARTH*self.get_property_all_planets('Mass')
        n = self.get_property_all_planets('n')*np.pi/180
        a = self.get_property_all_planets('a')
        e = self.get_property_all_planets('e')

        a_particle, n_particle = self.a_particle, self.n_particle

        f_mat = np.zeros(len(self.planets), dtype=complex)

        if matrix_id == 'A':
            j_laplace_coeff_j = 2
            front_factor = -1
            # gr_correction = 3*(a)**2*(n)**2/(LIGHT_SPD**2*(1+m/M))*1/(1-e**2)

        if matrix_id == 'B':
            j_laplace_coeff_j = 1
            front_factor = 1
            # gr_correction = np.zeros(n_planets)

        for j in range(0, len(self.planets)):
            alpha_j = a[j]/a_particle
            if alpha_j > 1:
                alpha_j = alpha_j**(-1)
            laplace_coeff = calculate_laplace_coeff(alpha_j, j_laplace_coeff_j, 3/2)
            alpha_j_bar = np.where(a_particle < a[j], 1, alpha_j)
            f_mat[j] = front_factor*(n_particle/4)*(m[j]/M)*alpha_j*alpha_j_bar*laplace_coeff
    
        return f_mat

    def A_particle(self):
        M = M_SUN*self.star_mass
        R = R_SUN*self.star_radius/AU
        m = M_EARTH*self.get_property_all_planets('Mass')
        a = self.get_property_all_planets('a')
        e = self.get_property_all_planets('e')

        a_particle, n_particle = self.a_particle, self.n_particle

        A = 0
        for j in range(0, len(self.planets)):
            alpha_j = a[j]/a_particle
            if alpha_j > 1:
                alpha_j = alpha_j**(-1)
            laplace_coeff = calculate_laplace_coeff(alpha_j, 2, 3/2)
            alpha_j_bar = np.where(a_particle < a[j], 1, a_particle/a[j])
            A += (1/4)*(m[j]/M)*alpha_j*alpha_j_bar*laplace_coeff
        A *= n_particle

        return A

    def B_particle(self):
        M = M_SUN*self.star_mass
        R = R_SUN*self.star_radius/AU
        m = M_EARTH*self.get_property_all_planets('Mass')
        a = self.get_property_all_planets('a')
        e = self.get_property_all_planets('e')

        a_particle, n_particle = self.a_particle, self.n_particle

        B = 0
        for j in range(1, len(self.planets)):
            alpha_j = a[j]/a_particle
            if alpha_j > 1:
                alpha_j = alpha_j**(-1)
            laplace_coeff = calculate_laplace_coeff(alpha_j, 1, 3/2)
            alpha_j_bar = np.where(a_particle < a[j], 1, alpha_j)
            B += (1/4)*(m[j]/M)*alpha_j*alpha_j_bar*laplace_coeff
        B *= -n_particle

        return B

    def get_nu(self, A_vec, e_mat):
        nu = np.zeros(len(A_vec), dtype=complex)
        for i in range(len(nu)):
            for j in range(len(nu)):
                # print(A_vec[j]*e_mat[j, i])
                nu[i] += A_vec[j]*e_mat[j, i]
        return nu

    def get_mu(self, B_vec, I_mat):
        mu = np.zeros(len(B_vec), dtype=complex)
        for i in range(len(mu)):
            for j in range(len(mu)):
                mu[i] += B_vec[j]*I_mat[j, i]
        return mu

    def get_h0(self, A, eigenvalues_A_mat, phase, nu, t):
        g = eigenvalues_A_mat

        h = np.zeros_like(t)
        for i in range(len(nu)):
            # print(nu[i]/(A-g[i]), nu[i], A-g[i])
            h += (nu[i]/(A-g[i]))*np.sin((g[i]*t+phase[i]))
        return -h

    def get_k0(self, A, eigenvalues_A_mat, phase, nu, t):
        g = eigenvalues_A_mat

        k = np.zeros_like(t)
        for i in range(len(nu)):
            k += (nu[i]/(A-g[i]))*np.cos((g[i]*t+phase[i]))
        return -k

    def simulate(self, e_mat, I_mat, t):

        A_vec = self.freq_vector('A')
        B_vec = self.freq_vector('B')
        A = self.A_particle()
        B = self.B_particle()
        
        nu = self.get_nu(A_vec, e_mat)
        mu = self.get_mu(B_vec, I_mat)

        h0 = self.get_h0(A, g, beta, nu, t)
        k0 = self.get_k0(A, g, beta, nu, t)

        e_forced = np.real(np.sqrt(h0*np.conjugate(h0)+k0*np.conjugate(k0)))
        # print(e_forced[0])
        h = self.e_particle*np.sin(self.pi_particle)
        k = self.e_particle*np.cos(self.pi_particle)
        e_free_real, e_free_imag, phase_real, phase_imag = fsolve(real_scaling_factor_and_phase, (1, 1, -1, -1), args=(h-h0[0], k-k0[0],))

        e_free, phase = e_free_real+1j*e_free_imag, phase_real+1j*phase_imag
        # print(e_free)
        h = e_free*np.sin(A*t+phase)+h0
        k = e_free*np.cos(A*t+phase)+k0
        # print

        ecc = np.real(np.sqrt(h*np.conjugate(h)+k*np.conjugate(k)))
        print(np.max(ecc), np.min(ecc))
        # print(np.min(1-ecc))

        plt.figure()
        plt.semilogx(t, 5*(1-ecc))
        plt.xlabel('Time (yrs)')
        plt.ylabel('a(1-e)')
        plt.show()

if __name__ == "__main__":
    t = np.linspace(0, 1*10**10, 30000)+0j

    # folder = 'SolarSystemData/'
    folder = 'KR_paper_tests/'
    star_id = '1st_test_2'
    # star_id = ''
    folder_name = folder+star_id

    star_sys = solar_System(folder_name+'/star.csv', folder_name+'/planets.csv')
    # star_sys = solar_System(folder_name+'/Sun.csv', folder_name+'/js_system.csv')
    A_mat, B_mat = [star_sys.frequency_matrix(matrix_id=mat_id, J2=-6.84*10**(-7), J4=2.8*10**(-12)) for mat_id in ['A', 'B']]
    g, x, f, y = *np.linalg.eig(A_mat), *np.linalg.eig(B_mat)
    S, beta, T, gamma = star_sys.find_all_scaling_factor_and_phase(x, y)
    # print(beta*180/np.pi)
    # print(A_mat*180/np.pi)
    e, I = S*x, T*y

    # star_sys = test_particle(folder_name+'/Sun.csv', folder_name+'/js_system.csv')
    star_sys = test_particle(folder_name+'/star.csv', folder_name+'/planets.csv')

    star_sys.a_particle = 5
    star_sys.n_particle = np.sqrt(G_CONST*star_sys.star_mass*M_SUN/(star_sys.a_particle*AU)**3)*365*24*3600
    star_sys.e_particle = 0.1
    star_sys.pi_particle = np.pi

    star_sys.simulate(e, I, t)
        
    # print(h0[0], star_sys.e_particle*np.sin(star_sys.pi_particle))
    # print(k0)

        


    # h0 = -np.sum((nu/(A_vec-g))*np.sin())

    # print(star_sys.freq_vector('A'))
    # print(star_sys.A_particle())
    # print(star_sys.freq_vector('B'))
    # print(star_sys.B_particle())
    # print(star_sys)
