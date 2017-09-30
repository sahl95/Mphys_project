import pandas as pd
import numpy as np
import numpy.ma as ma
from scipy import integrate
import scipy.linalg
from scipy.optimize import fsolve
from scipy.interpolate import interp1d, UnivariateSpline
from sympy import symbols, Matrix, linsolve, diag
import matplotlib.pyplot as plt
from matplotlib import rc
import time
from planet import planet
from unitConverter import *


def calculate_laplace_coeff(alpha, j, s):
    return integrate.quad(lambda psi, alpha, j, s: np.cos(j*psi)/(1-2*alpha*np.cos(psi)+alpha**2)**s,
                          0, 2*np.pi, args=(alpha, j, s,))[0]/np.pi

def scaling_factor_and_phase(p, *boundaries):
    s, phase = p
    return (s*np.sin(phase)-boundaries[0], s*np.cos(phase)-boundaries[1])

class solar_System():

    def __init__(self, starMass, starRadius, planet_data_file):
        self.star_mass = starMass
        self.star_radius = starRadius
        self.planets = self.addPlanets(planet_data_file)

    def addPlanets(self, planet_data_file):
        planets = pd.read_csv(planet_data_file)
        planet_list = []
        for p in range(len(planets)):
            planet_list.append(planet(**planets.ix[p]))

        return planet_list

    def print_planets(self):
        for p in self.planets:
            p.toString()

    def get_property_all_planets(self, property_name, data_type="float"):
        """
        Returns an array containing a specific property of all planets.

        Args:
            property_name (str): The name of the property. Can be one of *a*,
                 *b*, *e*, *period*, *mass*.
        """
        property_list = np.zeros(len(self.planets), dtype=data_type)
        for idx, p in enumerate(self.planets):
            property_list[idx] = p.__dict__[property_name]

        return property_list

    def frequency_matrix(self, matrix_id, J2=0, J4=0):
        M_star_kg = mSun_to_kg(self.star_mass)
        R = Rsun_to_m(self.star_radius)
        m = mEarth_to_kg(self.get_property_all_planets('mass'))
        n = self.get_property_all_planets('n')
        a = AU_to_m(self.get_property_all_planets('a'))
        n_planets = len(self.planets)
        f_mat = np.zeros([n_planets, n_planets])

        if matrix_id == 'A':
            j_laplace_coeff_jk, j_laplace_coeff_jj = 2, 1
            front_factor = -1
            J2_correction = (((3/2)*J2*(R/a)**2)-((9/8)*(J2**2)*(R/a)**4)-((15/4)*J4*(R/a)**4))
            # print(J2_correction)

        if matrix_id == 'B':
            j_laplace_coeff_jk = j_laplace_coeff_jj = 1
            front_factor = 1
            J2_correction = (((3/2)*J2*(R/a)**2)-((27/8)*(J2**2)*(R/a)**4)-((15/4)*J4*(R/a)**4))
            # print(J2_correction)

        for j in range(n_planets):
            for k in range(n_planets):
                # print(j, k)
                if j != k:
                    alpha_jk = a[j]/a[k]
                    if alpha_jk > 1:
                        alpha_jk = alpha_jk**(-1)
                    laplace_coeff = calculate_laplace_coeff(alpha_jk, j_laplace_coeff_jk, 3/2)
                    alpha_jk_bar = np.where(a[k] < a[j], 1, alpha_jk)
                    f_mat[j, k] = front_factor*(n[j]/4)*(m[k]/M_star_kg)*alpha_jk*alpha_jk_bar*laplace_coeff

                else:
                    for kk in range(n_planets):
                        if kk != j:
                            alpha_jj = a[j]/a[kk]
                            if alpha_jj > 1:
                                alpha_jj = alpha_jj**(-1)
                            laplace_coeff = calculate_laplace_coeff(alpha_jj, j_laplace_coeff_jj, 3/2)
                            alpha_jj_bar = np.where(a[kk] < a[j], 1, alpha_jj)
                            f_mat[j, k] += (1/4)*(m[kk]/M_star_kg)*alpha_jj*alpha_jj_bar*laplace_coeff
                    f_mat[j, k] += J2_correction[j]
                    f_mat[j, k] *= -front_factor*(n[j])
        return f_mat

    def initial_conditions(self):
        e = self.get_property_all_planets('e')
        omega_bar = self.get_property_all_planets('omega_bar')*np.pi/180
        i = self.get_property_all_planets('i')*np.pi/180
        omega = self.get_property_all_planets('omega')*np.pi/180

        h = e*np.sin(omega_bar)
        k = e*np.cos(omega_bar)
        p = i*np.sin(omega)
        q = i*np.cos(omega)

        return h, k, p, q

    def solve_property(self, eigenvectors, initial_conditions):
        n = len(self.planets)
        aug = Matrix(np.zeros([n, n+1]))
        aug[:, :n] = eigenvectors
        aug[:, n] = initial_conditions

        result = linsolve(aug, *symbols('x0:'+str(n)))

        answers = np.zeros(n)
        for ans in result:
            for a, answer in enumerate(ans):
                answers[a] = answer

        return answers

    def find_all_scaling_factor_and_phase(self, eigenvectors_of_A, eigenvectors_of_B):
        x, y = eigenvectors_of_A, eigenvectors_of_B

        init_conditions = np.array(star_system.initial_conditions())
        h_solved = self.solve_property(x, init_conditions[0, :])
        k_solved = self.solve_property(x, init_conditions[1, :])
        p_solved = self.solve_property(y, init_conditions[2, :])
        q_solved = self.solve_property(y, init_conditions[3, :])

        # print(h_solved)
        # print(k_solved)
        # print(p_solved)
        # print(q_solved)

        n = len(self.planets)
        S, beta = np.zeros(n), np.zeros(n)
        T, gamma = np.zeros(n), np.zeros(n)

        for i in range(n):
            S[i], beta[i] = fsolve(scaling_factor_and_phase, (1, -1), args=(h_solved[i], k_solved[i],))
            T[i], gamma[i] = fsolve(scaling_factor_and_phase, (-1, 1), args=(p_solved[i], q_solved[i],))

        return S, beta, T, gamma

    def eq_of_motion(self, scaled_eigenvector, eigenvalue, phase, t, eq_id):
        # eq_id = 'h', 'k', 'p', 'q'
        kwargs = {'scaled_eigenvector' : scaled_eigenvector, 'eigenvalue' : eigenvalue, 'phase' : phase,
                  't' : t}
        if eq_id == 'h' or eq_id == 'p':
            return self.get_h_or_p(**kwargs)
        if eq_id == 'k' or eq_id == 'q':
            return self.get_k_or_q(**kwargs)
    
    def get_h_or_p(self, scaled_eigenvector, eigenvalue, phase, t):
        n = len(self.planets)
        h_list = []
        for j in range(n):
            h = np.zeros_like(t)
            for i in range(n):
                h += scaled_eigenvector[j, i]*np.sin((eigenvalue[i]*t+phase[i])*np.pi/180)
            h_list.append(h)
        return np.array(h_list)

    def get_k_or_q(self, scaled_eigenvector, eigenvalue, phase, t):
        n = len(self.planets)
        k_list = []
        for j in range(n):
            k = np.zeros_like(t)
            for i in range(n):
                k += scaled_eigenvector[j, i]*np.cos((eigenvalue[i]*t+phase[i])*np.pi/180)
            k_list.append(k)
        return np.array(k_list)

    def disturbing_function(self, A, B, h, k, p, q):
        # plt.figure()
        # plt.plot(t, h[0])

        n = self.get_property_all_planets('n')
        a = self.get_property_all_planets('a')

        # A *= (365*24*3600*180/np.pi)
        # B *= (365*24*3600*180/np.pi)
        n_planets = len(self.planets)
        R_list = []
        for j in range(n_planets):
            R = 0
            R += 0.5*A[j, j]*(h[j]**2+k[j]**2)
            R += 0.5*B[j, j]*(p[j]**2+q[j]**2)
            for i in range(n_planets):
                if j != i:
                    R += A[j, i]*(h[j]*h[i]+k[j]*k[i])
                    R += B[j, i]*(p[j]*h[i]+q[j]*k[i])
            R_list.append(n[j]*a[j]**2*R)

        # plt.figure()
        # plt.plot(t, R_list[0])

        return R_list


    def get_eccentricity(self, scaled_eigenvector_of_A, eigenvalue_of_A, beta, t):
        n = len(self.planets)
        e = scaled_eigenvector_of_A
        beta *= 180/np.pi
        g = eigenvalue_of_A

        kwargs = {'scaled_eigenvector' : scaled_eigenvector_of_A, 'eigenvalue' : eigenvalue_of_A, 'phase' : beta,
                  't' : t}
        eccentricities = []
        h, k = self.eq_of_motion(**kwargs, eq_id='h'), self.eq_of_motion(**kwargs, eq_id='k')
        for j in range(n):  
            eccentricities.append(np.sqrt(h[j]**2+k[j]**2))
            # if j==2:
            #     # plt.figure()
            #     # plt.plot(t, k)
            #     omega_bar_h = np.arcsin(k/eccentricities[j])
            #     omega_bar_k = np.arccos(k/eccentricities[j])

            #     # spl = UnivariateSpline(t, omega_bar, k=3, s=0)
            #     # spl_dot = spl.derivative()
                
            #     plt.figure()
            #     plt.plot(t, omega_bar_h, '-')
            #     plt.figure()
            #     plt.plot(t, omega_bar_k, '-')
                # plt.plot(t, spl_dot(t), '-')
            

        return np.array(eccentricities)

    def get_inclination(self, scaled_eigenvector_of_B, eigenvalue_of_B, gamma, t):
        n = len(self.planets)
        I = scaled_eigenvector_of_B
        gamma *= 180/np.pi
        f = eigenvalue_of_B

        kwargs = {'scaled_eigenvector' : scaled_eigenvector_of_B, 'eigenvalue' : eigenvalue_of_B, 'phase' : gamma,
                  't' : t}
        inclinations = []
        p, q = self.eq_of_motion(**kwargs, eq_id='p'), self.eq_of_motion(**kwargs, eq_id='q')
        for j in range(n):
            inclinations.append(np.sqrt(p[j]**2+q[j]**2))

        return np.array(inclinations)

    def get_precession_rates(self, A, eccentricities, h_list, k_list):
        n = len(self.planets)
        d_pidot_dt_list = []
        masks = []

        for j in range(n):
            h_dot_j = A[j, j]*k_list[j]
            k_dot_j = -A[j, j]*h_list[j]

            for i in range(n):
                if i != j:
                    h_dot_j += A[j, i]*k_list[i]
                    k_dot_j -= A[j, i]*h_list[i]
            pidot_j = 3600*(k_list[j]*h_dot_j - h_list[j]*k_dot_j)/(eccentricities[j])**2
            # pidot_j = ma.masked_array(pidot_j, np.abs(pidot_j)>10).filled(0)

            d_pidot_dt_list.append(pidot_j)
        return d_pidot_dt_list

    def simulate(self, t=np.linspace(-100000, 100000, 300), plot=False, separate=True):
        A, B = [star_system.frequency_matrix(matrix_id=mat_id, J2=-6.84*10**(-7), J4=2.8*10**(-12)) for mat_id in ['A', 'B']]
        # print('\n', A, B)
        g, x, f, y = *np.linalg.eig(A), *np.linalg.eig(B)
        S, beta, T, gamma = self.find_all_scaling_factor_and_phase(x, y)

        eccentricities = self.get_eccentricity(S*x, g, beta, t)
        inclinations = self.get_inclination(T*y, f, gamma, t)*180/np.pi
        names = [self.planets[p].name for p in range(len(self.planets))]
        if plot:
            if separate:
                plot_simulation_separate(t/10**6, eccentricities, 'Time (Myr)', 'Eccentricity', names)
                plot_simulation_separate(t/10**6, inclinations, 'Time (Myr)', 'Inclination', names)
            else:
                plot_simulation_all(t/10**6, eccentricities, 'Time (Myr)', 'Eccentricity', names)
                plot_simulation_all(t/10**6, inclinations, 'Time (Myr)', 'Inclination', names)

        kwargs = {'scaled_eigenvector' : S*x, 'eigenvalue' : g, 'phase' : beta,
                  't' : t}
        h_list = self.eq_of_motion(**kwargs, eq_id='h')
        k_list = self.eq_of_motion(**kwargs, eq_id='k')
        kwargs = {'scaled_eigenvector' : S*x, 'eigenvalue' : f, 'phase' : gamma,
                    't' : t}
        p_list = self.eq_of_motion(**kwargs, eq_id='p')
        q_list = self.eq_of_motion(**kwargs, eq_id='q')

        idx = 0
        precession_rates = self.get_precession_rates(A, eccentricities, h_list, k_list)
        plt.figure()
        plt.plot((t/10**6), precession_rates[idx], linewidth=1)
        # plt.axhline(5.462, 0, 1, color='k', linestyle='--')
        plt.axhline(np.mean(precession_rates[idx]), 0, 1, color='r', linestyle='--')
        plt.xlabel('Time (Myr)')
        plt.ylabel(r"Precession rate [${}'{}'\ y^{-1}$]")

        plt.figure()
        plt.plot(t/10**6, eccentricities[idx], linewidth=1)
        # plt.axhline(5.462, 0, 1, color='k', linestyle='--')
        # plt.axhline(np.mean(precession_rates[2]), 0, 1, color='r', linestyle='--')
        plt.xlabel('Time (Myr)')
        plt.ylabel(r"Eccentricity")
        # plt.gca().invert_xaxis()

        names = [p.__dict__['name'] for p in self.planets]
        for idx in range(len(precession_rates)):
            print('Precession rate of {} = {:.2f} arcseconds per century'.format(names[idx],
                  np.mean(precession_rates[idx])*100))

        return eccentricities, inclinations

def plot_simulation_separate(t_data=None, y_data=None, xlabel="", ylabel="", data_labels=None):
    for idx in range(len(star_system.planets)):
        plt.figure()
        plt.plot(t_data, y_data[idx], '-', label=data_labels[idx])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.axis(xmin=t_data[0], xmax=t_data[-1])
        l = plt.legend(loc='upper left', bbox_to_anchor=(0., 1.0),
                ncol=3, fancybox=True, shadow=False, facecolor='black')
        for text in l.get_texts():
            text.set_color("white")

def plot_simulation_all(t_data=None, y_data=None, xlabel="", ylabel="", data_labels=None):
    plt.figure()
    for idx in range(len(star_system.planets)):
        plt.plot(t_data, y_data[idx], '-', label=data_labels[idx])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axis(xmin=t_data[0], xmax=t_data[-1], ymin=0, ymax=1.3*np.max(y_data))
    l = plt.legend(loc='upper left', bbox_to_anchor=(0., 1.0),
            ncol=3, fancybox=True, shadow=False, facecolor='black')
    for text in l.get_texts():
        text.set_color("white")




if __name__ == "__main__":
    star_system = solar_System(1., 1., 'solar_system.csv')
    # star_system.print_planets()
    # t = np.linspace(-5*10**6, 5*10**6, 15000)
    t = np.linspace(-10*10**6, 10*10**6, 3000)
    # t = np.linspace(-100000, 100000, 300)
    eccentricities, inclinations = star_system.simulate(t=t, plot=False, separate=False)

    plt.show()

