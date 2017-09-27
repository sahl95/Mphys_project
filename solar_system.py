import pandas as pd
import numpy as np
from scipy import integrate
import scipy.linalg
from scipy.optimize import fsolve
from sympy import symbols, Matrix, linsolve, diag
import matplotlib.pyplot as plt
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
            j_laplace_coeff_jk = 2
            j_laplace_coeff_jj = 1
            front_factor = -1
            J2_correction = (((3/2)*J2*(R/a)**2)-((9/8)*(J2**2)*(R/a)**4)-((15/4)*J4*(R/a)**4))

        if matrix_id == 'B':
            j_laplace_coeff_jk = j_laplace_coeff_jj = 1
            front_factor = 1
            J2_correction = (((3/2)*J2*(R/a)**2)-((27/8)*(J2**2)*(R/a)**4)-((15/4)*J4*(R/a)**4))

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

    def solve_property(self, eigenvectors, initial_condition):
        n = len(self.planets)
        aug = Matrix(np.zeros([n, n+1]))
        aug[:, :n] = eigenvectors
        aug[:, n] = initial_condition

        result = linsolve(aug, *symbols('x0:'+str(n)))

        answers = np.zeros(n)
        for ans in result:
            for d, answer in enumerate(ans):
                answers[d] = answer

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
        #     print(S[i], beta[i]*180/np.pi)
        #     print(T[i], gamma[i]*180/np.pi)

        # print()
        return S, beta, T, gamma
        
    def get_eccentricity(self, scaled_eigenvector_of_A, eigenvalue_of_A, beta, t):
        n = len(self.planets)
        e = scaled_eigenvector_of_A
        beta *= 180/np.pi
        g = eigenvalue_of_A

        eccentricities = []
        for j in range(n):
            h, k = np.zeros_like(t), np.zeros_like(t)
            for i in range(n):
                h += e[j, i]*np.sin((g[i]*t+beta[i])*np.pi/180)
                k += e[j, i]*np.cos((g[i]*t+beta[i])*np.pi/180)
            eccentricities.append(np.sqrt(h**2+k**2))

        return np.array(eccentricities)

    def get_inclination(self, scaled_eigenvector_of_B, eigenvalue_of_B, gamma, t):
        n = len(self.planets)
        I = scaled_eigenvector_of_B
        gamma *= 180/np.pi
        f = eigenvalue_of_B

        inclinations = []
        for j in range(n):
            t1 = time.clock()
            p, q = np.zeros_like(t), np.zeros_like(t)
            for i in range(n):
                p += I[j, i]*np.sin((f[i]*t+gamma[i])*np.pi/180)
                q += I[j, i]*np.cos((f[i]*t+gamma[i])*np.pi/180)
            inclinations.append(np.sqrt(p**2+q**2))

        return np.array(inclinations)

    def simulate(self, t=np.linspace(-100000, 100000, 300), plot=False, separate=True):
        A, B = [star_system.frequency_matrix(matrix_id=mat_id, J2=0., J4=0.) for mat_id in ['A', 'B']]
        g, x, f, y = *np.linalg.eig(A), *np.linalg.eig(B)
        S, beta, T, gamma = self.find_all_scaling_factor_and_phase(x, y)

        # print(S*x)
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
    t = np.linspace(-5*10**6, 5*10**6, 15000)
    # t = np.linspace(-100000, 100000, 300)
    eccentricities, inclinations = star_system.simulate(t=t, plot=True, separate=True)

    plt.show()

