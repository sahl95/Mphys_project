'''
Sahl Rowther
'''

import pandas as pd
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from planet import planet
from unitConverter import *

class starSystem():

    def __init__(self, starMass, planet_data_file):
        self.star_mass = starMass
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

    def get_property_all_planets(self, property_name):
        """
        Returns an array containing a specific property of all planets.

        Args:
            property_name (str): The name of the property. Can be one of *a*,
                 *b*, *e*, *period*, *mass*.
        """
        property_list = np.zeros(len(self.planets))
        for idx, p in enumerate(self.planets):
            property_list[idx] = p.__dict__[property_name]

        return property_list

    def laplace_coeff(self, alpha):
        return 1/np.pi*integrate.quad(laplace_coeff_integral_function, 0, 2*np.pi, args=(alpha,))[0]

    def matrix_B_eigenmodes(self):
        G_const = 6.6738*10**(-11)
        a = AU_to_m(self.get_property_all_planets('a'))
        M_star_kg = mSun_to_kg(self.star_mass)
        n = np.sqrt(G_const*M_star_kg/a**3)

        m = mEarth_to_kg(self.get_property_all_planets('mass'))
        # print(m)

        n_planets = len(self.planets)
        B = np.zeros([n_planets, n_planets])

        for j in range(n_planets):
            for k in range(n_planets):
                # print(j, k)
                if j != k:
                    alpha_jk = a[j]/a[k]
                    if alpha_jk > 1:
                        alpha_jk = alpha_jk**(-1)
                    laplace_coeff = self.laplace_coeff(alpha_jk)
                    alpha_jk_bar = np.where(a[k] < a[j], 1, alpha_jk)
                    # print(laplace_coeff)
                    # print(a[j], a[k], ' | ', alpha_jk, a[k] < a[j], alpha_jk_bar)
                    B[j, k] = (n[j]/4)*(m[k]/M_star_kg)*alpha_jk*alpha_jk_bar*laplace_coeff
                else:
                    for kk in range(n_planets):
                        if kk != j:
                            alpha_jj = a[j]/a[kk]
                            if alpha_jj > 1:
                                alpha_jj = alpha_jj**(-1)
                            laplace_coeff = self.laplace_coeff(alpha_jj)
                            alpha_jj_bar = np.where(a[kk] < a[j], 1, alpha_jj)
                            B[j, k] += (m[kk]/M_star_kg)*alpha_jj*alpha_jj_bar*laplace_coeff
                    B[j, k] *= -(n[j]/4)
                # print(j, k)
        # print(B, '\n')
        eigenvalues, eigenvectors = np.linalg.eig(B)
        # print(eigenvalues, '\n')
        # print(eigenvectors)
        # print(np.sum(eigenvectors[:, 0]**2))
        return eigenvalues, eigenvectors

    def z_vector(self, t_vector, eigenvalues, eigenvectors):
        n_planets = len(self.planets)
        z = [np.zeros(len(t_vector), dtype='complex128') for i in range(n_planets)]
        for j in range(n_planets):
            for k in range(n_planets):
                z[j] += eigenvectors[j, k]*np.exp(1j*eigenvalues[k]*t)

        return z

    def mutual_inclination(self, z, j, k):
        #pylint: disable=maybe-no-member
        # print(z[j]*np.conjugate(z[j])+z[k]*np.conjugate(z[k])-(z[j]*np.conjugate(z[k])+z[k]*np.conjugate(z[j])))
        return np.sqrt(z[j]*np.conjugate(z[j])+z[k]*np.conjugate(z[k])-(z[j]*np.conjugate(z[k])+z[k]*np.conjugate(z[j])))


def laplace_coeff_integral_function(psi, alpha):
    return np.cos(psi)/(1+alpha**2-(2*alpha*np.cos(psi)))**(3/2)

if __name__ == "__main__":
    star_system = starSystem(0.86, 'Planets.csv')
    f, beta = star_system.matrix_B_eigenmodes()
    
    t = np.linspace(0, 20000, 1000)*365*24*3600
    z = star_system.z_vector(t, f, beta)
    neta = star_system.mutual_inclination(z, 0, 2)

    plt.figure()
    plt.plot(t/(365*24*3600), np.sin(neta), '-')
    plt.show()
    # print(star_system.laplace_coeff())
    # print(laplace_coeff_integral_function(1, star_system.planets[2]))
    # star_system.print_planets()
