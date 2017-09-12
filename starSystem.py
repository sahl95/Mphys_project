'''
Sahl Rowther
'''

import pandas as pd
import numpy as np
from scipy import integrate
from planet import planet

class starSystem():

    def __init__(self, starMass, planet_data_file):
        self.star_mass = starMass
        self.planets = self.addPlanets(planet_data_file)

    def addPlanets(self, planet_data_file):
        planets = pd.read_csv(planet_data_file)
        planet_list = []
        for p in range(len(planets)):
            planet_list.append(planet(*planets.ix[p]))

        return planet_list

    def print_planets(self):
        for p in self.planets:
            p.toString()

    def laplace_coeff(self):
        l_coeff = np.zeros(len(self.planets))
        for l, a_planet in enumerate(self.planets):
            l_coeff[l] = integrate.quad(laplace_coeff_integral_function, 0, 2*np.pi, args=(a_planet,))[0]
        return 1/np.pi*l_coeff

    def matrix_B(self):
        s = 0

def laplace_coeff_integral_function(psi, a_planet):
    alpha = a_planet.b/a_planet.a
    return np.cos(psi)/(1+alpha**2-(2*alpha*np.cos(psi)))**(3/2)
    

if __name__ == "__main__":
    star_system = starSystem(0.86, 'Planets.csv')
    print(star_system.laplace_coeff())
    # print(laplace_coeff_integral_function(1, star_system.planets[2]))
    star_system.print_planets()
