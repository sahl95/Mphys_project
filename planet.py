'''
Object to store planet orbital properties.
'''

import pandas as pd
import numpy as np

class planet():
    """
    Object to store planet orbital properties.

    Args:
        Name (str): The name of the planet.
        a (float): The semi-major axis in AU.
        e (float): The eccentricity of the orbit.
        i (float): The inclination of the orbit in degrees.
        Omega (float): The longitude of the ascending node in degrees.
        pi (float): The longitude of the periapsis in degrees.
        Mass (float): The mass of the planet in Earth masses.
        r (float): The radius of the planet in meters.
        n (float): The orbital angular frequency in degrees per year.
        k (float): The Love number of degree 2 of the planet.
        Q (float): The tidal quality factor of the planet.
    """
    def __init__(self, Name="", Period=None, e=None, a=None, i=None, Omega=None, pi=None, l=None,
                 Mass=None, n=None, k=None, Q=None, r=None):
        self.Name = Name
        self.period = Period
        self.e = e
        self.a = a
        self.i = i
        self.Omega = Omega
        self.pi = pi
        self.l = l 
        self.Mass = Mass
        self.r = r
        self.n = n
        self.k = k
        self.Q = Q
        self.units = {'a' : 'AU', 'mass' : 'M_EARTH', 'period' : 'days', 'i' : 'degrees', 'omega' : 'degrees',
                      'pi' : 'degrees', 'n' : 'degrees yr^(-1)', 'l' : 'degrees', 'r' : 'm', }
        
        if np.isnan(self.i) or self.i == None: self.i = 0
        if np.isnan(self.pi) or self.pi == None: self.pi = 0
        if self.Omega == None: self.Omega = 0

    def __str__(self):
        unit_keys = list(self.units.keys())
        # print(unit_keys)
        output = ''
        for attr in self.__dict__:
            if attr is not 'units':
                if self.__dict__[attr] is not None:
                    if attr in unit_keys:
                        output += '{} : {:.5g} {}\n'.format(attr, self.__dict__[attr], self.units[attr])
                    else:
                        if attr == 'name' or attr == 'Name':
                            output += '{} : {}\n'.format(attr, self.__dict__[attr])
                        else:
                            output += '{} : {:.5g}\n'.format(attr, self.__dict__[attr])
        return output

# def print_kwargs(**kwargs):
#     for a in kwargs:
#         print(a, kwargs[a])

if __name__ == "__main__":
    planets = pd.read_csv('SolarSystemData/solar_system.csv')
    planets = pd.read_csv('Exoplanets_data/HD217107.csv')
    idx = 0
    planet_b = planet(**planets.ix[idx])
    # planet_b.mass *= 6*10**24; planet_b.update_unit('mass', 'kg')
    print(planet_b)
