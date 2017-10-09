'''
Sahl Rowther
'''

import pandas as pd

class planet():

    def __init__(self, Name="", Period=None, e=None, a=None, i=None, Omega=None, omega_bar=None, l=None,
                 Mass=None, n=None, k=None, Q=None, r=None):
        # print(kwargs)
        self.name = Name
        self.period = Period
        self.e = e
        self.a = a
        self.i = i
        self.omega = Omega
        self.omega_bar = omega_bar
        self.l = l 
        # self.z0 = self.i*np.exp(1j*self.omega)
        self.mass = Mass
        self.r = r
        self.n = n
        self.k = k
        self.Q = Q
        self.units = {'a' : 'AU', 'mass' : 'M_EARTH', 'period' : 'days', 'i' : 'degrees', 'omega' : 'degrees', 'omega_bar' : 'degrees', 'n' : 'degrees yr^(-1)'}
        # print(self.units)

    def update_unit(self, attribute, unit_to_update):
        """
        Updates the units post conversion.

        Args:
            attribute (str): Name of planet attribute.
            unit_to_update (str): The name of the new units.
        """
        self.units[attribute] = unit_to_update

    def toString(self):
        unit_keys = list(self.units.keys())
        # print(unit_keys)
        for attr in self.__dict__:
            if attr is not 'units':
                if self.__dict__[attr] is not None:
                    if attr in unit_keys:
                        print('{} : {} {}'.format(attr, self.__dict__[attr], self.units[attr]))
                    else:
                        print('{} : {}'.format(attr, self.__dict__[attr]))
        print()



'''
dict containing units in function that updates dict of units.
'''

# def print_kwargs(**kwargs):
#     for a in kwargs:
#         print(a, kwargs[a])

if __name__ == "__main__":
    planets = pd.read_csv('solar_system.csv')
    idx = 2
    planet_b = planet(**planets.ix[idx])
    # planet_b.mass *= 6*10**24; planet_b.update_unit('mass', 'kg')
    planet_b.toString()
