'''
Sahl Rowther
'''

import numpy as np
import pandas as pd

class planet():

    def __init__(self, name, period, e, a, b, mass):
        self.name = name
        self.period = period
        self.e = e
        self.a = a
        self.b = b
        self.mass = mass

    def toString(self):
        for attr in self.__dict__:
            print('{} : {}'.format(attr, self.__dict__[attr]))
        print()

if __name__ == "__main__":
    planets = pd.read_csv('Planets.csv')
    # print([planets.ix[0]])
    idx = 1
    planet_b = planet(*planets.ix[idx])
    planet_b.toString()
