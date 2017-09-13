"""
Converts from astronomical units to SI units and vice versa.
"""

M_SUN = 1.9885*10**30
M_EARTH = 5.9726*10**24
AU = 149597870700


def mSun_to_kg(value):
    return value*M_SUN

def kg_to_mSun(value):
    return value/M_SUN

def mEarth_to_kg(value):
    return value*M_EARTH

def kg_to_mEarth(value):
    return value/M_EARTH

def AU_to_m(value):
    return value*AU

def m_to_AU(value):
    return value/AU
