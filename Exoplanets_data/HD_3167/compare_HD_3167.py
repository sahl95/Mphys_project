import sys
sys.path.append('/Users/Sahl/Desktop/University/Year_5/Mphys_project/')

import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from solar_system import solar_System
from exoplanet_simulation import simulate

try:
    star_sys = solar_System('star.csv', 'planets.csv')
except:
    star_sys = solar_System('Exoplanets_data/HD_3167/star.csv', 'Exoplanets_data/HD_3167/planets.csv')

times = np.linspace(0, 5*10**(4), 1234)+0j
    # times = np.linspace(-0, .1, 500)+0j
    # times = np.linspace(10**6, 10**10, 10000)+0j
    # times = np.logspace(6, 10, 10000)+0j
eccs = simulate(star_sys, times, plot=False, plot_orbit=False, save=False, folder_name='Exoplanets_data/HD_3167')

times = np.real(times)
try:
    df = pd.read_csv('Exoplanets_data/HD_3167'+'/b_nbody.csv')
except:
    df = pd.read_csv('b_nbody.csv')
t, e = np.array(df.Time), np.array(df.e)
plt.figure()
plt.plot(t, e, 'k--', label='b', linewidth=1)
plt.plot(times, eccs[0], 'b')
plt.xlabel('Time')
plt.ylabel(r"Eccentricity")
plt.axis(xmin=t[0], xmax=t[-1])
l = plt.legend(loc='upper left',
        ncol=3, fancybox=True, shadow=False, facecolor='black',
        handlelength=0, handletextpad=0)
for text in l.get_texts():
    text.set_color("white")

try:
    df = pd.read_csv('Exoplanets_data/HD_3167'+'/c_nbody.csv')
except:
    df = pd.read_csv('c_nbody.csv')
t, e = np.array(df.Time), np.array(df.e)
plt.figure()
plt.plot(t, e, 'k--', label='c', linewidth=1)
plt.plot(times, eccs[1], 'b')
plt.xlabel('Time')
plt.ylabel(r"Eccentricity")
plt.axis(xmin=t[0], xmax=t[-1])
l = plt.legend(loc='upper left',
        ncol=3, fancybox=True, shadow=False, facecolor='black',
        handlelength=0, handletextpad=0)
for text in l.get_texts():
    text.set_color("white")

try:
    df = pd.read_csv('Exoplanets_data/HD_3167'+'/d_nbody.csv')
except:
    df = pd.read_csv('d_nbody.csv')
t, e = np.array(df.Time), np.array(df.e)
plt.figure()
plt.plot(t, e, 'k--', label='d', linewidth=1)
plt.plot(times, eccs[2], 'b')
plt.xlabel('Time')
plt.ylabel(r"Eccentricity")
plt.axis(xmin=t[0], xmax=t[-1])
l = plt.legend(loc='upper left',
        ncol=3, fancybox=True, shadow=False, facecolor='black',
        handlelength=0, handletextpad=0)
for text in l.get_texts():
    text.set_color("white")

plt.show()
# print(star_sys)
