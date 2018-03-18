import sys
import os
split_path = os.path.dirname(__file__).split('/')[:-2]
path = ''
for p in split_path:
    path += '/'+p
path += '/'
sys.path.append(path)

import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from solar_system import solar_System
from exoplanet_simulation import simulate

try:
    star_sys = solar_System('star.csv', 'planets.csv')
except:
    star_sys = solar_System('Exoplanets_data/Kepler-11/star.csv', 'Exoplanets_data/Kepler-11/planets.csv')

times = np.linspace(0, 1.25*10**(4), 1234)+0j
    # times = np.linspace(-0, .1, 500)+0j
    # times = np.linspace(10**6, 10**10, 10000)+0j
    # times = np.logspace(6, 10, 10000)+0j
eccs = simulate(star_sys, times, plot=False, plot_orbit=False, save=False, folder_name='Exoplanets_data/Kepler-11')

times = np.real(times)
try:
    df = pd.read_csv('Exoplanets_data/Kepler-11'+'/b_nbody.csv')
except:
    df = pd.read_csv('b_nbody.csv')

f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, sharex=True, figsize=(8,7))
ax0 = f.add_subplot(111, frame_on=False)   # creating a single axes
ax0.set_xticks([])
ax0.set_yticks([])
ax0.set_ylabel('Eccentricity', labelpad=35)
ax0.set_xlabel('Time (yrs)', labelpad=25)

t, e = np.array(df.Time), np.array(df.e)
ax1.plot(t, e, 'k--', label='b', linewidth=1)
ax1.plot(times, eccs[0], 'b')
# ax1.set_xlabel('Time')
# ax1.set_ylabel(r"Eccentricity")
ax1.axis(xmin=t[0], xmax=times[-1])
l = ax1.legend(loc='upper left',
        ncol=3, fancybox=True, shadow=False, facecolor='black',
        handlelength=0, handletextpad=0)
for text in l.get_texts():
    text.set_color("white")

try:
    df = pd.read_csv('Exoplanets_data/Kepler-11'+'/c_nbody.csv')
except:
    df = pd.read_csv('c_nbody.csv')
t, e = np.array(df.Time), np.array(df.e)
# plt.figure()
ax2.plot(t, e, 'k--', label='c', linewidth=1)
ax2.plot(times, eccs[1], 'b')
# plt.xlabel('Time')
# plt.ylabel(r"Eccentricity")
ax2.axis(xmin=t[0], xmax=times[-1])
l = ax2.legend(loc='upper left',
        ncol=3, fancybox=True, shadow=False, facecolor='black',
        handlelength=0, handletextpad=0)
for text in l.get_texts():
    text.set_color("white")

try:
    df = pd.read_csv('Exoplanets_data/Kepler-11'+'/d_nbody.csv')
except:
    df = pd.read_csv('d_nbody.csv')
t, e = np.array(df.Time), np.array(df.e)
# plt.figure()
ax3.plot(t, e, 'k--', label='d', linewidth=1)
ax3.plot(times, eccs[2], 'b')
# plt.xlabel('Time')
# plt.ylabel(r"Eccentricity")
ax3.axis(xmin=t[0], xmax=times[-1])
l = ax3.legend(loc='upper left',
        ncol=3, fancybox=True, shadow=False, facecolor='black',
        handlelength=0, handletextpad=0)
for text in l.get_texts():
    text.set_color("white")

try:
    df = pd.read_csv('Exoplanets_data/Kepler-11'+'/e_nbody.csv')
except:
    df = pd.read_csv('e_nbody.csv')
t, e = np.array(df.Time), np.array(df.e)
# plt.figure()
ax4.plot(t, e, 'k--', label='e', linewidth=1)
ax4.plot(times, eccs[3], 'b')
# plt.xlabel('Time')
# plt.ylabel(r"Eccentricity")
ax4.axis(xmin=t[0], xmax=times[-1])
l = ax4.legend(loc='upper left',
        ncol=3, fancybox=True, shadow=False, facecolor='black',
        handlelength=0, handletextpad=0)
for text in l.get_texts():
    text.set_color("white")

try:
    df = pd.read_csv('Exoplanets_data/Kepler-11'+'/f_nbody.csv')
except:
    df = pd.read_csv('f_nbody.csv')
t, e = np.array(df.Time), np.array(df.e)
# plt.figure()
ax5.plot(t, e, 'k--', label='f', linewidth=1)
ax5.plot(times, eccs[4], 'b')
# plt.xlabel('Time')
# plt.ylabel(r"Eccentricity")
ax5.axis(xmin=t[0], xmax=times[-1])
l = ax5.legend(loc='upper left',
        ncol=3, fancybox=True, shadow=False, facecolor='black',
        handlelength=0, handletextpad=0)
for text in l.get_texts():
    text.set_color("white")

try:
    df = pd.read_csv('Exoplanets_data/Kepler-11'+'/g_nbody.csv')
except:
    df = pd.read_csv('g_nbody.csv')
t, e = np.array(df.Time), np.array(df.e)
# plt.figure()
ax6.plot(t, e, 'k--', label='g', linewidth=1)
ax6.plot(times, eccs[5], 'b')
# plt.xlabel('Time')
# plt.ylabel(r"Eccentricity")
ax6.axis(xmin=t[0], xmax=times[-1])
l = ax6.legend(loc='upper left',
        ncol=3, fancybox=True, shadow=False, facecolor='black',
        handlelength=0, handletextpad=0)
for text in l.get_texts():
    text.set_color("white")

f.subplots_adjust(hspace=0, top=0.97)
plt.show()
# print(star_sys)
