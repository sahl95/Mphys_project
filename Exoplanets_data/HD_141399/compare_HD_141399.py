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
    star_sys = solar_System('Exoplanets_data/HD_141399/star.csv', 'Exoplanets_data/HD_141399/planets.csv')

times = np.linspace(0, 5*10**(4), 1234)+0j
    # times = np.linspace(-0, .1, 500)+0j
    # times = np.linspace(10**6, 10**10, 10000)+0j
    # times = np.logspace(6, 10, 10000)+0j
eccs = simulate(star_sys, times, plot=False, plot_orbit=False, save=False, folder_name='Exoplanets_data/HD_141399')

times = np.real(times)
try:
    df = pd.read_csv('Exoplanets_data/HD_141399'+'/b_nbody.csv')
except:
    df = pd.read_csv('b_nbody.csv')

f, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, figsize=(8,7))
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
    df = pd.read_csv('Exoplanets_data/HD_141399'+'/c_nbody.csv')
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
    df = pd.read_csv('Exoplanets_data/HD_141399'+'/d_nbody.csv')
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
    df = pd.read_csv('Exoplanets_data/HD_141399'+'/e_nbody.csv')
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

f.subplots_adjust(hspace=0, top=0.97)
plt.show()
# print(star_sys)
