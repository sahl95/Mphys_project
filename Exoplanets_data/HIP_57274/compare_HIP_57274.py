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
    star_sys = solar_System('Exoplanets_data/HIP_57274/star.csv', 'Exoplanets_data/HIP_57274/planets.csv')

times, timestep = np.linspace(0, 1*10**(5), 1234, retstep=True)
times = np.array(times, dtype='complex128')
eccs = simulate(star_sys, times, plot=False, plot_orbit=False, save=False, folder_name='Exoplanets_data/HIP_57274')

times = np.real(times)

n_body_data = glob.glob('Exoplanets_data/HIP_57274/*_nbody.csv')
# print(n_body_data)

n = len(eccs)

f, axes = plt.subplots(n, sharex=True, figsize=(8,7))
ax0 = f.add_subplot(111, frame_on=False)   # creating a single axes
ax0.set_xticks([])
ax0.set_yticks([])
ax0.set_ylabel('Eccentricity', labelpad=35)
ax0.set_xlabel('Time (yrs)', labelpad=25)

time_nbody, ecc_n_body = [], []

for idx in range(n):
    df = pd.read_csv(n_body_data[idx])
    t, e = np.array(df.Time), np.array(df.e)
    t, e = t[t<max(times)], e[t<max(times)]
    time_nbody.append(t), ecc_n_body.append(e)

    axes[idx].plot(t, e, 'k--', label=n_body_data[idx].split('/')[-1].split('_')[0], linewidth=1)
    axes[idx].plot(times, eccs[idx], 'b')
    # axes[0].set_xlabel('Time')
    # axes[0].set_ylabel(r"Eccentricity")
    axes[idx].axis(xmin=t[0], xmax=times[-1])
    l = axes[idx].legend(loc='upper left',
            ncol=3, fancybox=True, shadow=False, facecolor='black',
            handlelength=0, handletextpad=0)
    for text in l.get_texts():
        text.set_color("white")

f.subplots_adjust(hspace=0, top=0.97)
plt.show()