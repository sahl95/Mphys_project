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

times, timestep = np.linspace(0, 1.25*10**(4), 457, retstep=True)
times = np.array(times, dtype='complex128')
eccs = simulate(star_sys, times, plot=False, plot_orbit=False, save=False, folder_name='Exoplanets_data/Kepler-11')

times = np.real(times)

n_body_data = glob.glob('Exoplanets_data/Kepler-11/*_nbody.csv')
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

def find_frequency(times, dt, eccentricity):
    Fs = 1+1/dt;  # sampling rate
    Ts = 1.0/Fs; # sampling interval
    t = times # time vector
    # print(max(t))
    y = eccentricity

    thresh = 0

    n = len(y) # length of the signal
    k = np.arange(n)
    T = n/Fs
    frq = k/T # two sides frequency range
    frq = frq[range(int(n/2))]*n#*max(t)/(2*np.pi) # one side frequency range

    Y = np.fft.fft(y)/n # fft computing and normalization
    Y = abs(Y[range(int(n/2))])

    max_frq = frq[np.max(Y[frq>thresh])==Y]
    # print(max(t))
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(t,y)
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Eccentricity')
    ax[0].axis(xmax=max(t), xmin=0)

    ax[1].plot(frq[frq>thresh], abs(Y[frq>thresh]),'r') # plotting the spectrum
    # print(np.max([max_frq+10, 100]))
    # print([np.max(Y[frq>thresh])+10, 100])
    ax[1].axis(xmax=np.max([max_frq*1.1, 100]), xmin=0)
    ax[1].set_xlabel('Freq (Hz)')
    ax[1].set_ylabel('|fft(eccentricity)|')

    # print(np.where(np.max(Y[frq>thresh])==Y))
    print(max_frq)

# from scipy import signal

# xs = times
# data = eccs[0]
# peakind = signal.find_peaks_cwt(data, np.arange(1,10), max_distances=0.7*xs)
# print(peakind, xs[peakind], data[peakind])

# plt.figure()
# plt.plot(xs, data)
# plt.plot(xs[peakind], data[peakind], 'o')

t, dt = np.linspace(0, 1, 150, retstep=True)
# y = 1-np.sin(2*np.pi*7.5*t+np.pi/2)
y = np.sin(2*np.pi*5*t+np.pi/2)

# find_frequency(t, dt, y)

# print(timestep)

idx = 0

t, timestep = np.linspace(0, max(times), 457, retstep=True)
find_frequency(t, timestep, eccs[idx])

t, timestep = np.linspace(0, max(time_nbody[idx]), len(time_nbody[idx]), retstep=True)
find_frequency(t, timestep, ecc_n_body[idx])
print(max(time_nbody[1]))
print(len(time_nbody[0]))

plt.show()
# print(star_sys)


# Interesting cases:
# 1) Kepler 11 b, e
# 2) Kepler 51 b - Comparing 50000 with 100000 years
# 3) Wolf 1061 b