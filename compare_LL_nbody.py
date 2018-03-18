import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from solar_system import solar_System
from exoplanet_simulation import simulate


# Good examples: GJ273, HD_10180, GJ273 (t=17400), tau_cet
# Mediocre examples: HD_141399 (multiple jupiter sized planets), HIP_57274
# Bad examples: 47_UMa (multiple jupiter sized planets)
# Odd cases: GJ273

star_id = 'HD_69830'
star_sys = solar_System('Exoplanets_data/{}/star.csv'.format(star_id), 'Exoplanets_data/{}/planets.csv'.format(star_id))

def get_nbody_data(max_time=None):
    n_body_data = glob.glob('Exoplanets_data/{}/*_nbody.csv'.format(star_id))
    time_nbody, ecc_n_body = [], []

    n = len(n_body_data)
    for idx in range(n):
        df = pd.read_csv(n_body_data[idx])
        t, e = np.array(df.Time), np.array(df.e)
        if max_time is not None:
            t, e = t[t<max_time], e[t<max_time]
        time_nbody.append(t), ecc_n_body.append(e)

    return time_nbody, ecc_n_body

def compare_LL_nbody_eccs(t_LL, e_LL, t_nbody, e_nbody):
    n_body_data = glob.glob('Exoplanets_data/{}/*_nbody.csv'.format(star_id))
    n = len(n_body_data)

    f, axes = plt.subplots(n, sharex=True, figsize=(8,7))
    ax0 = f.add_subplot(111, frame_on=False)   # creating a single axes
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_ylabel('Eccentricity', labelpad=60, fontsize=16)
    ax0.set_xlabel('Time (yrs)', labelpad=25, fontsize=16)

    for idx in range(n):
        axes[idx].plot(t_nbody[idx], e_nbody[idx], 'k--', label=n_body_data[idx].split('/')[-1].split('_')[0], linewidth=1)
        axes[idx].plot(t_LL, e_LL[idx], 'b')
        # axes[0].set_xlabel('Time')
        # axes[0].set_ylabel(r"Eccentricity")
        axes[idx].axis(xmin=t_nbody[0][0], xmax=times[-1])
        l = axes[idx].legend(loc='upper left',
                ncol=3, fancybox=True, shadow=False, facecolor='black',
                handlelength=0, handletextpad=0)
        for text in l.get_texts():
            text.set_color("white")
    
    f.subplots_adjust(hspace=0, top=0.97, left=0.15, right=0.96)
    plt.savefig('Report/GJ832.pdf')

def find_frequency(times, dt, eccentricity):
    Fs = 1+1/dt;  # sampling rate
    Ts = 1.0/Fs; # sampling interval
    t = times # time vector
    y = eccentricity

    thresh = 0

    n = len(y) # length of the signal
    k = np.arange(n)
    T = n/Fs
    frq = k/T # two sides frequency range
    frq = frq[range(int(n/2))]*n # one side frequency range

    Y = np.fft.fft(y)/n # fft computing and normalization
    Y = abs(Y[range(int(n/2))])

    y_inds = Y.argsort()
    sorted_Y = Y[y_inds[::-1]]
    sorted_Freq = frq[y_inds[::-1]]

    idx = 5
    # print('Amplitudes - ', sorted_Y[1:idx], '\nFrequencies - ', sorted_Freq[1:idx])
    # print('-----')
    max_frq = frq[np.max(Y[frq>thresh])==Y][0]
    max_amp = np.max(Y[frq>thresh])
    # print(max(t), n)
    # print(max_frq)

    return frq[frq>thresh], abs(Y[frq>thresh]), max_frq, max_amp

def plot_all_cycles():
    n_body_data = glob.glob('Exoplanets_data/{}/*_nbody.csv'.format(star_id))
    n = len(n_body_data)

    f, axes = plt.subplots(n, sharex=True, figsize=(8,7))
    ax0 = f.add_subplot(111, frame_on=False)   # creating a single axes
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_ylabel('FFT(Eccentricity)', labelpad=65, fontsize=16)
    ax0.set_xlabel('Number of Cycles', labelpad=25, fontsize=16)
    for idx in range(n):
        # print('Eccentricity (L-L):   max - {:.4f} | min - {:.4f}'.format(np.max(eccs[idx]), np.min(eccs[idx])))
        # print('Eccentricity (nbody): max - {:.4f} | min - {:.4f}'.format(np.max(ecc_n_body[idx]), np.min(ecc_n_body[idx])))
        print('Eccentricity (L-L):   mean - {:.4f}'.format(np.mean(eccs[idx])))
        print('Eccentricity (nbody): mean - {:.4f}'.format(np.mean(ecc_n_body[idx])))
        print(np.abs(np.mean(eccs[idx])/np.mean(ecc_n_body[idx])*100-100))
        
        f_LL, y_LL, max_f_LL, max_amp_LL = find_frequency(times, timestep, eccs[idx])
        f_nbody, y_nbody, max_f_nbody, max_amp_nbody = find_frequency(time_nbody[idx], time_nbody[idx][1]-time_nbody[idx][0], ecc_n_body[idx])
        max_frq = max([max_f_LL, max_f_nbody])

        axes[idx].plot(f_nbody, y_nbody, 'k', label=n_body_data[idx].split('/')[-1].split('_')[0], linewidth=1)
        axes[idx].plot(f_LL, y_LL, 'b')

        axes[idx].axis(xmax=np.max([max_frq*1.1, 1*10**2]), xmin=0, ymin=np.min([y_LL, y_nbody]), ymax=1.1*np.max([y_LL, y_nbody]))
        f.subplots_adjust(hspace=0, top=0.97, left=0.15, right=0.96)

        max_t = max(times)
        # max_f_LL = max_t/max_f_LL
        # max_f_nbody = max_t/max_f_nbody
        # print('Amplitude: LL - {:.6f} | nbody - {:.6f}'.format(max_amp_LL, max_amp_nbody))
        # print('Frequency: LL - {:.6f} | nbody - {:.6f}'.format(max_f_LL, max_f_nbody))
        print()
    # plt.savefig('Report/GJ832_cycles.pdf')

def plot_all_freq():
    n_body_data = glob.glob('Exoplanets_data/{}/*_nbody.csv'.format(star_id))
    n = len(n_body_data)

    f, axes = plt.subplots(n, sharex=True, figsize=(8,7))
    ax0 = f.add_subplot(111, frame_on=False)   # creating a single axes
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_ylabel('FFT(Eccentricity)', labelpad=40, fontsize=16)
    ax0.set_xlabel('Frequency', labelpad=25, fontsize=16)
    for idx in range(n):
        f_LL, y_LL, max_f_LL, __ = find_frequency(times, timestep, eccs[idx])
        f_nbody, y_nbody, max_f_nbody, __ = find_frequency(time_nbody[idx], time_nbody[idx][1]-time_nbody[idx][0], ecc_n_body[idx])
        max_frq = max(times)/max([max_f_LL, max_f_nbody])

        axes[idx].semilogx(max(times)/f_nbody, y_nbody, 'k', label=n_body_data[idx].split('/')[-1].split('_')[0], linewidth=1)
        axes[idx].semilogx(max(times)/f_LL, y_LL, 'b')

        # axes[idx].axis(xmax=np.max(max(times)), xmin=2*10**3)
        f.subplots_adjust(hspace=0, top=0.97, left=0.1, right=0.97)

        max_t = max(times)
        max_f_LL = max_t/max_f_LL
        max_f_nbody = max_t/max_f_nbody
        # print('Frequency comparison: LL - {:.4f} | nbody - {:.4f}'.format(*max_f_LL, *max_f_nbody))

time_nbody, ecc_n_body = get_nbody_data(max_time=5*10**6)

times, timestep = np.linspace(0, max(time_nbody[0]), len(time_nbody[0]), retstep=True)
times = np.array(times, dtype='complex128')
eccs, __ = simulate(star_sys, times, plot=False, plot_orbit=False, save=False, folder_name='Exoplanets_data/{}'.format(star_id))
times = np.real(times)

compare_LL_nbody_eccs(times, eccs, time_nbody, ecc_n_body)
plot_all_cycles()
# plot_all_freq()


plt.show()