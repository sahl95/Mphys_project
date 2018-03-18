import os
import sys
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

try:
    star_sys = solar_System('star.csv', 'planets.csv')
except:
    star_sys = solar_System('Exoplanets_data/HD_11506/star.csv', 'Exoplanets_data/HD_11506/planets.csv')

def simulate_quadrupole(star_system, t):
    A, B = [star_sys.frequency_matrix(matrix_id=mat_id, J2=0, J4=0) for mat_id in ['A', 'B']]
    # A = star_sys.A_octupole()
    # print(A)
    g, x, f, y = *np.linalg.eig(A), *np.linalg.eig(B)
    S, beta, T, gamma = star_sys.find_all_scaling_factor_and_phase(x, y)
    # print(g*100*180/np.pi*3600, '\n')
    # print(A, B)
    kwargs = {'scaled_eigenvector' : S*x, 'eigenvalue' : g, 'phase' : beta,
                't' : t}
    h_list = star_sys.components_of_ecc_inc(**kwargs, eq_id='h')
    k_list = star_sys.components_of_ecc_inc(**kwargs, eq_id='k')

    eccentricities = star_sys.get_eccentricity(h_list, k_list)
    return eccentricities

def simulate_octupole(star_system, t):
    A, B = [star_sys.frequency_matrix(matrix_id=mat_id, J2=0, J4=0) for mat_id in ['A', 'B']]
    A = star_sys.A_octupole()
    # print(A)
    g, x, f, y = *np.linalg.eig(A), *np.linalg.eig(B)
    S, beta, T, gamma = star_sys.find_all_scaling_factor_and_phase(x, y)
    # print(g*100*180/np.pi*3600, '\n')
    # print(A, B)
    kwargs = {'scaled_eigenvector' : S*x, 'eigenvalue' : g, 'phase' : beta,
                't' : t}
    h_list = star_sys.components_of_ecc_inc(**kwargs, eq_id='h')
    k_list = star_sys.components_of_ecc_inc(**kwargs, eq_id='k')

    eccentricities = star_sys.get_eccentricity(h_list, k_list)
    return eccentricities

def comparison(star_system, times):
    ecc_quad = simulate_quadrupole(star_sys, times)
    ecc_oct = simulate_octupole(star_sys, times)

    times = np.real(times)
    try:
        df = pd.read_csv('Exoplanets_data/HD_11506'+'/b_nbody.csv')
    except:
        df = pd.read_csv('b_nbody.csv')

    f, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(8,7))
    ax0 = f.add_subplot(111, frame_on=False)   # creating a single axes
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_ylabel('Eccentricity', labelpad=35)
    ax0.set_xlabel('Time (yrs)', labelpad=25)

    t, e = np.array(df.Time), np.array(df.e)
    ax1.plot(t, e, 'k--', label='b', linewidth=1)
    ax1.plot(times, ecc_quad[0], 'b')
    ax1.plot(times, ecc_oct[0], 'orange')
    # ax1.set_xlabel('Time')
    # ax1.set_ylabel(r"Eccentricity")
    ax1.axis(xmin=t[0], xmax=times[-1])
    l = ax1.legend(loc='upper left',
            ncol=3, fancybox=True, shadow=False, facecolor='black',
            handlelength=0, handletextpad=0)
    for text in l.get_texts():
        text.set_color("white")

    try:
        df = pd.read_csv('Exoplanets_data/HD_11506'+'/c_nbody.csv')
    except:
        df = pd.read_csv('c_nbody.csv')
    t, e = np.array(df.Time), np.array(df.e)
    # plt.figure()
    ax2.plot(t, e, 'k--', label='c', linewidth=1)
    ax2.plot(times, ecc_quad[1], 'b')
    ax2.plot(times, ecc_oct[1], 'orange')
    # plt.xlabel('Time')
    # plt.ylabel(r"Eccentricity")
    ax2.axis(xmin=t[0], xmax=times[-1])
    l = ax2.legend(loc='upper left',
            ncol=3, fancybox=True, shadow=False, facecolor='black',
            handlelength=0, handletextpad=0)
    for text in l.get_texts():
        text.set_color("white")

    f.subplots_adjust(hspace=0, top=0.97)
    # plt.show()

if __name__ == "__main__":
    # star_sys = solar_System('KR_paper_tests/2nd_test/star.csv', 'KR_paper_tests/2nd_test/planets.csv')
    times = np.linspace(0, 1*10**(5), 12345)+0j
    # times = np.linspace(0, 2.5*10**(7), 12345)+0j
    # times = np.logspace(6, 9, 12345)+0j
    # ecc_quad = simulate_quadrupole(star_sys, times)
    # ecc_oct = simulate_octupole(star_sys, times)

    comparison(star_sys, times)
    # for e in range(len(ecc_quad)):
    #     plt.figure()
    #     plt.plot(times, ecc_quad[e])
    #     plt.plot(times, ecc_oct[e])

    


    plt.show()