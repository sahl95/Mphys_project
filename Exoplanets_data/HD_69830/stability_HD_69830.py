import sys
sys.path.append('/Users/Sahl/Desktop/University/Year_5/Mphys_project/')

from functools import reduce
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from solar_system import solar_System, plot_simulation_separate
from planet import planet

M_SUN = 1.9885*10**30
M_EARTH = 5.9726*10**24
AU = 149597870700
R_SUN = 6.9551*10**8
G_CONST = 6.6738*10**(-11)


def simulate(star_sys, t, plot_orbit=False, plot=False, separate=True, save=False, folder_name=None):
    # star_sys.set_n()
    A, B = [star_sys.frequency_matrix(matrix_id=mat_id, J2=0, J4=0) for mat_id in ['A', 'B']]
    # print(A)
    g, x, f, y = *np.linalg.eig(A), *np.linalg.eig(B)
    S, beta, T, gamma = star_sys.find_all_scaling_factor_and_phase(x, y)
    # print(g*100*180/np.pi*3600, '\n')
    # print(A, B)
    kwargs = {'scaled_eigenvector' : S*x, 'eigenvalue' : g, 'phase' : beta,
                't' : t}
    h_list = star_sys.components_of_ecc_inc(**kwargs, eq_id='h')
    k_list = star_sys.components_of_ecc_inc(**kwargs, eq_id='k')
    kwargs = {'scaled_eigenvector' : T*y, 'eigenvalue' : f, 'phase' : gamma,
                't' : t}
    p_list = star_sys.components_of_ecc_inc(**kwargs, eq_id='p')#*180/np.pi
    q_list = star_sys.components_of_ecc_inc(**kwargs, eq_id='q')#*180/np.pi

    eccentricities = star_sys.get_eccentricity(h_list, k_list)
    inclinations = star_sys.get_inclination(p_list, q_list)

    r_planets = []
    max_axis = 0

    names = star_sys.get_property_all_planets('Name')
    a = star_sys.get_property_all_planets('a')
    e = star_sys.get_property_all_planets('e')
    m = star_sys.get_property_all_planets('Mass')
    # print(m/M_EARTH)
    M = star_sys.star_mass*M_SUN/M_EARTH

    if plot_orbit:
        x, y, z = np.zeros(2), np.zeros(2), np.zeros(2)
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z, 'b*', markersize=3, zorder=-999)
        ax.set_title('a = {}, e = {}'.format(a[-1], e[-1]))
    for idx in range(len(star_sys.planets)):
        xyz = star_sys.kep2cart(eccentricities, inclinations, h_list, k_list, p_list, q_list, t, 0, idx)
        r_planets.append(np.sqrt(xyz[0]**2+xyz[1]**2+xyz[2]**2))

        if plot_orbit:
            ax.plot(*xyz, '.', markersize=2, label=names[idx], zorder=-idx)
                # print(xyz[:, 0])
            if np.max([np.abs(np.min(xyz)), np.max(xyz)]) > max_axis:
                max_axis = np.max([np.abs(np.min(xyz)), np.max(xyz)])
            ax.set_zlim(-max_axis, max_axis)
            ax.set_ylim(-max_axis, max_axis)
            ax.set_xlim(-max_axis, max_axis)

            ax.set_zlabel('z (AU)')
            plt.xlabel('x (AU)')
            plt.ylabel('y (AU)')
            # print('here')
        # print(np.mean(r_planets[idx]))

    r_planets = np.array(r_planets)
    sep_c = np.abs(r_planets[1]-r_planets[-1])
    sep_d = np.abs(r_planets[2]-r_planets[-1])
    # print(sep_d)

    
    r_hill_c = ((m[1]+m[-1])/(3*M))**(1./3.)*(0.5*(a[1]+a[-1]))
    r_hill_d = ((m[2]+m[-1])/(3*M))**(1./3.)*(0.5*(a[2]+a[-1]))
    # print(2*np.sqrt(3)*((m[2]+m[1])/(3*M))**(1./3.)*(0.5*(a[2]+a[1])))
    # print(r_hill_d, sep_d)

    masks = [sep_c < 2*np.sqrt(3)*r_hill_c, sep_d < 2*np.sqrt(3)*r_hill_d]
    unstable = reduce(np.logical_or, masks)
    # print(np.sum(unstable))
    try:
        unstable_time_idx = np.where(unstable == True)[0][0]
        unstable_time = np.real(t[unstable_time_idx])
    except:
        unstable_time = np.real(t[-1])

    # print(unstable_time)
    # print(star_sys.planets[-1])

    return unstable_time

if __name__ == "__main__":

    folder = 'Exoplanets_data/'
    star_id = 'HD_69830'
    # star_id = ''
    folder_name = folder+star_id
    # star_sys = solar_System(folder_name+'/Sun.csv', folder_name+'/solar_system.csv')

    times = np.linspace(0, 10*10**(4), 52345)+0j

    a, e = np.linspace(0.1, 1.2, 140), np.linspace(0, 0.3, 30)
    a_grid, e_grid = np.meshgrid(a, e)
    # print(a_grid.shape, e_grid.shape)
    t_grid = np.zeros((a_grid.shape[0], a_grid.shape[1]))
    # print(np.shape(t_grid))
    total = float(a_grid.shape[0]*a_grid.shape[1])
    pbar = tqdm(total=total)

    a_list = []
    e_list = []
    t_list = []

    for i in range(a_grid.shape[0]):
        for j in range(a_grid.shape[1]):
            # print(a_grid[i, j], e_grid[i, j])
            star_sys = solar_System(folder_name+'/star.csv', folder_name+'/planets.csv')

            e_planet = e_grid[i, j]
            a_planet = a_grid[i, j]
            m_planet = 0.3
            pi_planet, Omega_planet, i_planet = 0, 0, 0
            n_planet = np.sqrt(G_CONST*star_sys.star_mass*M_SUN/(a_planet*AU)**3)*365*24*3600*180/np.pi
            earth_like_planet = planet('Earth_test', e=e_planet, a=a_planet, pi=pi_planet, i=i_planet, Omega=Omega_planet, n=n_planet, Mass=m_planet)
            star_sys.planets.append(earth_like_planet)

            unstable_time = simulate(star_sys, times, plot=False, plot_orbit=False, save=False, folder_name=folder_name)
            # print(a_planet, e_planet, unstable_time)
            t_grid[i, j] = unstable_time

            a_list.append(a_planet)
            e_list.append(e_planet)
            t_list.append(unstable_time)

            del star_sys
            # print(1/(len(a_grid)*len(e_grid)), 1/5)
            pbar.update()
    pbar.close()

    # df = pd.DataFrame({"a": a_list, "e": e_list, "time": t_list})
    df = pd.DataFrame(data=t_grid, columns=a, index=e)
    # print(df)
    df.to_csv(folder_name+'/_stability.csv')

    fig, ax = plt.subplots()
    im = plt.imshow(t_grid, extent=(np.amin(a_list), np.amax(a_list), np.amin(e_list), np.amax(e_list)), vmin=0, vmax=np.real(times[-1]))
    plt.xlabel('a (AU)')
    plt.ylabel('e')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical',label='Time')
    plt.savefig(folder_name+'/stability.png')
    # plt.show()
    # print(star_sys)
    
    # times = np.linspace(-3, 3, 1513)+0j
    # times = np.linspace(10**6, 10**10, 10000)+0j
    # times = np.logspace(6, 10, 10000)+0j
    # eccs = simulate(star_sys, times, plot=False, plot_orbit=False, save=False, folder_name=folder_name)