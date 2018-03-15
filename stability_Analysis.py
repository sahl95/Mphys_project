from functools import reduce, partial
from  multiprocessing import Pool
import numpy as np
import pandas as pd
import time
from tqdm import *
import warnings
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.axes_grid1 import make_axes_locatable
from solar_system import solar_System, plot_simulation_separate
from planet import planet

M_SUN = 1.9885*10**30
M_EARTH = 5.9726*10**24
AU = 149597870700
R_SUN = 6.9551*10**8
G_CONST = 6.6738*10**(-11)

def simulate(star_sys, t):
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
    x_planets, y_planets, z_planets = [], [], []
    max_axis = 0

    names = star_sys.get_property_all_planets('Name')
    a = star_sys.get_property_all_planets('a')
    e = star_sys.get_property_all_planets('e')
    m = star_sys.get_property_all_planets('Mass')
    M = star_sys.star_mass*M_SUN/M_EARTH

    for idx in range(len(star_sys.planets)):
        xyz = star_sys.kep2cart_2(eccentricities, inclinations, h_list, k_list, p_list, q_list, t, 0, idx)
        r_planets.append(np.sqrt(xyz[0]**2+xyz[1]**2+xyz[2]**2))
        x_planets.append(xyz[0])
        y_planets.append(xyz[1])
        z_planets.append(xyz[2])

    masks = []
    for i in range(len(star_sys.planets)-1):
        sep = np.sqrt((x_planets[i]-x_planets[-1])**2+(y_planets[i]-y_planets[-1])**2+(z_planets[i]-z_planets[-1])**2)
        r_hill = ((m[i]+m[-1])/(3*M))**(1./3.)*(0.5*(a[i]+a[-1]))
        masks.append(sep < 2*np.sqrt(3)*r_hill)
    #pylint: disable=maybe-no-member
    unstable = reduce(np.logical_or, masks)

    try:
        unstable_time_idx = np.where(unstable == True)[0][0]
        unstable_time = np.real(t[unstable_time_idx])
    except:
        unstable_time = np.real(t[-1])

    return unstable_time, np.sum(unstable)

def add_planet(starSystem, **planet_kwargs):
    starSystem.planets.append(planet(**planet_kwargs))

def search_space(a_min, a_max, e_min, e_max, a_pts=21, e_pts=20):
    a, e = np.linspace(a_min, a_max, a_pts), np.linspace(e_min, e_max, e_pts)
    a_list, e_list = [], []
    for i in range(len(e)):
        for j in range(len(a)):
            a_list.append(a[j])
            e_list.append(e[i])
    return a_list, e_list

def parallel_search_stability(searchSpace, star_id, folder, times):
    folder_name = folder+star_id
    a_idx, e_idx = searchSpace
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        star_sys = solar_System(folder_name+'/star.csv', folder_name+'/planets.csv')

        e_planet = e_idx
        a_planet = a_idx
        m_planet = 0.7
        #pylint: disable=maybe-no-member
        pi_planet, Omega_planet, i_planet = 360*np.random.random(), 360*np.random.random(), np.random.uniform(5, 7)
        n_planet = np.sqrt(G_CONST*star_sys.star_mass*M_SUN/(a_planet*AU)**3)*365*24*3600*180/np.pi
        add_planet(star_sys, Name='Earth_test', e=e_planet, a=a_planet, pi=pi_planet, i=i_planet, Omega=Omega_planet, n=n_planet, Mass=m_planet)

        unstable_time, unstable_pts = simulate(star_sys, times)
    return unstable_time

def plot_stabilty(aList, eList, tList):
    triang = tri.Triangulation(aList, eList)
    fig, ax = plt.subplots()
    im = plt.tripcolor(aList, eList, tList, cmap='RdYlGn', shading='gouraud')
    plt.axis(xmin=np.amin(aList), xmax=np.amax(aList), ymin=np.amin(eList), ymax=np.amax(eList))
    plt.xlabel('a (AU)')
    plt.ylabel('e')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('top', size='5%', pad=0.55)
    fig.colorbar(im, cax=cax, orientation='horizontal',label='Time (years)')
    plt.subplots_adjust(top=0.95)
    # plt.savefig('Report/Images/pngs/stability_HD_69830_2.png')

def run(star_id, folder, simtime, a_pts, e_pts):
    a_list, e_list = search_space(a_min=0.07, a_max=1.2, e_min=0, e_max=0.2, a_pts=a_pts, e_pts=e_pts)

    with Pool(processes=4) as p:
        t_list = list(tqdm(p.imap(partial(parallel_search_stability, star_id=star_id, folder=folder, times=np.linspace(0, simtime, 12345)+0j), zip(a_list, e_list)), total=a_pts*e_pts))
    t_list = np.array(t_list)
    df = pd.DataFrame({"a": a_list, "e": e_list, "time": t_list})
    # df = pd.DataFrame(data=t_grid, columns=a, index=e)
    df.to_csv('stability_data.csv')

    data = pd.read_csv('stability_data.csv')
    plot_stabilty(data['a'], data['e'], data['time'])

if __name__ == "__main__":
    run(star_id='HD_69830', folder='Exoplanets_data/', simtime=10**5, a_pts=20, e_pts=20)
    plt.show()