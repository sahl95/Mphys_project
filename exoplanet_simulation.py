import numpy as np
import pandas as pd
import time
from functools import reduce
import os
import matplotlib.pyplot as plt
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
    # print(star_sys.A_octupole())
    # A = star_sys.A_octupole()
    # print(A)
    g, x = np.linalg.eig(A)
    f, y = np.linalg.eig(B)
    S, beta, T, gamma = star_sys.find_all_scaling_factor_and_phase(x, y)
    # print(g*100*180/np.pi*3600, '\n')
    # print(A, B)
    kwargs = {'scaled_eigenvector' : S*x, 'eigenvalue' : g, 'phase' : beta,
                't' : t}
    h_list = star_sys.components_of_ecc_inc(scaled_eigenvector=S*x, eigenvalue=g, phase=beta, t=t, eq_id='h')
    k_list = star_sys.components_of_ecc_inc(scaled_eigenvector=S*x, eigenvalue=g, phase=beta, t=t, eq_id='k')
    kwargs = {'scaled_eigenvector' : T*y, 'eigenvalue' : f, 'phase' : gamma,
                't' : t}
    p_list = star_sys.components_of_ecc_inc(scaled_eigenvector=T*y, eigenvalue=g, phase=gamma, t=t, eq_id='p')#*180/np.pi
    q_list = star_sys.components_of_ecc_inc(scaled_eigenvector=T*y, eigenvalue=f, phase=gamma, t=t, eq_id='q')#*180/np.pi


    eccentricities = star_sys.get_eccentricity(h_list, k_list)
    inclinations = star_sys.get_inclination(p_list, q_list)

    names = star_sys.get_property_all_planets('Name')
    a = np.reshape(star_sys.get_property_all_planets('a'), (len(star_sys.planets), 1))
    a = star_sys.get_property_all_planets('a')
    e = star_sys.get_property_all_planets('e')
    m = star_sys.get_property_all_planets('Mass')
    # print(m/M_EARTH)
    M = star_sys.star_mass*M_SUN/M_EARTH

    xyz_center, uvw_center = star_sys.centre_of_mass()

    t = np.real(t)
    if plot:
        plot_simulation_separate(t/10**6, eccentricities, 'Time (Myr)', 'Eccentricity', names)
        # plot_simulation_separate(t/10**6, inclinations*180/np.pi, 'Time (Myr)', 'Inclination', names)

    if plot_orbit:
        x, y, z = np.zeros(2), np.zeros(2), np.zeros(2)
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z, 'b*', markersize=3, zorder=-999)
        # ax = fig.add_subplot(111)
        # ax.plot(0, 0, 'b*', markersize=3)
        
        n_body_string = "PLANET DATA FOR big.in\n"+"----------------------\n"

        r_planets = []
        x_planets, y_planets, z_planets = [], [], []   
        max_axis = 0
        for idx in range(len(star_sys.planets)):
            star_id = folder_name.split('/')[-1]
            prefix = ''
            for s in star_id.split('_'):
                prefix += s

            xyz_init, uvw_init = star_sys.initial_pos_vel(idx)
            xyz = star_sys.kep2cart_2(eccentricities, inclinations, h_list, k_list, p_list, q_list, t, 0, idx)
            
            n_body_string += '{}{}  m=  {} '.format(prefix, names[idx], star_sys.planets[idx].Mass*M_EARTH/M_SUN)
            n_body_string += ' r=   2.0000000000000000      d=   3.0000000000000000\n'
            n_body_string += ' {} {} {}\n'.format(xyz_init[0], xyz_init[1], xyz_init[2])
            n_body_string += ' {} {} {}\n'.format(uvw_init[0], uvw_init[1], uvw_init[2])
            n_body_string += ' 0.0000000000000000        0.0000000000000000        0.0000000000000000\n'

            # print('x = {:.4f}, y = {:.4f}, z = {:.4f} | r = {:.4f}'.format(*xyz[:, 0], np.sqrt(xyz[0, 0]**2+xyz[1, 0]**2+xyz[2, 0]**2)))

            r_planets.append(np.sqrt(xyz[0]**2+xyz[1]**2+xyz[2]**2))
            x_planets.append(xyz[0])
            y_planets.append(xyz[1])
            z_planets.append(xyz[2])

            ax.plot(xyz[0], xyz[1], xyz[2], '.', markersize=2, label=names[idx], zorder=-idx)
            # print(xyz[:, 0])
            if np.max([np.abs(np.min(xyz)), np.max(xyz)]) > max_axis:
                max_axis = np.max([np.abs(np.min(xyz)), np.max(xyz)])
            ax.set_zlim(-max_axis, max_axis)
            ax.set_ylim(-max_axis, max_axis)
            ax.set_xlim(-max_axis, max_axis)

            ax.set_zlabel('z (AU)')
            plt.xlabel('x (AU)')
            plt.ylabel('y (AU)')

            if save:
                df = pd.DataFrame({"time": t ,"x" : xyz[0], "y" : xyz[1], "z" : xyz[2]})
                df.to_csv(folder_name+'/'+names[idx]+'_xyz.csv', index=False)

        print(n_body_string)

    # sep_b = np.sqrt((x_planets[0]-x_planets[-1])**2+(y_planets[0]-y_planets[-1])**2+(z_planets[0]-z_planets[-1])**2)
    # sep_c = np.sqrt((x_planets[1]-x_planets[-1])**2+(y_planets[1]-y_planets[-1])**2+(z_planets[1]-z_planets[-1])**2)
    # sep_d = np.sqrt((x_planets[2]-x_planets[-1])**2+(y_planets[2]-y_planets[-1])**2+(z_planets[2]-z_planets[-1])**2)
    # r_hill_b = ((m[0]+m[-1])/(3*M))**(1./3.)*(0.5*(a[0]+a[-1]))
    # r_hill_c = ((m[1]+m[-1])/(3*M))**(1./3.)*(0.5*(a[1]+a[-1]))
    # r_hill_d = ((m[2]+m[-1])/(3*M))**(1./3.)*(0.5*(a[2]+a[-1]))

    # masks = [sep_b < 2*np.sqrt(3)*r_hill_b, sep_c < 2*np.sqrt(3)*r_hill_c, sep_d < 2*np.sqrt(3)*r_hill_d]
    # unstable = reduce(np.logical_or, masks)

    # print(np.sum(unstable))

    # precession_rates, xlabel = star_sys.get_perihelion_precession_rates(A, eccentricities, h_list, k_list), 'Pericenter'
    # plt.figure()
    # plt.plot(t, sep_b, '.')
    # plt.plot(t, np.ones(len(t))*r_hill_b)

    # plt.figure()
    # plt.plot(t, sep_c, '.')
    # plt.plot(t, np.ones(len(t))*r_hill_c)
    
    # plt.figure()
    # plt.plot(t, sep_d, '.')
    # plt.plot(t, np.ones(len(t))*r_hill_d)

    # for idx in range(len(star_sys.planets)):
    #     # print('Eccentricity of {} = {:.4f}'.format(names[idx],
    #         #   np.mean(eccentricities[idx])))
    #     print('Precession rate of {} = {} arcseconds per century'.format(names[idx],
    #             np.mean(precession_rates[idx])*180/np.pi*3600*100))
        
    return eccentricities, inclinations

if __name__ == "__main__":
    t1 = time.clock()

    # folder = 'SolarSystemData/'
    # folder = 'KR_paper_tests/'
    folder = 'Exoplanets_data/'
    star_id = 'HD_141399'
    # star_id = ''
    folder_name = folder+star_id
    star_sys = solar_System(folder_name+'/star.csv', folder_name+'/planets.csv')
    # star_sys = solar_System(folder_name+'/Sun.csv', folder_name+'/solar_system.csv')

    # e_planet = 0.3
    # a_planet = 1.004892086
    # m_planet = 0.3
    # #pylint: disable=maybe-no-member
    # pi_planet, Omega_planet, i_planet = 360*np.random.random(), 360*np.random.random(), np.random.uniform(5, 7)
    # n_planet = np.sqrt(G_CONST*star_sys.star_mass*M_SUN/(a_planet*AU)**3)*365*24*3600*180/np.pi
    # earth_like_planet = planet('Earth_test', e=e_planet, a=a_planet, pi=pi_planet, i=i_planet, Omega=Omega_planet, n=n_planet, Mass=m_planet)
    # star_sys.planets.append(earth_like_planet)

    # print(star_sys)
    times = np.linspace(0, 1*10**(5), 12345)+0j
    # times = np.linspace(-3, 3, 1513)+0j
    # times = np.linspace(10**6, 10**10, 10000)+0j
    # times = np.logspace(6, 10, 10000)+0j
    eccs = simulate(star_sys, times, plot=True, plot_orbit=False, save=False, folder_name=folder_name)
    # print('Time taken: {}s'.format(time.clock()-t1))

    print("\nSTELLAR DATA FOR param.in")
    print("-------------------------") 
    print(star_sys.star_mass)
    print(star_sys.star_radius*R_SUN/AU)

    plt.show()
    # print(star_sys.get_property_all_planets('Mass')*M_EARTH/M_SUN)
