import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from solar_system import solar_System, plot_simulation_separate

M_SUN = 1.9885*10**30
M_EARTH = 5.9726*10**24
AU = 149597870700
R_SUN = 6.9551*10**8

def simulate(star_sys, t, plot_orbit=False, plot=False, separate=True, save=False, folder_name=None):
    star_sys.set_n()

    A, B = [star_sys.frequency_matrix(matrix_id=mat_id, J2=0, J4=0) for mat_id in ['A', 'B']]
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

    names = star_sys.get_property_all_planets('Name')
    a = np.reshape(star_sys.get_property_all_planets('a'), (len(star_sys.planets), 1))

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
        
        print("PLANET DATA FOR big.in")
        print("----------------------")
        for idx in range(len(star_sys.planets)):
            star_id = folder_name.split('/')[-1]
            prefix = ''
            for s in star_id.split('_'):
                prefix += s
            
            print('{}{}  m=  {} '.format(prefix, names[idx], star_sys.planets[idx].Mass*M_EARTH/M_SUN), end='')
            print(' r=   2.0000000000000000      d=   3.0000000000000000')
            xyz_init, uvw_init = star_sys.initial_pos_vel(idx, t[0])
            # print(uvw_init)
            xyz = star_sys.kep2cart(eccentricities, inclinations, h_list, k_list, p_list, q_list, t, 0, idx)
            print(' {} {} {}'.format(*xyz_init))#, np.sqrt(xyz_init[0]**2+xyz_init[1]**2+xyz_init[2]**2)))
            print(' {} {} {}'.format(*uvw_init))#, np.sqrt(uvw_init[0]**2+uvw_init[1]**2+uvw_init[2]**2)))
            # print('x = {:.4f}, y = {:.4f}, z = {:.4f} | r = {:.4f}'.format(*xyz[:, 0], np.sqrt(xyz[0, 0]**2+xyz[1, 0]**2+xyz[2, 0]**2)))
            print(' 0.0000000000000000        0.0000000000000000        0.0000000000000000')

            ax.plot(*xyz, '.', markersize=2, label=names[idx], zorder=-idx)
            # print(xyz[:, 0])
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

    precession_rates, xlabel = star_sys.get_perihelion_precession_rates(A, eccentricities, h_list, k_list), 'Pericenter'

    # for idx in range(len(star_sys.planets)):
    #     # print('Eccentricity of {} = {:.4f}'.format(names[idx],
    #         #   np.mean(eccentricities[idx])))
    #     print('Precession rate of {} = {} arcseconds per century'.format(names[idx],
    #             np.mean(precession_rates[idx])*180/np.pi*3600*100))
        
    return eccentricities

if __name__ == "__main__":

    # folder = 'SolarSystemData/'
    # folder = 'KR_paper_tests/'
    folder = 'Exoplanets_data/'
    star_id = 'HD_12661'
    # star_id = ''
    folder_name = folder+star_id
    star_sys = solar_System(folder_name+'/star.csv', folder_name+'/planets.csv')
    # star_sys = solar_System(folder_name+'/Sun.csv', folder_name+'/solar_system.csv')

    # print(star_sys)
    times = np.linspace(0, 5*10**(4), 12345)+0j
    # times = np.linspace(-0, .1, 500)+0j
    # times = np.linspace(10**6, 10**10, 10000)+0j
    # times = np.logspace(6, 10, 10000)+0j
    eccs = simulate(star_sys, times, plot=True, plot_orbit=True, save=False, folder_name=folder_name)

    print("\nSTELLAR DATA FOR param.in")
    print("-------------------------") 
    print(star_sys.star_mass)
    print(star_sys.star_radius*R_SUN/AU)

    # plt.show() 
    # print(star_sys.get_property_all_planets('Mass')*M_EARTH/M_SUN)
