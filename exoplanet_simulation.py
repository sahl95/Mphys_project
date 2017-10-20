import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from solar_system import solar_System, plot_simulation_separate

def simulate(star_sys, t, plot_orbit=False, plot=False, separate=True, save=False, folder_name=None):
    A, B = [star_sys.frequency_matrix(matrix_id=mat_id, J2=0, J4=0, do_ecc_damp=False) for mat_id in ['A', 'B']]
    g, x, f, y = *np.linalg.eig(A), *np.linalg.eig(B)
    S, beta, T, gamma = star_sys.find_all_scaling_factor_and_phase(x, y)
    # print(g*100*180/np.pi*3600, '\n')

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
    # a = star_sys.get_property_all_planets('a')
    t = np.real(t)
    if plot:
        plot_simulation_separate(t/10**6, 5*(1-eccentricities), 'Time (Myr)', 'Eccentricity', names)
        # plot_simulation_separate(t/10**6, inclinations*180/np.pi, 'Time (Myr)', 'Inclination', names)

    if plot_orbit:
        x, y, z = np.zeros(2), np.zeros(2), np.zeros(2)
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z, 'b*', markersize=3, zorder=-999)
        # ax = fig.add_subplot(111)
        # ax.plot(0, 0, 'b*', markersize=3)
        for idx in range(len(star_sys.planets)):
            xyz = star_sys.kep2cart(eccentricities, inclinations, h_list, k_list, p_list, q_list, t, 0, idx)

            ax.plot(*xyz, '.', markersize=2, label=names[idx], zorder=-idx)
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

if __name__ == "__main__":

    folder = 'KR_paper_tests/'
    # folder = 'Exoplanets_data/'
    star_id ='1st_test'
    folder_name = folder+star_id
    star_sys = solar_System(folder_name+'/star.csv', folder_name+'/planets.csv')

    # print(star_sys)
    # t = np.linspace(-5*10**(4), 5*10**(4), 5000)+0j
    # t = np.linspace(-0, .1, 500)+0j
    t = np.linspace(10**6, 2.5*10**10, 10000)+0j
    simulate(star_sys, t, plot=True, plot_orbit=True, save=False, folder_name=folder_name)


    # df = pd.read_csv(folder_name+'/c_nbody.csv')
    # plt.figure()
    # plt.plot(df['Time'], df['e'])
    plt.show()
    # print(star_sys)