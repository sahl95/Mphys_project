import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from solar_system import solar_System, plot_simulation_separate

def simulate(star_sys, t, plot_orbit=False, plot=False, separate=True, save=False, folder_name=None):
    A, B = [star_sys.frequency_matrix(matrix_id=mat_id, J2=-6.84*10**(-7), J4=2.8*10**(-12)) for mat_id in ['A', 'B']]
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
    t = np.real(t)
    if plot:
        plot_simulation_separate(t/10**3, eccentricities, 'Time (kyr)', 'Eccentricity', names)
        plot_simulation_separate(t/10**3, inclinations*180/np.pi, 'Time (kyr)', 'Inclination', names)

    if plot_orbit:
        x, y, z = np.zeros(2), np.zeros(2), np.zeros(2)
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z, 'b*', markersize=3, zorder=-999)
        # ax = fig.add_subplot(111)
        # ax.plot(0, 0, 'b*', markersize=3)
        for idx in range(len(star_sys.planets)):
            xyz = star_sys.kep2cart(eccentricities, inclinations, h_list, k_list, p_list, q_list, t, 0, idx)
            # ax.plot(X, Y, '.', markersize=2, label=names[idx])
            # if idx == 2:
            #     ax.plot(xyz[0][80:], xyz[1][80:], xyz[2][80:], '--', markersize=2, label=names[idx], zorder=-idx)
            # else:
            #     ax.plot(*xyz, '--', markersize=2, label=names[idx], zorder=-idx)
            ax.plot(*xyz, '.', markersize=2, label=names[idx], zorder=-idx)
            ax.set_zlim(-np.max([np.max(xyz[0]), np.max(xyz[1])]), np.max([np.max(xyz[0]), np.max(xyz[1])]))
            # ax.set_ylim(-np.max([np.max(xyz[0]), np.max(xyz[1])]), np.max([np.max(xyz[0]), np.max(xyz[1])]))
            # ax.set_xlim(-np.max([np.max(xyz[0]), np.max(xyz[1])]), np.max([np.max(xyz[0]), np.max(xyz[1])]))
            ax.set_zlabel('z (AU)')
            plt.xlabel('x (AU)')
            plt.ylabel('y (AU)')

            if save:
                df = pd.DataFrame({"time": t ,"x" : xyz[0], "y" : xyz[1], "z" : xyz[2]})
                df.to_csv(folder_name+'/'+names[idx]+'_xyz.csv', index=False)

if __name__ == "__main__":

    folder = 'Exoplanets_data/'
    star_id ='HD_12661'
    folder_name = folder+star_id
    star_name = folder_name.split('/')[-1]
    star_sys = solar_System('Exoplanets_Data/'+star_name+'/star.csv', 'Exoplanets_Data/'+star_name+'/planets.csv')

    # t = np.linspace(-1*10**(0), 1*10**(0), 500)+0j
    t = np.linspace(-3, 3, 500)+0j
    simulate(star_sys, t, plot=False, plot_orbit=True, save=True, folder_name=folder_name)

    plt.show()
    # print(star_sys)