import pandas as pd
import numpy as np
import numpy.ma as ma
from scipy import integrate
import scipy.linalg
from scipy.optimize import fsolve
from sympy import symbols, Matrix, linsolve, diag
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from planet import planet
from unitConverter import *


M_SUN = 1.9885*10**30
R_SUN = 6.9551*10**8
M_EARTH = 5.9726*10**24
AU = 149597870700
SECS_IN_YEAR = 365.25*24*3600
G_CONST = 6.6738*10**(-11)
LIGHT_SPD = 63241.0774

def calculate_laplace_coeff(alpha, j, s):
    return integrate.quad(lambda psi, alpha, j, s: np.cos(j*psi)/(1-2*alpha*np.cos(psi)+alpha**2)**s,
                          0, 2*np.pi, args=(alpha, j, s,))[0]/np.pi

def scaling_factor_and_phase(p, *boundaries):
    s, phase = p
    # test what real and imaginary componets of the equations look like of s*np.sin(phase)=(a+b*1j)*np.sin(c+d*1j)
    return (s*np.sin(phase)-boundaries[0], s*np.cos(phase)-boundaries[1])

class solar_System():

    def __init__(self, starMass, starRadius, planet_data_file):
        self.star_mass = starMass
        self.star_radius = starRadius
        self.planets = self.addPlanets(planet_data_file)

    def addPlanets(self, planet_data_file):
        planets = pd.read_csv(planet_data_file)
        planet_list = []
        for p in range(len(planets)):
            planet_list.append(planet(**planets.ix[p]))

        return planet_list

    def print_planets(self):
        for p in self.planets:
            p.toString()

    def get_property_all_planets(self, property_name, data_type="float"):
        """
        Returns an array containing a specific property of all planets.

        Args:
            property_name (str): The name of the property. Can be one of *a*,
                 *b*, *e*, *period*, *mass*.
        """
        property_list = np.zeros(len(self.planets), dtype=data_type)
        for idx, p in enumerate(self.planets):
            property_list[idx] = p.__dict__[property_name]

        return property_list

    def frequency_matrix(self, matrix_id, J2=0, J4=0):
        M_star_kg = M_SUN*self.star_mass
        R = R_SUN*self.star_radius
        m = M_EARTH*self.get_property_all_planets('mass')
        n = self.get_property_all_planets('n')*np.pi/180
        a = AU*self.get_property_all_planets('a')
        e = self.get_property_all_planets('e')
        n_planets = len(self.planets)
        f_mat = np.zeros([n_planets, n_planets])

        gr_correction = np.zeros(n_planets)

        if matrix_id == 'A':
            j_laplace_coeff_jk, j_laplace_coeff_jj = 2, 1
            front_factor = -1
            J2_correction = (((3/2)*J2*(R/a)**2)-((9/8)*(J2**2)*(R/a)**4)-((15/4)*J4*(R/a)**4))
            gr_correction[0] = 3*(a[0]/AU)**2*(n[0])**3/(LIGHT_SPD**2*(1+m[0]/M_star_kg))*1/(1-e[0]**2)
            # gr_correction = 3*(a/AU)**2*(n)**3/(LIGHT_SPD**2*(1+m/M_star_kg))*1/(1-e**2)

        if matrix_id == 'B':
            j_laplace_coeff_jk = j_laplace_coeff_jj = 1
            front_factor = 1
            J2_correction = (((3/2)*J2*(R/a)**2)-((27/8)*(J2**2)*(R/a)**4)-((15/4)*J4*(R/a)**4))
            n_merc = np.sqrt(G_CONST*M_star_kg/a[0]**3)

        for j in range(n_planets):
            for k in range(n_planets):
                if j != k:
                    alpha_jk = a[j]/a[k]
                    if alpha_jk > 1:
                        alpha_jk = alpha_jk**(-1)
                    laplace_coeff = calculate_laplace_coeff(alpha_jk, j_laplace_coeff_jk, 3/2)
                    alpha_jk_bar = np.where(a[k] < a[j], 1, alpha_jk)
                    f_mat[j, k] = front_factor*(n[j]/4)*(m[k]/(M_star_kg+m[j]))*alpha_jk*alpha_jk_bar*laplace_coeff

                else:
                    for kk in range(n_planets):
                        if kk != j:
                            alpha_jj = a[j]/a[kk]
                            if alpha_jj > 1:
                                alpha_jj = alpha_jj**(-1)
                            laplace_coeff = calculate_laplace_coeff(alpha_jj, j_laplace_coeff_jj, 3/2)
                            alpha_jj_bar = np.where(a[kk] < a[j], 1, alpha_jj)
                            f_mat[j, k] += (1/4)*(m[kk]/(M_star_kg+m[j]))*alpha_jj*alpha_jj_bar*laplace_coeff
                    f_mat[j, k] += J2_correction[j]
                    f_mat[j, k] *= -front_factor*(n[j])
                    f_mat[j, k] += gr_correction[j]
        # print(f_mat[0, 0], gr_correction[0])
        return f_mat

    def initial_conditions(self):
        e = self.get_property_all_planets('e')
        omega_bar = self.get_property_all_planets('omega_bar')*np.pi/180
        i = self.get_property_all_planets('i')*np.pi/180
        omega = self.get_property_all_planets('omega')*np.pi/180

        h = e*np.sin(omega_bar)
        k = e*np.cos(omega_bar)
        p = i*np.sin(omega)
        q = i*np.cos(omega)

        return h, k, p, q

    def solve_property(self, eigenvectors, initial_conditions):
        n = len(self.planets)
        aug = Matrix(np.zeros([n, n+1]))
        aug[:, :n] = eigenvectors
        aug[:, n] = initial_conditions

        result = linsolve(aug, *symbols('x0:'+str(n)))

        answers = np.zeros(n)
        for ans in result:
            for a, answer in enumerate(ans):
                answers[a] = answer
        # answers = np.linalg.solve(eigenvectors, initial_conditions)
        return answers

    def find_all_scaling_factor_and_phase(self, eigenvectors_of_A, eigenvectors_of_B):
        x, y = eigenvectors_of_A, eigenvectors_of_B

        init_conditions = np.array(self.initial_conditions())
        h_solved = self.solve_property(x, init_conditions[0, :])
        k_solved = self.solve_property(x, init_conditions[1, :])
        p_solved = self.solve_property(y, init_conditions[2, :])
        q_solved = self.solve_property(y, init_conditions[3, :])

        # print(h_solved)
        # print(k_solved)
        # print(p_solved)
        # print(q_solved)

        n = len(self.planets)
        S, beta = np.zeros(n), np.zeros(n)
        T, gamma = np.zeros(n), np.zeros(n)

        for i in range(n):
            S[i], beta[i] = fsolve(scaling_factor_and_phase, (1, -1), args=(h_solved[i], k_solved[i],))
            T[i], gamma[i] = fsolve(scaling_factor_and_phase, (-1, 1), args=(p_solved[i], q_solved[i],))

        return S, beta, T, gamma

    def eq_of_motion(self, scaled_eigenvector, eigenvalue, phase, t, eq_id):
        # eq_id = 'h', 'k', 'p', 'q'
        kwargs = {'scaled_eigenvector' : scaled_eigenvector, 'eigenvalue' : eigenvalue, 'phase' : phase,
                  't' : t}
        if eq_id == 'h' or eq_id == 'p':
            return self.get_h_or_p(**kwargs)
        if eq_id == 'k' or eq_id == 'q':
            return self.get_k_or_q(**kwargs)
    
    def get_h_or_p(self, scaled_eigenvector, eigenvalue, phase, t):
        n = len(self.planets)
        h_list = []
        for j in range(n):
            h = np.zeros_like(t)
            for i in range(n):
                h += scaled_eigenvector[j, i]*np.sin((eigenvalue[i]*t+phase[i]))
            h_list.append(h)
        return np.array(h_list)

    def get_k_or_q(self, scaled_eigenvector, eigenvalue, phase, t):
        n = len(self.planets)
        k_list = []
        for j in range(n):
            k = np.zeros_like(t)
            for i in range(n):
                k += scaled_eigenvector[j, i]*np.cos((eigenvalue[i]*t+phase[i]))
            k_list.append(k)
        return np.array(k_list)

    def disturbing_function(self, A, B, h, k, p, q):
        # plt.figure()
        # plt.plot(t, h[0])

        n = self.get_property_all_planets('n')
        a = self.get_property_all_planets('a')

        # A *= (365*24*3600*180/np.pi)
        # B *= (365*24*3600*180/np.pi)
        n_planets = len(self.planets)
        R_list = []
        for j in range(n_planets):
            R = 0
            R += 0.5*A[j, j]*(h[j]**2+k[j]**2)
            R += 0.5*B[j, j]*(p[j]**2+q[j]**2)
            for i in range(n_planets):
                if j != i:
                    R += A[j, i]*(h[j]*h[i]+k[j]*k[i])
                    R += B[j, i]*(p[j]*h[i]+q[j]*k[i])
            R_list.append(n[j]*a[j]**2*R)

        # plt.figure()
        # plt.plot(t, R_list[0])

        return R_list


    def get_eccentricity(self, scaled_eigenvector_of_A, eigenvalue_of_A, beta, t):
        n = len(self.planets)
        kwargs = {'scaled_eigenvector' : scaled_eigenvector_of_A, 'eigenvalue' : eigenvalue_of_A, 'phase' : beta,
                  't' : t}
        eccentricities = []
        h, k = self.eq_of_motion(**kwargs, eq_id='h'), self.eq_of_motion(**kwargs, eq_id='k')
        for j in range(n):  
            eccentricities.append(np.sqrt(h[j]**2+k[j]**2))
        return np.array(eccentricities)

    def get_inclination(self, scaled_eigenvector_of_B, eigenvalue_of_B, gamma, t):
        n = len(self.planets)
        kwargs = {'scaled_eigenvector' : scaled_eigenvector_of_B, 'eigenvalue' : eigenvalue_of_B, 'phase' : gamma,
                  't' : t}
        inclinations = []
        p, q = self.eq_of_motion(**kwargs, eq_id='p'), self.eq_of_motion(**kwargs, eq_id='q')
        for j in range(n):
            inclinations.append(np.sqrt(p[j]**2+q[j]**2))
        return np.array(inclinations)

    def get_perihelion_precession_rates(self, A, eccentricities, h_list, k_list):
        n = len(self.planets)
        d_pidot_dt_list = []

        for j in range(n):
            h_dot_j, k_dot_j = 0, 0
            for i in range(n):
                h_dot_j += A[j, i]*k_list[i]
                k_dot_j -= A[j, i]*h_list[i]
            pidot_j = (k_list[j]*h_dot_j - h_list[j]*k_dot_j)/(eccentricities[j])**2
            # pidot_j = ma.masked_array(pidot_j, np.abs(pidot_j)>10).filled(0)

            d_pidot_dt_list.append(pidot_j)
        return d_pidot_dt_list

    


    def get_ascending_node_precession_rates(self, B, inclinations, p_list, q_list):
        n = len(self.planets)
        d_Omega_dt_list = []
        masks = []

        for j in range(n):
            p_dot_j, q_dot_j = 0, 0
            for i in range(n):
                p_dot_j += B[j, i]*q_list[i]
                q_dot_j -= B[j, i]*p_list[i]
            pidot_j = 3600*(q_list[j]*p_dot_j - p_list[j]*q_dot_j)/(inclinations[j])**2
            # pidot_j = ma.masked_array(pidot_j, np.abs(pidot_j)>10).filled(0)

            d_Omega_dt_list.append(pidot_j)
        return d_Omega_dt_list

    def get_pi_or_omega(self, hp, kq):
        pi_om = []
        for i in range(len(self.planets)):
            pi_om.append(np.arctan2(hp[i], kq[i]))
        return np.array(pi_om)

    def kep2cart(self, ecc, inc, h_list, k_list, p_list, q_list, time, t0, idx):
        O_list = self.get_pi_or_omega(p_list, q_list)
        w_list = O_list-self.get_pi_or_omega(h_list, k_list)

        n = self.get_property_all_planets('n')#/(SECS_IN_YEAR)*np.pi/180
        a = self.get_property_all_planets('a')
        # l = self.get_property_all_planets('l')*np.pi/180
        # pi = self.get_property_all_planets('omega_bar')*np.pi/180

        # # T = 2*np.pi*np.sqrt(((a[idx]*AU)**3)/(G_CONST*self.star_mass*M_SUN))/SECS_IN_YEAR
        # # print(T)

        # M0 = l[idx]-pi[idx]
        Mt = n[idx]*np.pi/180*(time-t0)

        EA = []
        e, w, O, i = ecc[idx], w_list[idx], O_list[idx], inc[idx]
        for t in range(len(time)):
            E = Mt[t]
            f_by_dfdE = (E-e[t]*np.sin(E)-Mt[t])/(1-e[t]*np.cos(E))
            j, maxIter, delta = 0, 30, 0.0000000001
            while (j < maxIter)*(np.abs(f_by_dfdE) > delta):
                E = E-f_by_dfdE
                f_by_dfdE = (E-e[t]*np.sin(E)-Mt[t])/(1-e[t]*np.cos(E))
                j += 1
            EA.append(E)
        EA = np.array(EA)
        nu = 2*np.arctan2(np.sqrt(1+e)*np.sin(EA/2), np.sqrt(1-e)*np.cos(EA/2))

        rc = a[idx]*(1-e*np.cos(EA))
        # print('a: {}, {}\nr_max: {}, {}\nr_min {}, {}\n'.format(a[idx], np.mean(rc), a[idx]*(1+np.max(e)), np.max(rc), a[idx]*(1-np.min(e)), np.min(rc)))

        o_vec = np.array([rc*np.cos(nu), rc*np.sin(nu), 0])

        rx = (o_vec[0]*(np.cos(w)*np.cos(O)-np.sin(w)*np.cos(i)*np.sin(O))-
              o_vec[1]*(np.sin(w)*np.cos(O)+np.cos(w)*np.cos(i)*np.sin(O)))
        ry = (o_vec[0]*(np.cos(w)*np.sin(O)+np.sin(w)*np.cos(i)*np.cos(O))+
              o_vec[1]*(np.cos(w)*np.cos(i)*np.cos(O)-np.sin(w)*np.sin(O)))
        rz = (o_vec[0]*(np.sin(w)*np.sin(i)) + o_vec[1]*(np.cos(w)*np.sin(i)))

        return rx, ry, rz

    def simulate(self, t, plot_orbit=False, plot=False, separate=True):
        A, B = [self.frequency_matrix(matrix_id=mat_id, J2=-6.84*10**(-7), J4=2.8*10**(-12)) for mat_id in ['A', 'B']]
        # print('\n', A, B)
        g, x, f, y = *np.linalg.eig(A), *np.linalg.eig(B)
        S, beta, T, gamma = self.find_all_scaling_factor_and_phase(x, y)
        # print(g*100*180/np.pi*3600, '\n')

        eccentricities = self.get_eccentricity(S*x, g, beta, t)
        inclinations = self.get_inclination(T*y, f, gamma, t)
        names = [self.planets[p].name for p in range(len(self.planets))]
        if plot:
            if separate:
                plot_simulation_separate(t/10**6, eccentricities, 'Time (Myr)', 'Eccentricity', names)
                plot_simulation_separate(t/10**6, inclinations, 'Time (Myr)', 'Inclination', names)
            else:
                plot_simulation_all(t/10**6, eccentricities, 'Time (Myr)', 'Eccentricity', names)
                plot_simulation_all(t/10**6, inclinations, 'Time (Myr)', 'Inclination', names)

        kwargs = {'scaled_eigenvector' : S*x, 'eigenvalue' : g, 'phase' : beta,
                  't' : t}
        h_list = self.eq_of_motion(**kwargs, eq_id='h')
        k_list = self.eq_of_motion(**kwargs, eq_id='k')
        kwargs = {'scaled_eigenvector' : T*y, 'eigenvalue' : f, 'phase' : gamma,
                    't' : t}
        p_list = self.eq_of_motion(**kwargs, eq_id='p')#*180/np.pi
        q_list = self.eq_of_motion(**kwargs, eq_id='q')#*180/np.pi

        if plot_orbit:
            x, y, z = np.zeros(2), np.zeros(2), np.zeros(2)
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(x, y, z, 'b*', markersize=3, zorder=-999)
            # ax = fig.add_subplot(111)
            # ax.plot(0, 0, 'b*', markersize=3)
            for idx in range(len(self.planets)-5):
                X, Y, Z = self.kep2cart(eccentricities, inclinations, h_list, k_list, p_list, q_list, t, 0, idx)
                # ax.plot(X, Y, '.', markersize=2, label=names[idx])
                ax.plot(X, Y, Z, '.', markersize=2, label=names[idx], zorder=-idx)
                ax.set_zlim(-np.max([np.max(X), np.max(Y)])/2, np.max([np.max(X), np.max(Y)])/2)
                ax.set_zlabel('z (AU)')
                plt.xlabel('x (AU)')
                plt.ylabel('y (AU)')
            # print(np.mean(np.sqrt(X**2+Y**2+Z**2)))
            # plt.legend()

        precession_rates, xlabel = self.get_perihelion_precession_rates(A, eccentricities, h_list, k_list), 'Pericenter'
        # precession_rates, xlabel = self.get_ascending_node_precession_rates(B, inclinations*180/np.pi, p_list, q_list), 'Ascending node'

        # inclinations *= 180/np.pi
        # idx = 0
        # plot_precession_rate(t, precession_rates[idx], xlabel+r" precession rate [${}'{}'\ y^{-1}$]", names[idx])
        # plot_eccentricity(t, eccentricities[idx], names[idx])

        for idx in range(len(precession_rates)):
            print('Precession rate of {} = {:.4f} arcseconds per century'.format(names[idx],
                  np.mean(precession_rates[idx])*180/np.pi*3600*100))
            # print('Eccentricity of {} = {:.2f}'.format(names[idx],
            #       np.mean(eccentricities[idx])))
            # print('Inclination of {} = {:.2f} degrees'.format(names[idx],
            #       np.mean(inclinations[idx])*180/np.pi))

        return eccentricities, inclinations
    
    


def plot_simulation_separate(t_data=None, y_data=None, xlabel="", ylabel="", data_labels=None):
    for idx in range(len(star_system.planets)):
        plt.figure()
        plt.plot(t_data, y_data[idx], 'b-', label=data_labels[idx])
        plt.axvline(linestyle='--', color='r')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.axis(xmin=t_data[0], xmax=t_data[-1])
        l = plt.legend(loc='upper left', bbox_to_anchor=(0., 1.0),
                ncol=3, fancybox=True, shadow=False, facecolor='black', prop={'size' : 12})
        for text in l.get_texts():
            text.set_color("white")
        plt.subplots_adjust(top=0.95, right=0.95, left=0.11, bottom=0.11)
        # plt.savefig('Logbook/'+ylabel+'_'+data_labels[idx])

def plot_simulation_all(t_data=None, y_data=None, xlabel="", ylabel="", data_labels=None):
    plt.figure()
    for idx in range(len(star_system.planets)):
        plt.plot(t_data, y_data[idx], '-', label=data_labels[idx])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axis(xmin=t_data[0], xmax=t_data[-1], ymin=0, ymax=1.3*np.max(y_data))
    l = plt.legend(loc='upper left', bbox_to_anchor=(0., 1.0),
            ncol=3, fancybox=True, shadow=False, facecolor='black')
    for text in l.get_texts():
        text.set_color("white")

def plot_precession_rate(t, precession_rate, xlabel, label):
    # precession_rates = self.get_ascending_node_precession_rates(B, inclinations, p_list, q_list)
    plt.figure()
    plt.plot((t/10**6), precession_rate, 'b', label=label)
    # plt.axhline(5.462, 0, 1, color='k', linestyle='--')
    plt.axhline(np.mean(precession_rate), 0, 1, color='r', linestyle='--')
    plt.xlabel('Time (Myr)')
    plt.ylabel(xlabel)
    plt.axis(xmin=t[0]/10**6, xmax=t[-1]/10**6)
    l = plt.legend(loc='upper left', bbox_to_anchor=(0., 1.0),
            ncol=3, fancybox=True, shadow=False, facecolor='black',
            handlelength=0, handletextpad=0)
    for text in l.get_texts():
        text.set_color("white")

def plot_eccentricity(t, eccentricity, label):
    plt.figure()
    plt.plot((t/10**6), eccentricity, 'b', linewidth=1, label=label)
    plt.xlabel('Time (Myr)')
    plt.ylabel(r"Eccentricity")
    plt.axis(xmin=t[0]/10**6, xmax=t[-1]/10**6)
    l = plt.legend(loc='upper left', bbox_to_anchor=(0., 1.0),
            ncol=3, fancybox=True, shadow=False, facecolor='black',
            handlelength=0, handletextpad=0)
    for text in l.get_texts():
        text.set_color("white")

if __name__ == "__main__":
    star_system = solar_System(1., 1., 'solar_system.csv')
    # star_system.print_planets()
    # t = np.linspace(-10*10**6, 10*10**6, 10000)
    # print(t[100]-t[0])
    t = np.linspace(-5*10**6, 5*10**6, 10000)
    # t = np.linspace(0, 1000, 1508)
    eccentricities, inclinations = star_system.simulate(t=t, plot_orbit=False, plot=False, separate=False)

    plt.show()

