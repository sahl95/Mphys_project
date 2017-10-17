import numpy as np
import pylab
import glob
import os
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation
import pandas as pd


# a = np.random.rand(2500, 3)*10
# t = np.array([np.ones(100)*i for i in range(25)]).flatten()
# t, h = np.linspace(-12, 0, 1006, retstep=True)
files = glob.glob('Animate_solar_system/*.csv')
files.sort(key=os.path.getmtime)
files = "\n".join(files).split('\n')

print(files)
n_planets = 4
colours = ['r', 'orange', 'b', 'g', 'brown', 'r', 'orange', 'b', 'g']

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# title = ax.set_title('3D Test')
ttl = ax.text(0, 1.05, 0, 'Time = years', transform = ax.transAxes, va='center')

all_x, all_y, all_z = [], [], []
points = []
for idx in range(n_planets):
    points.append(None)
    df = pd.read_csv(files[idx])
    # print(len(df))
    # df = df[df['time'] ]
    # print(len(df))

    x, y, z = np.array(df.x), np.array(df.y), np.array(df.z)
    all_x.append(x)
    all_y.append(y)
    all_z.append(z)
# print(x)
# print(len(all_x))

def update_graph(num):
    global points
    global ttl
    data=df
    # print(data)
    for idx in range(n_planets):
        if points[idx] is not None:
            points[idx].set_color(colours[idx])
            points[idx].set_markersize(1)
        points[idx], = ax.plot(all_x[idx][num:num+1], all_y[idx][num:num+1], all_z[idx][num:num+1], linestyle="", marker="o", color=colours[idx])
    ttl.set_text('Time = {:.2f} years'.format(df.time[num]))
    if num%5 == 0 and num > 10:
        # print(num)
        if num%25 == 0:
            plt.cla()
            ttl = ax.text(0, 1.05, 0, '', transform = ax.transAxes, va='center')
            ax.plot(x0, y0, z0, 'b*', markersize=3, zorder=-999)
            ax.set_zlabel('z (AU)')
            ax.set_xlabel('x (AU)')
            ax.set_ylabel('y (AU)')
            # for idx in range(n_planets):
            #     ax.plot(all_x[idx][:num-1], all_y[idx][:num-1], all_z[idx][:num-1], linestyle="", marker="o", color=colours[idx], markersize=1)
        
        ttl.set_text('Time = {:.2f} years'.format(df.time[num]))
        for idx in range(n_planets):
            ax.plot(all_x[idx][:num-1], all_y[idx][:num-1], all_z[idx][:num-1], linestyle="", marker="o", color=colours[idx], markersize=1)
        ax.set_zlim(-np.max([np.max(all_x[idx]), np.max(all_y[idx])]), np.max([np.max(all_x[idx]), np.max(all_y[idx])]))
        ax.set_ylim(-np.max([np.max(all_x[idx]), np.max(all_y[idx])]), np.max([np.max(all_x[idx]), np.max(all_y[idx])]))
        ax.set_xlim(-np.max([np.max(all_x[idx]), np.max(all_y[idx])]), np.max([np.max(all_x[idx]), np.max(all_y[idx])]))
    # ax.set_title('Time={:.2f} years'.format(num*h), 1, 1)
    return graph, 


# data=df[df['time']==0]
# graph, = ax.plot([],[],[], linestyle="", marker=".")
x0, y0, z0 = np.zeros(2), np.zeros(2), np.zeros(2)
graph, = ax.plot(x0, y0, z0, 'b*', markersize=3, zorder=-999)
ax.view_init(45, 45)
# print(df)
ax.set_zlim(-np.max([np.max(x), np.max(y)]), np.max([np.max(x), np.max(y)]))
ax.set_ylim(-np.max([np.max(x), np.max(y)]), np.max([np.max(x), np.max(y)]))
ax.set_xlim(-np.max([np.max(x), np.max(y)]), np.max([np.max(x), np.max(y)]))
ax.set_zlabel('z (AU)')
ax.set_xlabel('x (AU)')
ax.set_ylabel('y (AU)')
ani = matplotlib.animation.FuncAnimation(fig, update_graph, len(x), 
                               interval=1, save_count=50, repeat=False) 

ani.save('Animate_solar_system/Plots/no_ven_jup.mp4', fps=24)
# plt.show()
