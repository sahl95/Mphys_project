import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.interpolate
from mpl_toolkits.axes_grid1 import make_axes_locatable

# plt.rcParams['xtick.color'] = 'white'
# plt.rcParams['ytick.color'] = 'white'
# plt.rcParams['text.color'] = 'white'
# plt.rcParams['axes.labelcolor'] = 'white'
# # plt.rcParams['savefig.facecolor']='black'
# matplotlib.rc('axes', edgecolor='white')

data = pd.read_csv('Exoplanets_data/HD_69830/Stability/a_0.07_1.2__e_0.0_0.2__m0.7_ii/stability_times_m0.7.csv')
a, e = np.linspace(0.1, 1.2, 120), np.linspace(0, 0.2, 30)
a_grid, e_grid = np.meshgrid(a, e)
data = np.array(data)

data = data[:, 1:]
# print(a_grid.shape, e_grid.shape, data.shape)
rbf = scipy.interpolate.Rbf(a_grid, e_grid, data, function='linear')

n = 435
xi, yi = np.linspace(0.07, 1.2, n), np.linspace(0, 0.2, n)
xi, yi = np.meshgrid(xi, yi)
zi = rbf(xi, yi)

fig, ax = plt.subplots()
im = plt.imshow(zi, extent=(np.min(a), np.max(a), np.min(e), np.max(e)), cmap='RdYlGn', aspect='auto')
plt.xlabel('a (AU)')
plt.ylabel('e')
divider = make_axes_locatable(ax)
cax = divider.append_axes('top', size='5%', pad=0.55)
fig.colorbar(im, cax=cax, orientation='horizontal',label='Time (years)')

# fig, ax = plt.subplots()
# im = plt.imshow(zi, extent=(np.min(a), np.max(a), np.min(e), np.max(e)), cmap='RdYlGn', aspect='auto')
# plt.xlabel('a (AU)')
# plt.ylabel('e')
# divider = make_axes_locatable(ax)
# cax = divider.append_axes('top', size='5%', pad=0.55)
# fig.colorbar(im, cax=cax, orientation='horizontal',label='Time (years)')
plt.savefig('Report/stability_HD_69830.pdf', transparent=True)
# plt.show()
# print(data.shape)