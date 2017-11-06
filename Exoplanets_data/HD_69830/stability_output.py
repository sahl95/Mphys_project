import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
from mpl_toolkits.axes_grid1 import make_axes_locatable

data = pd.read_csv('Exoplanets_data/HD_69830/stability_times_m0.3.csv')
a, e = np.linspace(0.1, 1.2, 140), np.linspace(0, 0.3, 30)
a_grid, e_grid = np.meshgrid(a, e)
data = np.array(data)

data = data[:, 1:]
print(a_grid.shape, e_grid.shape, data.shape)
rbf = scipy.interpolate.Rbf(a_grid, e_grid, data, function='linear')

n = 175
xi, yi = np.linspace(0.1, 1.2, n), np.linspace(0, 0.3, n)
xi, yi = np.meshgrid(xi, yi)
zi = rbf(xi, yi)

fig, ax = plt.subplots()
im = plt.imshow(data, extent=(np.min(a), np.max(a), np.min(e), np.max(e)))
plt.xlabel('a (AU)')
plt.ylabel('e')
divider = make_axes_locatable(ax)
cax = divider.append_axes('top', size='15%', pad=0.55)
fig.colorbar(im, cax=cax, orientation='horizontal',label='Time')

fig, ax = plt.subplots()
im = plt.imshow(zi, extent=(np.min(a), np.max(a), np.min(e), np.max(e)), vmin=0, vmax=10**5)
plt.xlabel('a (AU)')
plt.ylabel('e')
divider = make_axes_locatable(ax)
cax = divider.append_axes('top', size='15%', pad=0.55)
fig.colorbar(im, cax=cax, orientation='horizontal',label='Time')
plt.show()
# print(data.shape)