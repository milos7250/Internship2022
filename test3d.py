import numpy as np
from mayavi import mlab
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from scipy import interpolate


# Import data. Reversing y axis is necessary to make sure north stays on the top of the graphs.
# NN17 is an ordnance tile from Fort William, NO44 north of Dundee, NO51 in St Andrews, NO33 in Dundee
# datagrid = np.loadtxt("data/NO44.asc", skiprows=5)[::-1, :]
# Each tile is of dimension 10km x 10km, sampled by 50m, thus we have 200 x 200 samples
# x = np.linspace(0, 10000, datagrid.shape[1])
# y = np.linspace(0, 10000, datagrid.shape[0])

import_step = 5
datagrid = Image.open("data/slovakia.tif")
datagrid = np.array(datagrid)[::-import_step, ::import_step]
datagrid = np.maximum(0, datagrid)
mask = datagrid.T == 0
x = np.arange(0, datagrid.shape[1]) * 50 * import_step
y = np.arange(0, datagrid.shape[0]) * 50 * import_step

cmap = plt.get_cmap("terrain")
maximum = np.max(datagrid)
vmin = -0.25 * maximum * 1.1
vmax = maximum * 1.1
warp_scale = 2
norm = Normalize(vmin, vmax)  # Leave extra 10% for interpolation overshoot
colors = ScalarMappable(norm=norm, cmap=cmap)

print(datagrid.shape, np.min(datagrid), np.max(datagrid))
plt.pcolormesh(x, y, datagrid, cmap=cmap, norm=norm, rasterized=True)
plt.show()
mlab.surf(x, y, datagrid.T, mask=mask, warp_scale=10, colormap="terrain", vmin=vmin, vmax=vmax)
mlab.show()
exit()
