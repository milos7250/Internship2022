import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, ListedColormap

from functions_2D import (
    plot_contours,
    interpolate_low_output_resolution,
    isolate_contour_datapoints,
    fix_contours,
    clip_to_data,
)
from helpers.save_figure_position import ShowLastPos


"""
This script implements a test for 2D interpolation techniques
"""


height_levels = 50

# Import data. Reversing y axis is necessary to make sure north stays on the top of the graphs.
# NN17 is an ordnance tile from Fort William, NO44 north of Dundee, NO51 in St Andrews, NO33 in Dundee
datagrid = np.loadtxt("data/NO44.asc", skiprows=5)[::-1, :]
# Each tile is of dimension 10km x 10km, sampled by 50m, thus we have 200 x 200 samples
x = np.linspace(0, 10, datagrid.shape[1])
y = np.linspace(0, 10, datagrid.shape[0])
maximum = int(np.ceil(np.max(datagrid)))
levels = [x for x in range(0, maximum + height_levels, height_levels)]
maximum = levels[-1]

cmap = plt.get_cmap("terrain")
norm = Normalize(0, maximum)
colors = ScalarMappable(norm=norm, cmap=cmap)

plt.figure()
plt.pcolormesh(x, y, datagrid, cmap=cmap, norm=norm, rasterized=True)
plt.colorbar(colors, ticks=levels)

datagrid = height_levels * np.floor(datagrid / height_levels)

plt.figure()
plt.pcolormesh(x, y, datagrid, cmap=cmap, norm=norm, rasterized=True)
plt.figure()
plot_contours(x, y, datagrid, low_output_resolution=True, interpolate=False, ax=plt.gca(), colors=colors)


interpolated_datagrid = interpolate_low_output_resolution(x, y, datagrid, method="rfb_thin_plate_spline")

plt.figure()
plt.pcolormesh(x, y, interpolated_datagrid, cmap=cmap, norm=norm, rasterized=True)
plt.figure()
plot_contours(x, y, interpolated_datagrid, levels=levels, interpolate=False, ax=plt.gca(), colors=colors)

ShowLastPos(plt)
