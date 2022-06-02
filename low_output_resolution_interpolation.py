import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from functions_2D import plot_contours, interpolate_low_output_resolution, isolate_datapoints
from helpers import save_figure_position


"""
This script investigates how different interpolation techniques compare when interpolating data with low output
resolution.
"""


height_levels = 50

# Import data. Reversing y axis is necessary to make sure north stays on the top of the graphs.
# NN17 is an ordnance tile from Fort William, NO44 north of Dundee, NO51 in St Andrews, NO33 in Dundee
datagrid = np.loadtxt("data/NO44.asc", skiprows=5)[::-1, :]
x = np.arange(datagrid.shape[0])
y = np.arange(datagrid.shape[1])
maximum = int(np.ceil(np.max(datagrid)))
levels = [x for x in range(height_levels // 2, maximum, height_levels)]

cmap = plt.get_cmap("terrain")
norm = Normalize(0, maximum)
colors = ScalarMappable(norm=norm, cmap=cmap)
plt.figure()
plt.pcolormesh(x, y, datagrid, rasterized=True, cmap=cmap, norm=norm)

datagrid = height_levels * np.floor(datagrid / height_levels)
plt.figure()
plt.pcolormesh(x, y, datagrid, rasterized=True, cmap=cmap, norm=norm)

isolated_datapoints = isolate_datapoints(datagrid, borders=True)
plt.figure()
plt.scatter(
    isolated_datapoints[:, 0],
    isolated_datapoints[:, 1],
    c=isolated_datapoints[:, 2],
    rasterized=True,
    cmap=cmap,
    norm=norm,
    s=5,
)
plt.colorbar(colors)
plt.gca().set_aspect("equal", "box")

for method in ["linear", "rfb_thin_plate_spline", "rfb_linear"]:
    interpolated_datagrid = interpolate_low_output_resolution(datagrid, method=method, smoothing=0.1)
    plt.figure()
    plt.pcolormesh(x, y, interpolated_datagrid, rasterized=True, cmap=cmap, norm=norm)
    plt.figure()
    plot_contours(interpolated_datagrid, levels, interpolate=True, ax=plt.gca(), colors=colors)

save_figure_position.ShowLastPos(plt)
