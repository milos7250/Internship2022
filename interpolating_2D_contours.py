import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from functions_2D import plot_contours

"""
This script creates smooth contours without attempting to interpolate the data itself.
"""


block_stride = 8
height_levels = 50

# Import data. Reversing y axis is necessary to make sure north stays on the top of the graphs.
# NN17 is an ordnance tile from Fort William, NO44 north of Dundee, NO51 in St Andrews, NO33 in Dundee
datagrid = np.loadtxt("data/NO44.asc", skiprows=5)[::-1, :]
# Each tile is of dimension 10km x 10km, sampled by 50m, thus we have 200 x 200 samples
x = np.linspace(0, 10, datagrid.shape[1])
y = np.linspace(0, 10, datagrid.shape[0])
maximum = int(np.ceil(np.max(datagrid)))
levels = [i for i in range(0, maximum + height_levels, height_levels)]
maximum = levels[-1]

# Set up subplots and colormaps
fig, axes = plt.subplots(nrows=3, ncols=4)
for ax in axes.flatten():
    ax.set_aspect("equal", "box")
cmap = plt.get_cmap("terrain")
norm = Normalize(min(np.min(datagrid) * 1.1, -10), maximum * 1.1)  # Leave extra 10% for interpolation overshoot
colors = ScalarMappable(norm=norm, cmap=cmap)
plt.subplots_adjust(hspace=0.3)
plt.colorbar(colors, ticks=levels, ax=axes.ravel().tolist())


def plot_contours_wrap(x, y, data, axes, plot_title, levels: list = None, low_output_resolution: bool = False):
    plot_contours(
        x,
        y,
        data,
        levels=levels,
        interpolate=False,
        ax=axes[0],
        colors=colors,
        plot_title=plot_title,
        low_output_resolution=low_output_resolution,
    )
    plot_contours(
        x,
        y,
        data,
        levels=levels,
        interpolate=True,
        ax=axes[1],
        colors=colors,
        plot_title=plot_title,
        low_output_resolution=low_output_resolution,
    )
    axes[2].pcolormesh(x, y, data, rasterized=True, cmap=cmap, norm=norm)
    axes[2].set_title("Color Mesh")


plot_contours_wrap(x, y, datagrid, axes[:, 0], "Original Data", levels=levels)

# Change spatial resolution
datagrid2 = datagrid[::block_stride, ::block_stride]
x2 = np.linspace(0, 10, datagrid2.shape[1])
y2 = np.linspace(0, 10, datagrid2.shape[0])
plot_contours_wrap(x2, y2, datagrid2, axes[:, 2], "Lower Spatial Resolution", levels=levels)

# Change measuring resolution
datagrid3 = height_levels * np.floor(datagrid / height_levels)
plot_contours_wrap(x, y, datagrid3, axes[:, 1], "Lower Measuring Resolution", low_output_resolution=True)

# Change spatial and measuring resolution
datagrid4 = height_levels * np.floor(datagrid[::block_stride, ::block_stride] / height_levels)
plot_contours_wrap(x2, y2, datagrid4, axes[:, 3], "Lower Spatial and Measuring Resolution", low_output_resolution=True)

plt.show()
