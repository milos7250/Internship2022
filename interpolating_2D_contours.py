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
data = np.loadtxt("data/NO44.asc", skiprows=5)[::-1, :]
maximum = int(np.ceil(np.max(data)))
levels = [x for x in range(0, maximum, height_levels)]

# Set up subplots and colormaps
fig, axes = plt.subplots(nrows=3, ncols=4)
for ax in axes.flatten():
    ax.set_aspect("equal", "box")
cmap = plt.get_cmap("terrain")
norm = Normalize(0, maximum)
colors = ScalarMappable(norm=norm, cmap=cmap)
plt.subplots_adjust(hspace=0.3)
plt.colorbar(colors, ticks=levels, ax=axes.ravel().tolist())


def plot_contours_wrap(data, axes, plot_title):
    plot_contours(data, levels, interpolate=False, ax=axes[0], colors=colors, plot_title=plot_title)
    plot_contours(data, levels, interpolate=True, ax=axes[1], colors=colors, plot_title=plot_title)
    axes[2].pcolormesh(np.arange(data.shape[0]), np.arange(data.shape[0]), data, rasterized=True, cmap=cmap, norm=norm)
    axes[2].set_title("Color Mesh")


plot_contours_wrap(data, axes[:, 0], "Original Data")

# Change spatial resolution
data2 = data[::block_stride, ::block_stride]
plot_contours_wrap(data2, axes[:, 2], "Lower Spatial Resolution")

# Change measuring resolution
data3 = height_levels * np.floor(data / height_levels)
plot_contours_wrap(data3, axes[:, 1], "Lower Measuring Resolution")

# Change spatial and measuring resolution
data4 = height_levels * np.floor(data[::block_stride, ::block_stride] / height_levels)
plot_contours_wrap(data4, axes[:, 3], "Lower Spatial and Measuring Resolution")

plt.show()
