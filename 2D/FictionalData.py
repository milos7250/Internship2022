import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mayavi import mlab

from functions_2D import interpolate_discretized_data, plot_contours

"""
This script interpolates a map of fictional landscape by using a combination of RBF interpolation and cubic spline
interpolation.
"""

# Load the data
datagrid = np.load("../data/FictionalData.npz", mmap_mode="c", allow_pickle=False)["arr_0"]
maximum = np.max(datagrid)
x = np.arange(0, datagrid.shape[1])
y = np.arange(0, datagrid.shape[0])
levels = list(np.arange(0, 8))

# Set up subplots and colormaps
cmap = "terrain"
vmin = -0.25 * maximum * 1.1
vmax = maximum * 1.1
norm = Normalize(vmin, vmax)
colors = ScalarMappable(norm=norm, cmap=cmap)
fig, axes = plt.subplots(nrows=2, ncols=3)
for ax in axes.flatten():
    ax.set_aspect("equal", "box")
axes = axes.flatten()

# Plot discretized elevation map
axes[0].pcolormesh(x, y, datagrid, cmap=cmap, norm=norm, rasterized=True)
plot_contours(x, y, datagrid, ax=axes[3], colors=colors, discretized_data=True, interpolate=False)
plt.colorbar(colors, ax=axes[0], ticks=[i for i in range(8)])

# Interpolate elevation map
interpolated_image = interpolate_discretized_data(
    x,
    y,
    datagrid.astype(float) - 1,
    allow_hybrid_interpolation=True,
)
sea_fix = lambda x: np.where(np.all([0 < x, x < 1], axis=0), -(x**3) + 2 * x**2, np.maximum(x, 0))
interpolated_image = sea_fix(interpolated_image)

# Plot interpolated elevation map
axes[1].pcolormesh(x, y, interpolated_image, cmap=cmap, norm=norm, rasterized=True)
plot_contours(x, y, interpolated_image + 1, ax=axes[4], colors=colors, levels=levels, interpolate=False)
plt.colorbar(colors, ax=axes[1], ticks=[i for i in range(8)])

plt.show()

# Show 3D visualization
mlab.surf(x, y, interpolated_image.T, warp_scale=20, colormap="terrain", vmin=vmin, vmax=vmax)
mlab.show()
