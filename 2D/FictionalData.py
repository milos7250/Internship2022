import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mayavi import mlab

import helpers.save_figure_position
from functions_2D import interpolate_discretized_data, plot_contours

"""
This script interpolates a map of fictional landscape by using a combination of RBF interpolation and cubic spline
interpolation.
"""

datagrid = np.load("../data/FictionalData.npy", allow_pickle=False)
maximum = np.max(datagrid)
x = np.arange(0, datagrid.shape[1])
y = np.arange(0, datagrid.shape[0])
levels = list(np.arange(0, 8))

cmap = "terrain"
vmin = -0.25 * maximum * 1.1
vmax = maximum * 1.1
norm = Normalize(vmin, vmax)
colors = ScalarMappable(norm=norm, cmap=cmap)
fig, axes = plt.subplots(nrows=2, ncols=3)
for ax in axes.flatten():
    ax.set_aspect("equal", "box")
axes = axes.flatten()


# axes[0].pcolormesh(x, y, datagrid, cmap=cmap, norm=norm, rasterized=True, rasterized=True)
# plot_contours(x, y, datagrid, ax=axes[3], colors=colors, low_output_resolution=True, interpolate=False)
# plt.colorbar(colors, ax=axes[0], ticks=[i for i in range(8)])
#
#

interpolated_image = interpolate_discretized_data(
    x,
    y,
    datagrid - 1,
    allow_hybrid_interpolation=True,
)
sea_fix = lambda x: np.where(np.all([0 < x, x < 1], axis=0), -(x**3) + 2 * x**2, np.maximum(x, 0))
interpolated_image = sea_fix(interpolated_image)
# axes[1].pcolormesh(x, y, interpolated_image, cmap=cmap, norm=norm, rasterized=True, rasterized=True)
mlab.surf(x, y, interpolated_image.T, warp_scale=20, colormap="terrain", vmin=vmin, vmax=vmax)
mlab.show()

# plot_contours(x, y, interpolated_image, ax=axes[4], colors=colors, levels=levels, interpolate=False)

# purerbf = np.load("data/FictionalData_Interpolated.npy")
# purerbf = np.where(datagrid == 0, datagrid, purerbf)
# axes[2].pcolormesh(x, y, purerbf, cmap=cmap, norm=norm, rasterized=True, rasterized=True)
# plot_contours(x, y, datagrid, ax=axes[5], colors=colors, levels=levels, interpolate=False)


# norm_diff = Normalize(-np.max(np.abs(interpolated_image - purerbf)), np.max(np.abs(interpolated_image - purerbf)))
# axes[3].pcolormesh(x, y, interpolated_image - purerbf, cmap="coolwarm", norm=norm_diff, rasterized=True)
# plt.colorbar(ScalarMappable(cmap="coolwarm", norm=norm_diff), ax=axes[3])
# plt.show()
