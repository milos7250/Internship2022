import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy import interpolate
from mayavi import mlab

from functions_2D import clip_to_data, interpolate_low_output_resolution, plot_contours

"""
This script creates smooth contours by attempting to interpolate the data itself first, then creating smooth contours.
"""


block_stride = 8
height_levels = 50

# Import data. Reversing y axis is necessary to make sure north stays on the top of the graphs.
# NN17 is an ordnance tile from Fort William, NO44 north of Dundee, NO51 in St Andrews, NO33 in Dundee
datagrid = np.loadtxt("data/NN17.asc", skiprows=5)[::-1, :]
datagrid = np.rot90(datagrid)
# Each tile is of dimension 10km x 10km, sampled by 50m, thus we have 200 x 200 samples
x = np.linspace(0, 10000, datagrid.shape[1])
y = np.linspace(0, 10000, datagrid.shape[0])
maximum = int(np.ceil(np.max(datagrid)))
levels = [i for i in range(0, maximum + height_levels, height_levels)]
maximum = levels[-1]

cmap = plt.get_cmap("terrain")
vmin = -0.25 * maximum * 1.1
vmax = maximum * 1.1
warp_scale = 2
norm = Normalize(vmin, vmax)  # Leave extra 10% for interpolation overshoot
colors = ScalarMappable(norm=norm, cmap=cmap)


# Set up subplots
fig, axes = plt.subplots(nrows=3, ncols=4)
for ax in axes.flatten():
    ax.set_aspect("equal", "box")
plt.tight_layout()
plt.subplots_adjust(hspace=0.5)
plt.colorbar(colors, ticks=levels, ax=axes.ravel().tolist())

# Plot original data
plt.figure()
plt.pcolormesh(x, y, datagrid, rasterized=True, cmap=cmap, norm=norm)
plt.gca().set_title("Original Data")
plt.gca().set_aspect("equal", "box")
mlab.surf(x, y, datagrid, warp_scale=warp_scale, colormap="terrain", vmin=vmin, vmax=vmax)

# # Change spatial resolution
datagrid2 = datagrid[::block_stride, ::block_stride]
x2 = np.linspace(0, 10000, datagrid2.shape[1])
y2 = np.linspace(0, 10000, datagrid2.shape[0])
# axes[0, 0].pcolormesh(x2, y2, datagrid2, rasterized=True, cmap=cmap, norm=norm)
# axes[0, 0].set_title("Lower Spatial Resolution")
# mlab.figure()
# mlab.surf(x2, y2, datagrid2, warp_scale=10, colormap="terrain", vmin=vmin, vmax=vmax)
#
# # Interpolate data with changed spatial resolution using bivariate cubic splines
# interpolated_datagrid2 = interpolate.RectBivariateSpline(x2, y2, datagrid2)(x, y)
# axes[0, 1].pcolormesh(x, y, interpolated_datagrid2, rasterized=True, cmap=cmap, norm=norm)
# axes[0, 1].set_title("Lower Spatial Resolution\nInterpolated")
# plot_contours(x, y, interpolated_datagrid2, interpolate=True, ax=axes[0, 2], colors=colors, levels=levels)
# axes[0, 2].set_title("Contours from Interpolated Data")
# plot_contours(x2, y2, datagrid2, interpolate=True, ax=axes[0, 3], colors=colors, levels=levels)
# axes[0, 3].set_title("Contours from Non-Interpolated Data")
# mlab.figure()
# mlab.surf(x, y, interpolated_datagrid2, warp_scale=10, colormap="terrain", vmin=vmin, vmax=vmax)
#
# # Change measuring resolution
# datagrid3 = height_levels * np.floor(datagrid / height_levels)
# axes[1, 0].pcolormesh(x, y, datagrid3, rasterized=True, cmap=cmap, norm=norm)
# axes[1, 0].set_title("Lower Measuring Resolution")
# mlab.figure()
# mlab.surf(x, y, datagrid3, warp_scale=10, colormap="terrain", vmin=vmin, vmax=vmax)
#
# # Interpolate using thin plate splines
# interpolated_datagrid3 = interpolate_low_output_resolution(x, y, datagrid3, method="rbf_thin_plate_spline")
# axes[1, 1].pcolormesh(x, y, interpolated_datagrid3, rasterized=True, cmap=cmap, norm=norm)
# axes[1, 1].set_title("Lower Measuring Resolution\nInterpolated")
# plot_contours(x, y, interpolated_datagrid3, interpolate=True, ax=axes[1, 2], colors=colors, levels=levels)
# axes[1, 2].set_title("Contours from Interpolated Data")
# plot_contours(x, y, datagrid3, interpolate=True, ax=axes[1, 3], colors=colors, low_output_resolution=True)
# axes[1, 3].set_title("Contours from Non-Interpolated Data")
# mlab.figure()
# mlab.surf(x, y, interpolated_datagrid3, warp_scale=10, colormap="terrain", vmin=vmin, vmax=vmax)


# Change measuring and spatial resolution
datagrid4 = height_levels * np.floor(datagrid[::block_stride, ::block_stride] / height_levels)
axes[2, 0].pcolormesh(x2, y2, datagrid4, rasterized=True, cmap=cmap, norm=norm)
axes[2, 0].set_title("Lower Spatial and Measuring Resolution")
mlab.figure()
mlab.surf(x2, y2, datagrid4, warp_scale=warp_scale, colormap="terrain", vmin=vmin, vmax=vmax)

# Interpolate using thin plate splines
interpolated_datagrid4 = interpolate_low_output_resolution(
    x2,
    y2,
    datagrid4,
    x,
    y,
    method="rbf_thin_plate_spline",
    use_fix_contours=False,
    use_clip_to_data=False,
)
# axes[2, 1].pcolormesh(x, y, interpolated_datagrid4, rasterized=True, cmap=cmap, norm=norm)
# axes[2, 1].set_title("Lower Spatial and Measuring Resolution\nInterpolated")
# plot_contours(x, y, interpolated_datagrid4, interpolate=True, ax=axes[2, 2], colors=colors, levels=levels)
# axes[2, 2].set_title("Contours from Interpolated Data")
# plot_contours(x2, y2, datagrid4, interpolate=False, ax=axes[2, 3], colors=colors, low_output_resolution=True)
# axes[2, 3].set_title("Contours from Non-Interpolated Data")
mlab.figure()
mlab.surf(x, y, interpolated_datagrid4, warp_scale=warp_scale, colormap="terrain", vmin=vmin, vmax=vmax)

mlab.show()
plt.show()
