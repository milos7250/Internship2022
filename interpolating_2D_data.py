import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, BoundaryNorm, LinearSegmentedColormap
from scipy import interpolate
from mayavi import mlab

from functions_2D import interpolate_discretized_data, plot_contours

"""
This script creates smooth contours by attempting to interpolate the data itself first, then creating smooth contours.
"""

block_stride = 8
height_levels = 50  # Use 150 for Fort William, 50 otherwise

# NN17 - Fort William, NO44 - north of Dundee, NO51 - in St Andrews, NO33 - in Dundee, NH52 - Drumnadrochit
tile = "NH52"
tile_name = "Drumnadrochit, Loch Ness"
dim_x = 10 * 1e3  # Dimensions of loaded data in m
dim_y = 10 * 1e3  # Dimensions of loaded data in m


# Set up Figure
plt.rc("text", usetex=True)
plt.rc("font", family="serif")
size = 1
fig = plt.figure(figsize=(16 * size, 9 * size))
fig.tight_layout()
axes = fig.subplots(
    3,
    4,
    sharex="all",
    sharey="all",
    subplot_kw={
        "adjustable": "box",
        "aspect": "equal",
        "xticks": [i for i in np.linspace(0, dim_x, 6)],
        "yticks": [i for i in np.linspace(0, dim_y, 6)],
        "xticklabels": [f"{i:d} km" for i in np.linspace(0, dim_x / 1e3, 6, dtype=int)],
        "yticklabels": [f"{i:d} km" for i in np.linspace(0, dim_x / 1e3, 6, dtype=int)],
        "xlim": (0, dim_x),
        "ylim": (0, dim_y),
    },
    gridspec_kw={"hspace": 0.3},
)
axes[1, 0].axis(False)
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor="none", which="both", top=False, bottom=False, left=False, right=False)
plt.xlabel("Easting")
plt.ylabel("Northing")

# Import data. Reversing y axis is necessary to make sure north stays on the top of the graphs.
datagrid = np.loadtxt(f"data/{tile}.asc", skiprows=5)[::-1, :]
# Each tile is of dimension 10km x 10km, sampled by 50m, thus we have 200 x 200 samples
x = np.linspace(0, dim_x, datagrid.shape[1])
y = np.linspace(0, dim_y, datagrid.shape[0])
minimum = int(np.floor(np.min(datagrid) / height_levels) * height_levels)
maximum = int(np.ceil(np.max(datagrid) / height_levels) * height_levels)
levels = np.arange(minimum, maximum + 1e-10, height_levels, dtype=int)
contour_levels = levels[:-1] + 25

# Set up Colormaps
cmap = plt.get_cmap("terrain")
vmin = -0.25 * maximum * 1.1
vmax = maximum * 1.1
norm = Normalize(vmin, vmax)  # Leave extra 10% for interpolation overshoot
colors = ScalarMappable(norm=norm, cmap=cmap)
plt.colorbar(colors, ticks=levels, format="%d m", ax=axes[0:2, :].ravel().tolist())
contour_colors = ScalarMappable(
    norm=BoundaryNorm(
        [
            1.5 * contour_levels[0] - 0.5 * contour_levels[1],
            *(contour_levels[0:-1] + contour_levels[1:]) / 2,
            1.5 * contour_levels[-1] - 0.5 * contour_levels[-2],
        ],
        levels.size,
    ),
    cmap=LinearSegmentedColormap.from_list("", colors.to_rgba(contour_levels), N=levels.size),
)
plt.colorbar(contour_colors, ticks=contour_levels, format="%d m", ax=axes[2, :].ravel().tolist(), aspect=10)


def plot_data_wrap(x, y, datagrid, xi=None, yi=None, interpolated_datagrid=None, axes=None, plot_title=None):
    axes[0].pcolormesh(x, y, datagrid, cmap=cmap, norm=norm)
    axes[0].set_title(plot_title)
    if interpolated_datagrid is not None:
        axes[1].pcolormesh(xi, yi, interpolated_datagrid, cmap=cmap, norm=norm)
        axes[1].set_title("Interpolated Data")
        plot_contours(
            xi,
            yi,
            interpolated_datagrid,
            levels=contour_levels,
            interpolate=False,
            ax=axes[2],
            colors=colors,
        )
    else:
        plot_contours(
            x,
            y,
            datagrid,
            levels=contour_levels,
            interpolate=False,
            ax=axes[2],
            colors=colors,
        )


# Plot original data
plot_data_wrap(x, y, datagrid, axes=axes[:, 0], plot_title=f"{tile_name}\n50m Resolution Raster")
# Lower spatial resolution
low_spatial_res_datagrid = datagrid[::block_stride, ::block_stride]
x2 = np.linspace(0, dim_x, low_spatial_res_datagrid.shape[1])
y2 = np.linspace(0, dim_y, low_spatial_res_datagrid.shape[0])

# Interpolate data with changed spatial resolution using bivariate cubic splines
interpolated_low_spatial_res_datagrid = interpolate.RectBivariateSpline(x2, y2, low_spatial_res_datagrid)(x, y)
plot_data_wrap(
    x2,
    y2,
    low_spatial_res_datagrid,
    x,
    y,
    interpolated_low_spatial_res_datagrid,
    axes=axes[:, 1],
    plot_title="Low Spatial Resolution",
)

# Discretize Data
discretized_datagrid = height_levels * np.floor(datagrid / height_levels)

# Interpolate using thin plate splines
interpolated_discretized_datagrid = interpolate_discretized_data(
    x, y, discretized_datagrid, method="rbf_thin_plate_spline"
)
plot_data_wrap(
    x,
    y,
    discretized_datagrid,
    x,
    y,
    interpolated_discretized_datagrid,
    axes=axes[:, 2],
    plot_title="Discretized Data",
)


# Lower spatial resolution and discretize data
discretized_low_spatial_res_datagrid = height_levels * np.floor(
    datagrid[::block_stride, ::block_stride] / height_levels
)

# Interpolate using thin plate splines
interpolated_discretized_low_spatial_res_datagrid = interpolate_discretized_data(
    x2,
    y2,
    discretized_low_spatial_res_datagrid,
    x,
    y,
    method="rbf_thin_plate_spline",
)
plot_data_wrap(
    x2,
    y2,
    discretized_low_spatial_res_datagrid,
    x,
    y,
    interpolated_discretized_low_spatial_res_datagrid,
    axes=axes[:, 3],
    plot_title="Discretized Low Spatial Resolution",
)

try:
    os.mkdir(f"images/{tile}")
except FileExistsError:
    pass

plt.savefig(f"images/{tile}/2D_Data_Interpolation.png")
# plt.show()

os.system("zenity --info --text 'Interpolation Finished' --icon-name=emblem-success")
exit()

# 3D plots
warp_scale = 2  # Use 2 for Fort William, 5 otherwise
azimuth = 225  # Use 225 for Fort William and Drumnadrochit, 0 otherwise

mlab.options.offscreen = True
mlab.figure(bgcolor=(1, 1, 1))
mlab.surf(y, x, np.rot90(datagrid.T), warp_scale=warp_scale, colormap="terrain", vmin=vmin, vmax=vmax)
mlab.view(azimuth=azimuth)
mlab.savefig(f"images/{tile}/2D_Data_Interpolation_3D_orig.png", magnification=10)
mlab.figure(bgcolor=(1, 1, 1))
mlab.surf(
    y2,
    x2,
    np.rot90(low_spatial_res_datagrid.T),
    warp_scale=warp_scale,
    colormap="terrain",
    vmin=vmin,
    vmax=vmax,
)
mlab.view(azimuth=azimuth)
mlab.savefig(f"images/{tile}/2D_Data_Interpolation_3D_low_spatial_res.png", magnification=10)
mlab.figure(bgcolor=(1, 1, 1))
mlab.surf(
    y,
    x,
    np.rot90(interpolated_low_spatial_res_datagrid.T),
    warp_scale=warp_scale,
    colormap="terrain",
    vmin=vmin,
    vmax=vmax,
)
mlab.view(azimuth=azimuth)
mlab.savefig(f"images/{tile}/2D_Data_Interpolation_3D_low_spatial_res_interpolated.png", magnification=10)
mlab.figure(bgcolor=(1, 1, 1))
mlab.surf(y, x, np.rot90(discretized_datagrid.T), warp_scale=warp_scale, colormap="terrain", vmin=vmin, vmax=vmax)
mlab.view(azimuth=azimuth)
mlab.savefig(f"images/{tile}/2D_Data_Interpolation_3D_discretized_datagrid.png", magnification=10)
mlab.figure(bgcolor=(1, 1, 1))
mlab.surf(
    y,
    x,
    np.rot90(interpolated_discretized_datagrid.T),
    warp_scale=warp_scale,
    colormap="terrain",
    vmin=vmin,
    vmax=vmax,
)
mlab.view(azimuth=azimuth)
mlab.savefig(f"images/{tile}/2D_Data_Interpolation_3D_discretized_datagrid_interpolated.png", magnification=10)
mlab.figure(bgcolor=(1, 1, 1))
mlab.surf(
    y2,
    x2,
    np.rot90(discretized_low_spatial_res_datagrid.T),
    warp_scale=warp_scale,
    colormap="terrain",
    vmin=vmin,
    vmax=vmax,
)
mlab.view(azimuth=azimuth)
mlab.savefig(f"images/{tile}/2D_Data_Interpolation_3D_discretized_low_spatial_res_datagrid.png", magnification=10)
mlab.figure(bgcolor=(1, 1, 1))
mlab.surf(
    y,
    x,
    np.rot90(interpolated_discretized_low_spatial_res_datagrid.T),
    warp_scale=warp_scale,
    colormap="terrain",
    vmin=vmin,
    vmax=vmax,
)
mlab.view(azimuth=azimuth)
mlab.savefig(
    f"images/{tile}/2D_Data_Interpolation_3D_discretized_low_spatial_res_datagrid_interpolated.png", magnification=10
)

# Use ImageMagick to remove background from images.
for image in os.listdir(f"images/{tile}/"):
    if "2D_Data_Interpolation_3D_" in image:
        os.system(f"convert images/{tile}/{image} -transparent white images/{tile}/{image}")

# mlab.show()
os.system("zenity --info --text 'Interpolation Finished' --icon-name=emblem-success")
