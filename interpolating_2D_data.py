import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, BoundaryNorm, LinearSegmentedColormap
from scipy import interpolate
from mayavi import mlab

from functions_2D import interpolate_discretized_data, plot_contours
from helpers.colors import DEMScalarMappable

"""
This script creates smooth contours by attempting to interpolate the data itself first, then creating smooth contours.
"""

block_stride = 8
tiles = [
    ["NO33", "Dundee West", 50, 0, False, 5, 45],
    ["NO44", "South of Forfar", 50, 0, False, 5, 45],
    ["NN17", "Fort William", 150, 0, False, 2, 225],
    ["NH52", "Drumnadrochit, Loch Ness", 100, 15.3, True, 5, 225],
    ["NO51", "St Andrews", 50, 0, False, 5, 45],
]
tile, tile_name, height_levels, water_level, zero_is_water, warp_scale, azimuth = tiles[3]
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
vmin = minimum / 1.1 if minimum > 0 else minimum * 1.1  # Leave extra 10% for interpolation overshoot
vmax = maximum * 1.1  # Leave extra 10% for interpolation overshoot
colors = DEMScalarMappable(vmin, vmax, water_level, zero_is_water)
ticks = levels
ticks = (*ticks, water_level) if (water_level not in ticks and water_level > vmin) else ticks
plt.colorbar(
    colors,
    ticks=ticks,
    format="%d m",
    ax=axes[0:2, :].ravel().tolist(),
)
plt.colorbar(
    colors.segmented(contour_levels, kind="middle"),
    ticks=contour_levels,
    format="%d m",
    ax=axes[2, :].ravel().tolist(),
    aspect=10,
)


def plot_data_wrap(x, y, datagrid, xi=None, yi=None, interpolated_datagrid=None, axes=None, plot_title=None):
    axes[0].pcolormesh(x, y, datagrid, cmap=colors.cmap, norm=colors.norm, rasterized=True)
    axes[0].set_title(plot_title)
    if interpolated_datagrid is not None:
        axes[1].pcolormesh(xi, yi, interpolated_datagrid, cmap=colors.cmap, norm=colors.norm, rasterized=True)
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

plt.savefig(f"images/{tile}/2D_Data_Interpolation.svg", transparent=True, dpi=300, bbox_inches="tight")
# plt.show()

os.system("zenity --info --text 'Interpolation Finished' --icon-name=emblem-success")
exit()

# 3D plots

mlab.options.offscreen = True

# 1
mlab.figure(bgcolor=(1, 1, 1))
surf = mlab.surf(y, x, np.rot90(datagrid.T), warp_scale=warp_scale, colormap="terrain", vmin=vmin, vmax=vmax)
surf.module_manager.scalar_lut_manager.lut.table = colors.lut
mlab.draw()
mlab.gcf().scene._lift()
mlab.view(azimuth=azimuth, distance="auto")
mlab.savefig(f"images/{tile}/2D_Data_Interpolation_3D_orig.png", magnification=10)

# 2
mlab.figure(bgcolor=(1, 1, 1))
surf = mlab.surf(
    y2,
    x2,
    np.rot90(low_spatial_res_datagrid.T),
    warp_scale=warp_scale,
    colormap="terrain",
    vmin=vmin,
    vmax=vmax,
)
surf.module_manager.scalar_lut_manager.lut.table = colors.lut
mlab.draw()
mlab.gcf().scene._lift()
mlab.view(azimuth=azimuth, distance="auto")
mlab.savefig(f"images/{tile}/2D_Data_Interpolation_3D_low_spatial_res.png", magnification=10)

# 3
mlab.figure(bgcolor=(1, 1, 1))
surf = mlab.surf(
    y,
    x,
    np.rot90(interpolated_low_spatial_res_datagrid.T),
    warp_scale=warp_scale,
    colormap="terrain",
    vmin=vmin,
    vmax=vmax,
)
surf.module_manager.scalar_lut_manager.lut.table = colors.lut
mlab.draw()
mlab.gcf().scene._lift()
mlab.view(azimuth=azimuth, distance="auto")
mlab.savefig(f"images/{tile}/2D_Data_Interpolation_3D_low_spatial_res_interpolated.png", magnification=10)

# 4
mlab.figure(bgcolor=(1, 1, 1))
surf = mlab.surf(
    y, x, np.rot90(discretized_datagrid.T), warp_scale=warp_scale, colormap="terrain", vmin=vmin, vmax=vmax
)
# Using colormap for contours is a dirty fix for mayavi using weird lookup table values
surf.module_manager.scalar_lut_manager.lut.table = colors.segmented_lut(levels, kind="middle")
mlab.draw()
mlab.draw()
mlab.gcf().scene._lift()
mlab.view(azimuth=azimuth, distance="auto")
mlab.savefig(f"images/{tile}/2D_Data_Interpolation_3D_discretized_datagrid.png", magnification=10)

# 5
mlab.figure(bgcolor=(1, 1, 1))
surf = mlab.surf(
    y,
    x,
    np.rot90(interpolated_discretized_datagrid.T),
    warp_scale=warp_scale,
    colormap="terrain",
    vmin=vmin,
    vmax=vmax,
)
surf.module_manager.scalar_lut_manager.lut.table = colors.lut
mlab.draw()
mlab.gcf().scene._lift()
mlab.view(azimuth=azimuth, distance="auto")
mlab.savefig(f"images/{tile}/2D_Data_Interpolation_3D_discretized_datagrid_interpolated.png", magnification=10)

# 6
mlab.figure(bgcolor=(1, 1, 1))
surf = mlab.surf(
    y2,
    x2,
    np.rot90(discretized_low_spatial_res_datagrid.T),
    warp_scale=warp_scale,
    colormap="terrain",
    vmin=vmin,
    vmax=vmax,
)
# Using colormap for contours is a dirty fix for mayavi using weird lookup table values
surf.module_manager.scalar_lut_manager.lut.table = colors.segmented_lut(levels, kind="middle")
mlab.draw()
mlab.gcf().scene._lift()
mlab.view(azimuth=azimuth, distance="auto")
mlab.savefig(f"images/{tile}/2D_Data_Interpolation_3D_discretized_low_spatial_res_datagrid.png", magnification=10)

# 7
mlab.figure(bgcolor=(1, 1, 1))
surf = mlab.surf(
    y,
    x,
    np.rot90(interpolated_discretized_low_spatial_res_datagrid.T),
    warp_scale=warp_scale,
    colormap="terrain",
    vmin=vmin,
    vmax=vmax,
)
surf.module_manager.scalar_lut_manager.lut.table = colors.lut
mlab.draw()
mlab.gcf().scene._lift()
mlab.view(azimuth=azimuth, distance="auto")
mlab.savefig(
    f"images/{tile}/2D_Data_Interpolation_3D_discretized_low_spatial_res_datagrid_interpolated.png", magnification=10
)

# Use ImageMagick to remove background from images and crop out fully transparent region.
for image in os.listdir(f"images/{tile}/"):
    if "2D_Data_Interpolation_3D_" in image:
        os.system(f"convert images/{tile}/{image} -transparent white -trim +repage images/{tile}/{image}")

# mlab.show()
os.system("zenity --info --text 'Interpolation Finished' --icon-name=emblem-success")
