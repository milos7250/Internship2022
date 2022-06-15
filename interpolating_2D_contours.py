import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, BoundaryNorm, LinearSegmentedColormap

from functions_2D import plot_contours

"""
This script creates smooth contours without attempting to interpolate the data itself.
"""

block_stride = 8
tiles = [
    ["NO33", "Dundee West", 50],
    ["NO44", "South of Forfar", 50],
    ["NN17", "Fort William", 150],
    ["NH52", "Drumnadrochit, Loch Ness", 100],
    ["NO51", "St Andrews", 50],
]
tile, tile_name, height_levels = tiles[0]
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
contour_levels = levels[:-1]

# Set up Colormaps
cmap = plt.get_cmap("terrain")
vmin = -0.25 * maximum * 1.1
vmax = maximum * 1.1
norm = Normalize(vmin, vmax)  # Leave extra 10% for interpolation overshoot
colors = ScalarMappable(norm=norm, cmap=cmap)
plt.colorbar(colors, ticks=levels, format="%d m", ax=axes[0, :2].ravel().tolist(), aspect=10, pad=0.05)
discretized_colors = ScalarMappable(
    norm=BoundaryNorm(levels, levels.size),
    cmap=LinearSegmentedColormap.from_list("", colors.to_rgba(levels), N=levels.size),
)
# print(minimum, maximum, levels)
# for i in np.arange(np.floor(vmin), np.max(vmax), dtype=int):
#     if i % 25 == 0:
#         print(i, np.array(discretized_colors.to_rgba(i)) - np.array(colors.to_rgba(i)))
# exit()
plt.colorbar(discretized_colors, ticks=levels, format="%d m", ax=axes[0, 2:].ravel().tolist(), aspect=10)
contour_colors = ScalarMappable(
    norm=BoundaryNorm(
        [
            1.5 * contour_levels[0] - 0.5 * contour_levels[1],
            *(contour_levels[0:-1] + contour_levels[1:]) / 2,
            1.5 * contour_levels[-1] - 0.5 * contour_levels[-2],
        ],
        contour_levels.size,
    ),
    cmap=LinearSegmentedColormap.from_list("", colors.to_rgba(contour_levels), N=contour_levels.size),
)
plt.colorbar(contour_colors, ticks=contour_levels, format="%d m", ax=axes[1:, :].ravel().tolist())


def plot_contours_wrap(x, y, datagrid, axes, plot_title, levels=None, discretized_data=False):
    plot_contours(
        x,
        y,
        datagrid,
        levels=levels,
        interpolate=False,
        ax=axes[1],
        colors=colors,
        discretized_data=discretized_data,
    )
    plot_contours(
        x,
        y,
        datagrid,
        levels=levels,
        interpolate=True,
        ax=axes[2],
        colors=colors,
        discretized_data=discretized_data,
    )
    axes[0].pcolormesh(x, y, datagrid, cmap=cmap, norm=norm, rasterized=True)
    axes[0].set_title(plot_title)


plot_contours_wrap(x, y, datagrid, axes[:, 0], plot_title=f"{tile_name}\n50m Resolution Raster", levels=levels)

# Lower Spatial Resolution
low_spatial_res_datagrid = datagrid[::block_stride, ::block_stride]
x2 = np.linspace(0, dim_x, low_spatial_res_datagrid.shape[1])
y2 = np.linspace(0, dim_y, low_spatial_res_datagrid.shape[0])
plot_contours_wrap(x2, y2, low_spatial_res_datagrid, axes[:, 1], "Low Spatial Resolution", levels=levels)

# Discretize Data
discretized_datagrid = height_levels * np.floor(datagrid / height_levels)
plot_contours_wrap(x, y, discretized_datagrid, axes[:, 2], "Discretized Data", discretized_data=True)

# Change spatial and measuring resolution
discretized_low_spatial_res_datagrid = height_levels * np.floor(
    datagrid[::block_stride, ::block_stride] / height_levels
)
plot_contours_wrap(
    x2,
    y2,
    discretized_low_spatial_res_datagrid,
    axes[:, 3],
    "Discretized Low Spatial Resolution",
    discretized_data=True,
)

plt.savefig(f"images/{tile}/2D_Contour_Interpolation.svg", transparent=True, dpi=300, bbox_inches="tight")
plt.show()
