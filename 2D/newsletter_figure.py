import numpy as np
from matplotlib import pyplot as plt

from functions_2D import plot_contours
from helpers.colors import DEMScalarMappable

"""
This script creates smooth contours without attempting to interpolate the data itself.
"""

block_stride = 8
tiles = [
    ["NO33", "Dundee West", 50, 0, False],
    ["NO44", "South of Forfar", 50, 0, False],
    ["NN17", "Fort William", 150, 0, False],
    ["NH52", "Drumnadrochit, Loch Ness", 100, 15.3, True],
    ["NO51", "St Andrews", 50, 0, False],
]
tile, tile_name, height_levels, water_level, zero_is_water = tiles[0]
dim_x = 10 * 1e3  # Dimensions of loaded data in m
dim_y = 10 * 1e3  # Dimensions of loaded data in m


# Set up Figure
plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=20)
size = 1
fig = plt.figure(figsize=(16 * size, 9 * size))
ax = fig.subplots(
    1,
    1,
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
fig.tight_layout()
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
vmin = minimum / 1.1 if minimum > 0 else minimum * 1.1  # Leave extra 10% for interpolation overshoot
vmax = maximum * 1.1  # Leave extra 10% for interpolation overshoot
colors = DEMScalarMappable(vmin, vmax, water_level, zero_is_water)

plt.colorbar(
    colors.segmented(contour_levels, kind="middle"),
    ticks=contour_levels,
    format="%d m",
    ax=ax,
)


plot_contours(
    x, y, datagrid, levels=levels, interpolate=True, ax=ax, colors=colors, discretized_data=False, linewidth=8
)
plt.savefig(f"images/newsletter/Contour.svg", transparent=True, dpi=300, bbox_inches="tight")
plt.show()
