import numpy as np
from mayavi import mlab
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, BoundaryNorm, LinearSegmentedColormap
from functions_2D import plot_contours
import os

import_step = 1
height_levels = 200

datagrid = Image.open("data/slovakia.tif")
datagrid = np.array(datagrid)[::-1][1200:2200:import_step, 2300:3300:import_step]
datagrid = np.maximum(0, datagrid)
dim_x = 50 * datagrid.shape[1]  # Dimensions of loaded data in m
dim_y = 50 * datagrid.shape[0]  # Dimensions of loaded data in m

minimum = int(np.floor(np.min(datagrid) / height_levels) * height_levels)
maximum = int(np.ceil(np.max(datagrid) / height_levels) * height_levels)
levels = np.arange(minimum, maximum + 1e-10, height_levels, dtype=int)
contour_levels = levels[:-1]

# Set up Figure
plt.rc("text", usetex=True)
plt.rc("font", family="serif")
size = 1
fig = plt.figure(figsize=(16 * size, 9 * size))
fig.tight_layout()
axes = fig.subplots(
    1,
    2,
    sharex="all",
    sharey="all",
    subplot_kw={
        "adjustable": "box",
        "aspect": "equal",
        "xticks": [x for x in np.linspace(0, dim_x, 6)],
        "yticks": [y for y in np.linspace(0, dim_y, 6)],
        "xticklabels": [f"{x:d} km" for x in np.linspace(0, dim_x / 1e3, 6, dtype=int)],
        "yticklabels": [f"{y:d} km" for y in np.linspace(0, dim_x / 1e3, 6, dtype=int)],
        "xlim": (0, dim_x),
        "ylim": (0, dim_y),
    },
    gridspec_kw={"hspace": 0.3},
)
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor="none", which="both", top=False, bottom=False, left=False, right=False)
plt.xlabel("Easting")
plt.ylabel("Northing")


mask = datagrid.T == 0
x = np.arange(0, dim_x, 50)
y = np.arange(0, dim_x, 50)

# Set up Colormaps
cmap = plt.get_cmap("terrain")
vmin = -0.25 * maximum * 1.1
vmax = maximum * 1.1
norm = Normalize(vmin, vmax)  # Leave extra 10% for interpolation overshoot
colors = ScalarMappable(norm=norm, cmap=cmap)
plt.colorbar(colors, ticks=levels, format="%d m", ax=axes[0], shrink=0.7)
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
plt.colorbar(contour_colors, ticks=contour_levels, format="%d m", ax=axes[1], shrink=0.7)

axes[0].pcolormesh(x, y, datagrid, cmap=cmap, norm=norm, rasterized=True)
axes[0].set_title("Zarnovica and Ziar nad Hronom Regions\n50m Resolution Raster")
plot_contours(x, y, datagrid, levels=levels, ax=axes[1], interpolate=True, colors=colors)

try:
    os.mkdir(f"images/Slovakia")
except FileExistsError:
    pass

plt.savefig("images/Slovakia/2D.svg", transparent=True, dpi=300, bbox_inches="tight")
# plt.show()
mlab.options.offscreen = True
mlab.surf(y, x, np.rot90(datagrid.T), mask=mask, warp_scale=5, colormap="terrain", vmin=vmin, vmax=vmax)
mlab.view(azimuth=315)

mlab.savefig("images/Slovakia/3D.png", magnification=10)
