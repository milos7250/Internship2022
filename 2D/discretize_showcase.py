import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, BoundaryNorm, LinearSegmentedColormap
from scipy import interpolate
from mayavi import mlab

from functions_2D import interpolate_discretized_data, plot_contours, isolate_contour_datapoints
from helpers.colors import DEMScalarMappable

"""
This script creates 3D graphics to showcase data discretization
"""


block_stride = 8
tiles = [
    ["NO33", "Dundee West", 50, 0, False, 5, 45],
    ["NO44", "South of Forfar", 50, 0, False, 5, 45],
    ["NN17", "Fort William", 150, 0, False, 2, 225],
    ["NH52", "Drumnadrochit, Loch Ness", 100, 15.3, True, 5, 225],
    ["NO51", "St Andrews", 50, 0, False, 5, 45],
]
tile, tile_name, height_levels, water_level, zero_is_water, warp_scale, azimuth = tiles[2]
dim_x = 10 * 1e3  # Dimensions of loaded data in m
dim_y = 10 * 1e3  # Dimensions of loaded data in m

# Import data. Reversing y axis is necessary to make sure north stays on the top of the graphs.
datagrid = np.loadtxt(f"data/{tile}.asc", skiprows=5)[::-1, :]
# Each tile is of dimension 10km x 10km, sampled by 50m, thus we have 200 x 200 samples
x = np.linspace(0, dim_x, datagrid.shape[1])
y = np.linspace(0, dim_y, datagrid.shape[0])
minimum = int(np.floor(np.min(datagrid) / height_levels) * height_levels)
maximum = int(np.ceil(np.max(datagrid) / height_levels) * height_levels)
levels = np.arange(minimum, maximum + 1e-10, height_levels, dtype=int)
contour_levels = levels

# Set up Colormaps
vmin = minimum / 1.1 if minimum > 0 else minimum * 1.1  # Leave extra 10% for interpolation overshoot
vmax = maximum * 1.1  # Leave extra 10% for interpolation overshoot
colors = DEMScalarMappable(vmin, vmax, water_level, zero_is_water)

# Discretize Data
discretized_datagrid = height_levels * (np.floor(datagrid / height_levels))

mlab.options.offscreen = True
# 1
mlab.figure(bgcolor=(1, 1, 1))
surf = mlab.surf(
    y, x, np.rot90(discretized_datagrid.T), warp_scale=warp_scale, colormap="terrain", vmin=vmin, vmax=vmax
)
# Using colormap for contours is a dirty fix for mayavi using weird lookup table values
surf.module_manager.scalar_lut_manager.lut.table = colors.segmented_lut(levels, kind="middle")
mlab.draw()
mlab.gcf().scene._lift()
mlab.view(azimuth=azimuth, distance="auto")
mlab.savefig(f"../images/discretize/Discretized.png", magnification=10)

# 2
mlab.figure(bgcolor=(1, 1, 1))
surf = mlab.surf(y, x, np.rot90(datagrid.T), warp_scale=warp_scale, colormap="terrain", vmin=vmin, vmax=vmax)
surf.module_manager.scalar_lut_manager.lut.table = colors.segmented_lut(levels, kind="bottom")
mlab.draw()
mlab.gcf().scene._lift()
mlab.view(azimuth=azimuth, distance="auto")
mlab.savefig(f"../images/discretize/Color_Graded.png", magnification=10)

# 3
mlab.figure(bgcolor=(1, 1, 1))
surf = mlab.surf(y, x, np.rot90(datagrid.T), warp_scale=warp_scale, colormap="terrain", vmin=vmin, vmax=vmax)
surf.module_manager.scalar_lut_manager.lut.table = colors.lut
contours = isolate_contour_datapoints(x, y, np.rot90(datagrid, 3), levels=levels, return_separate_contours=True)
for contour in contours:
    obj = mlab.plot3d(
        contour[:, 0],
        contour[:, 1],
        contour[:, 2] * warp_scale,
        contour[:, 2],
        tube_radius=20,
        colormap="terrain",
        vmin=vmin,
        vmax=vmax,
    )
    obj.module_manager.scalar_lut_manager.lut.table = colors.lut
"""
Contour alternative:

surf = mlab.contour_surf(
    y,
    x,
    np.rot90(discretized_datagrid.T),
    warp_scale=warp_scale,
    colormap="terrain",
    vmin=vmin,
    vmax=vmax,
    contours=contour_levels[:-1].tolist(),
    line_width=3,
)
"""
mlab.gcf().scene._lift()
mlab.view(azimuth=azimuth, distance="auto")
mlab.savefig(f"../images/discretize/Accurate.png", magnification=10)
# Use ImageMagick to remove background from images and crop out fully transparent region.
for image in os.listdir(f"../images/discretize/"):
    os.system(f"convert images/discretize/{image} -transparent white -trim +repage images/discretize/{image}")
