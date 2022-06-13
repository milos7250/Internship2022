import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import helpers.save_figure_position
from functions_2D import interpolate_low_output_resolution, plot_contours

block_stride = 8
height_levels = 50


# Generate a small bump
maximum = 75
x = np.linspace(-2, 2, 400)
y = np.linspace(-2, 2, 400)
X, Y = np.meshgrid(x, y)
z = np.exp(-(X**2 + Y**2)) * maximum

levels = [i for i in range(0, maximum + height_levels, height_levels)]
maximum = levels[-1]

cmap = plt.get_cmap("terrain")
norm = Normalize(-0.25 * maximum * 1.1, maximum * 1.1)  # Leave extra 10% for interpolation overshoot
colors = ScalarMappable(norm=norm, cmap=cmap)
fig, axes = plt.subplots(nrows=2, ncols=3)
for ax in axes.flatten():
    ax.set_aspect("equal", "box")
axes = axes.flatten()

# axes[0].pcolormesh(x, y, z, cmap=cmap, norm=norm)
plt.subplot(231, projection="3d")
plt.gca().plot_surface(X, Y, z, cmap=cmap, norm=norm)

z_discrete = height_levels * np.floor(z / height_levels)

axes[1].pcolormesh(x, y, z_discrete, cmap=cmap, norm=norm)

plot_contours(x, y, z_discrete, colors=colors, ax=axes[2], low_output_resolution=True, interpolate=False)

z_interp = interpolate_low_output_resolution(x, y, z_discrete, use_fix_contours=False, use_clip_to_data=False)
axes[3].pcolormesh(x, y, z_interp, cmap=cmap, norm=norm)

norm_diff = Normalize(-np.max(np.abs(z_interp - z)), np.max(np.abs(z_interp - z)))
axes[4].pcolormesh(x, y, z_interp - z, cmap="coolwarm", norm=norm_diff)
plt.colorbar(ScalarMappable(cmap="coolwarm", norm=norm_diff), ax=axes[4])


plt.subplot(236, projection="3d")
plt.gca().plot_surface(X, Y, z_interp, cmap=cmap, norm=norm)

helpers.save_figure_position.ShowLastPos(plt)
