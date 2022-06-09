import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import helpers.save_figure_position
from functions_2D import interpolate_low_output_resolution, plot_contours

datagrid = np.load("data/FictionalData.npy", allow_pickle=False) - 1
maximum = np.max(datagrid)
x = np.arange(0, datagrid.shape[1])
y = np.arange(0, datagrid.shape[0])
levels = list(np.arange(0, 8))

cmap = "terrain"
norm = Normalize(-0.25 * maximum * 1.1, maximum * 1.1)
colors = ScalarMappable(norm=norm, cmap=cmap)
fig, axes = plt.subplots(nrows=2, ncols=3)
for ax in axes.flatten():
    ax.set_aspect("equal", "box")
axes = axes.flatten()


axes[0].pcolormesh(x, y, datagrid, cmap=cmap, norm=norm, rasterized=True)
plot_contours(x, y, datagrid, ax=axes[3], colors=colors, low_output_resolution=True, interpolate=False)
plt.colorbar(colors, ax=axes[0], ticks=[i for i in range(8)])


interpolated_image = interpolate_low_output_resolution(
    x,
    y,
    datagrid,
    use_clip_to_data=False,
    allow_hybrid_interpolation=True,
)
# interpolated_image = np.where(datagrid == 0, datagrid, interpolated_image)
axes[1].pcolormesh(x, y, interpolated_image, cmap=cmap, norm=norm, rasterized=True)
plt.figure()
plt.axes(projection="3d")
X, Y = np.meshgrid(x, y)
plt.gca().plot_surface(X, Y, interpolated_image, cmap=cmap, norm=norm)
# plot_contours(x, y, interpolated_image, ax=axes[4], colors=colors, levels=levels, interpolate=False)

# purerbf = np.load("data/FictionalData_Interpolated.npy")
# purerbf = np.where(datagrid == 0, datagrid, purerbf)
# axes[2].pcolormesh(x, y, purerbf, cmap=cmap, norm=norm, rasterized=True)
# plot_contours(x, y, datagrid, ax=axes[5], colors=colors, levels=levels, interpolate=False)


# norm_diff = Normalize(-np.max(np.abs(interpolated_image - purerbf)), np.max(np.abs(interpolated_image - purerbf)))
# axes[3].pcolormesh(x, y, interpolated_image - purerbf, cmap="coolwarm", norm=norm_diff)
# plt.colorbar(ScalarMappable(cmap="coolwarm", norm=norm_diff), ax=axes[3])
helpers.save_figure_position.ShowLastPos(plt)
