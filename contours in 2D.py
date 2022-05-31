import matplotlib.cm
import numpy as np
from matplotlib import pyplot as plt
from skimage import measure

from functions import smooth_contour

# Import data. Reversing y axis is necessary to make sure north stays on the top of the graphs.
data = np.loadtxt("data/NO44.asc", skiprows=5)[
    ::-1, :
]  # NN17 is an ordnance tile from Fort William, NO44 around Dundee
maximum = int(np.ceil(np.max(data)))

# Set up subplots and colormaps
fig, axes = plt.subplots(nrows=3, ncols=4)
for ax in axes.flatten():
    ax.set_aspect("equal", "box")
cmap = plt.get_cmap("terrain")
norm = matplotlib.colors.Normalize(0, maximum)
colors = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
levels = [x for x in range(0, maximum, 50)]
plt.colorbar(colors, ticks=levels, ax=axes.ravel().tolist())


def plot_contours(data, axes, name):
    # Find contours
    for level in levels:
        for contour in measure.find_contours(data.T, level):
            # Plot unsmoothened contour in first row
            axes[0].plot(contour[:, 0], contour[:, 1], color=colors.to_rgba(level))
            axes[0].set_title(name + "\nUnsmoothened Contours")
            # Plot smoothened contour in second row
            interpolated_contour = smooth_contour(contour)
            axes[1].plot(interpolated_contour[:, 0], interpolated_contour[:, 1], color=colors.to_rgba(level))
            axes[1].set_title("Smoothened Contours")
    # Plot the colormesh in third row
    axes[2].pcolormesh(np.arange(data.shape[0]), np.arange(data.shape[0]), data, rasterized=True, cmap=cmap, norm=norm)
    axes[2].set_title("Color Mesh")


plot_contours(data, axes[:, 0], "Original Data")

# Change spatial resolution
data2 = data[::8, ::8]
plot_contours(data2, axes[:, 2], "Lower Spatial Resolution")

# Change measuring resolution
data3 = 50 * np.floor(data / 50)
plot_contours(data3, axes[:, 1], "Lower Measuring Resolution")

# Change spatial and measuring resolution
data4 = 50 * np.floor(data[::8, ::8] / 50)
plot_contours(data4, axes[:, 3], "Lower Spatial and Measuring Resolution")

plt.show()
