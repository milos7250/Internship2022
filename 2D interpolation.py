import matplotlib.cm
import numpy as np
from matplotlib import pyplot as plt
from skimage import measure
from scipy import interpolate

from functions import smooth_contour

# Import data. Reversing y axis is necessary to make sure north stays on the top of the graphs.
data = np.loadtxt("data/NO44.asc", skiprows=5)[
    ::-1, :
]  # NN17 is an ordnance tile from Fort William, NO44 around Dundee
maximum = int(np.ceil(np.max(data)))

# Set up subplots and colormaps
fig, axes = plt.subplots(nrows=2, ncols=4)
for ax in axes.flatten():
    ax.set_aspect("equal", "box")
cmap = plt.get_cmap("terrain")
norm = matplotlib.colors.Normalize(0, maximum)
colors = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
levels = [x for x in range(0, maximum, 50)]
plt.colorbar(colors, ticks=levels, ax=axes.ravel().tolist())


def plot_contours(data, ax):
    """
    Plots contours on the given axes.
    """
    # Find contours
    for level in levels:
        for contour in measure.find_contours(data.T, level):
            # Plot smoothened contour
            interpolated_contour = smooth_contour(contour)
            ax.plot(interpolated_contour[:, 0], interpolated_contour[:, 1], color=colors.to_rgba(level))


def isolate_contours(data):
    """
    Isolates contour points from the image
    """
    contours = np.ndarray((0, 3))
    # Find contours
    for level in levels:
        for contour in measure.find_contours(data.T, level):
            contours = np.append(
                contours,
                np.concatenate([contour, np.array([level] * contour.shape[0]).T[:, np.newaxis]], axis=1),
                axis=0,
            )
    return contours


x = np.arange(data.shape[0])
y = np.arange(data.shape[1])
axes[0, 0].pcolormesh(x, y, data, rasterized=True, cmap=cmap, norm=norm)
axes[0, 0].set_title("Original Data")
axes[1, 0].pcolormesh(x, y, data, rasterized=True, cmap=cmap, norm=norm)
axes[1, 0].set_title("Original Data")

# Change spatial resolution
data2 = data[::8, ::8]
x2 = np.arange(data2.shape[0]) * 8
y2 = np.arange(data2.shape[1]) * 8
axes[0, 1].pcolormesh(x2, y2, data2, rasterized=True, cmap=cmap, norm=norm)
axes[0, 1].set_title("Lower Spatial Resolution")

# Interpolate data with changed spatial resolution using bivariate cubic splines
interpolated2 = interpolate.RectBivariateSpline(x2, y2, data2)
axes[0, 2].pcolormesh(x, y, interpolated2(x, y), rasterized=True, cmap=cmap, norm=norm)
axes[0, 2].set_title("Lower Spatial Resolution\nInterpolated")
plot_contours(interpolated2(x, y), axes[0, 3])
axes[0, 3].set_title("Contours from Interpolated Data")

# Change measuring resolution
data3 = 50 * np.floor(data / 50)
axes[1, 1].pcolormesh(x, y, data3, rasterized=True, cmap=cmap, norm=norm)
axes[1, 1].set_title("Lower Measuring Resolution")
plot_contours(data3, axes[1, 2])
axes[1, 2].set_title("Isolated Contours")

# Attempt to interpolate based on contours using bivariate cubic splines
isolated3 = isolate_contours(data3)
interpolated3 = interpolate.SmoothBivariateSpline(isolated3[:, 0], isolated3[:, 1], isolated3[:, 2])
axes[1, 3].pcolormesh(x, y, interpolated3(x, y).T, rasterized=True, cmap=cmap, norm=norm)
axes[1, 3].set_title("Interpolated Data")


plt.show()
