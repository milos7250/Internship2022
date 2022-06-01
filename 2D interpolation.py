import matplotlib.cm
import numpy as np
from matplotlib import pyplot as plt
from skimage import measure
from scipy import interpolate

from functions import smooth_contour

block_stride = 5
height_levels = 50

# Import data. Reversing y axis is necessary to make sure north stays on the top of the graphs.
# NN17 is an ordnance tile from Fort William, NO44 north of Dundee, NO51 in St Andrews, NO33 in Dundee
data = np.loadtxt("data/NO44.asc", skiprows=5)[::-1, :]
maximum = int(np.ceil(np.max(data)))

# Set up subplots and colormaps
fig, axes = plt.subplots(nrows=4, ncols=4)
for ax in axes.flatten():
    ax.set_aspect("equal", "box")
cmap = plt.get_cmap("terrain")
norm = matplotlib.colors.Normalize(0, maximum)
colors = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
levels = [x for x in range(0, maximum, height_levels)]
plt.subplots_adjust(hspace=0.3)
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
    # Add boundaries to the isolated contours
    contours = np.append(
        contours,
        np.concatenate([[[0, x] for x in range(data.shape[0])], data[:, 0].T[:, np.newaxis]], axis=1),
        axis=0,
    )
    contours = np.append(
        contours,
        np.concatenate([[[data.shape[1], x] for x in range(data.shape[0])], data[:, -1].T[:, np.newaxis]], axis=1),
        axis=0,
    )
    contours = np.append(
        contours,
        np.concatenate([[[x, 0] for x in range(data.shape[1])], data[0, :].T[:, np.newaxis]], axis=1),
        axis=0,
    )
    contours = np.append(
        contours,
        np.concatenate([[[x, data.shape[0]] for x in range(data.shape[1])], data[-1, :].T[:, np.newaxis]], axis=1),
        axis=0,
    )
    return contours


x = np.arange(data.shape[0])
y = np.arange(data.shape[1])
plt.figure()
plt.pcolormesh(x, y, data, rasterized=True, cmap=cmap, norm=norm)
plt.gca().set_title("Original Data")
plt.gca().set_aspect("equal", "box")

# Change spatial resolution
data2 = data[::block_stride, ::block_stride]
x2 = np.arange(data2.shape[0]) * block_stride
y2 = np.arange(data2.shape[1]) * block_stride
axes[0, 0].pcolormesh(x2, y2, data2, rasterized=True, cmap=cmap, norm=norm)
axes[0, 0].set_title("Lower Spatial Resolution")

# Interpolate data with changed spatial resolution using bivariate cubic splines
interpolated2 = interpolate.RectBivariateSpline(x2, y2, data2)
axes[0, 1].pcolormesh(x, y, interpolated2(x, y), rasterized=True, cmap=cmap, norm=norm)
axes[0, 1].set_title("Lower Spatial Resolution\nInterpolated")
plot_contours(interpolated2(x, y), axes[0, 2])
axes[0, 2].set_title("Contours from Interpolated Data")
plot_contours(data2, axes[0, 3])
axes[0, 3].set_title("Contours from Non-Interpolated Data")

# Change measuring resolution
data3 = height_levels * np.floor(data / height_levels)
axes[1, 0].pcolormesh(x, y, data3, rasterized=True, cmap=cmap, norm=norm)
axes[1, 0].set_title("Lower Measuring Resolution")
axes[2, 0].pcolormesh(x, y, data3, rasterized=True, cmap=cmap, norm=norm)
axes[2, 0].set_title("Lower Measuring Resolution")

# Isolate contours and borders from the data
isolated3 = isolate_contours(data3)
X, Y = np.meshgrid(x, y)

# Interpolate the data by using linear interpolation
# More info: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LinearNDInterpolator.html
interpolated3 = interpolate.griddata(isolated3[:, 0:2], isolated3[:, 2], (X, Y), method="linear")
axes[1, 1].pcolormesh(x, y, interpolated3, rasterized=True, cmap=cmap, norm=norm)
axes[1, 1].set_title("Interpolated Data, Linear")
plot_contours(interpolated3, axes[1, 2])
axes[1, 2].set_title("Contours from Interpolated Data")
plot_contours(data3, axes[1, 3])
axes[1, 3].set_title("Contours from Non-Interpolated Data")

# Interpolate the data by using piecewise cubic interpolating Bezier polynomials
# More info: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CloughTocher2DInterpolator.html
interpolated3 = interpolate.griddata(isolated3[:, 0:2], isolated3[:, 2], (X, Y), method="cubic")
axes[2, 1].pcolormesh(x, y, interpolated3, rasterized=True, cmap=cmap, norm=norm)
axes[2, 1].set_title("Interpolated Data, Cubic")
plot_contours(interpolated3, axes[2, 2])
axes[2, 2].set_title("Contours from Interpolated Data")
plot_contours(data3, axes[2, 3])
axes[2, 3].set_title("Contours from Non-Interpolated Data")

# Change measuring and spatial resolution
data4 = data3[::block_stride, ::block_stride]
axes[3, 0].pcolormesh(x2, y2, data4, rasterized=True, cmap=cmap, norm=norm)
axes[3, 0].set_title("Lower Spatial and Measuring Resolution")

isolated4 = isolate_contours(data4)
X, Y = np.meshgrid(x, y)

# Interpolate the data by using linear interpolation
interpolated4 = interpolate.griddata(isolated4[:, 0:2] * block_stride, isolated4[:, 2], (X, Y), method="linear")
axes[3, 1].pcolormesh(x, y, interpolated4, rasterized=True, cmap=cmap, norm=norm)
axes[3, 1].set_title("Interpolated Data, Linear")
plot_contours(interpolated4, axes[3, 2])
axes[3, 2].set_title("Contours from Interpolated Data")
plot_contours(data4, axes[3, 3])
axes[3, 3].set_title("Contours from Non-Interpolated Data")


# Interpolate data with changed spatial resolution using bivariate cubic splines
# interpolated4 = interpolate.RectBivariateSpline(x2, y2, data4)
# axes[3, 1].pcolormesh(x, y, interpolated4(x, y), rasterized=True, cmap=cmap, norm=norm)
# axes[3, 1].set_title("Lower Spatial and Measuring Resolution\nInterpolated")
# plot_contours(interpolated4(x, y), axes[3, 2])
# axes[3, 2].set_title("Contours from Interpolated Data")
# plot_contours(data4, axes[3, 3])
# axes[3, 3].set_title("Contours from Non-Interpolated Data")

plt.show()
