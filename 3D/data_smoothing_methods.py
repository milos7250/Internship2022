import numpy as np
from mayavi import mlab
from numpy.typing import NDArray
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve
from skimage.measure import marching_cubes

"""
In this script we smooth the data by using either rolling average, gaussian filter or a custom method which
isolates vertices and interpolates data on the edges of the contour.
"""


# Isolates corner vertices
def isolate_vertices(values: NDArray[float], block_size: int = 2) -> tuple[NDArray[bool], NDArray[float]]:
    """
    This method isolates corner vertices from the surface and returns the corner vertices in an array and a
    mask specifying which parts of the values array are on the edge of the surface.

    :param values: The 3D array to isolate vertices from
    :param block_size: The block size to use. This alters how many points from values get identified as those on the
    surface edge.
    :return: The mask and the isolated vertices.
    """
    vertices = []
    mask = np.zeros_like(values, dtype=bool)
    kernel = np.ones([block_size] * 3)  # A rolling sum kernel
    blocksum = convolve(surface, kernel, mode="valid", method="direct")
    for x in range(blocksum.shape[0]):
        for y in range(blocksum.shape[1]):
            for z in range(blocksum.shape[2]):
                if not (blocksum[x, y, z] == 0 or blocksum[x, y, z] == block_size**3):
                    mask[x : x + block_size, y : y + block_size, z : z + block_size] = True
                if blocksum[x, y, z] == 1 or blocksum[x, y, z] == block_size**3 - 1:
                    vertices.append(
                        (
                            x + (block_size - 1) / 2,
                            y + (block_size - 1) / 2,
                            z + (block_size - 1) / 2,
                            0 if blocksum[x, y, z] == 1 else 1,
                        )
                    )
    vertices = np.vstack(vertices)
    return mask, vertices


# Generate an ellipsoid in Cartesian coordinates
res = 30  # The resolution of the surface array
surface_kind = ["ellipsoid", "saddle"][0]  # Change number here to change surface type
surface = np.zeros([res] * 3)
for i in range(res):
    for j in range(res):
        for k in range(res):
            x, y, z = [-res + 2 * (val * (1 + 1 / (res - 1))) for val in (i, j, k)]
            if surface_kind == "ellipsoid":
                a, b, c = res - 0.5, res / 1.5, res / 2
                if (x / a) ** 2 + (y / b) ** 2 + (z / c) ** 2 <= 1:
                    surface[i, j, k] = 1
            elif surface_kind == "saddle":
                x /= res / 2
                y /= res / 2
                z /= res / 2
                if z - x**2 + y**2 < 0:
                    surface[i, j, k] = 1
n = 1  # The factor by which to stretch the surface array
res = res * n + 2
if n > 1:
    for dim in range(3):
        surface = surface.repeat(n, axis=dim)
surface = np.pad(surface, 5, constant_values=0)
X, Y, Z = np.indices(surface.shape)

# Plot the ellipse voxels and contour
vertices, faces, _, _ = marching_cubes(surface, spacing=(1, 1, 1))
mlab.points3d(surface, mode="cube", scale_factor=1, colormap="coolwarm")
mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2] + res, faces, colormap="coolwarm")
mlab.text3d(res * 1, res * 0.5, res * 2.5, "Original terraced object")

# Use gaussian kernel to smooth the ellipsoid
gaussian_surface = gaussian_filter(surface, 2, truncate=1)

vertices, faces, _, _ = marching_cubes(gaussian_surface, spacing=(1, 1, 1), level=0.5)
mlab.triangular_mesh(vertices[:, 0] + res * 2, vertices[:, 1], vertices[:, 2] + res, faces, colormap="coolwarm")
mlab.points3d(X + res * 2, Y, Z, gaussian_surface, mode="cube", scale_factor=1, colormap="coolwarm")
mlab.text3d(res * 3, res * 0.5, res * 2.5, "Gaussian kernel\nsmoothened object")
# Use block averages to smooth the ellipsoid
kernel_size = 2
kernel = np.ones([kernel_size] * 3) / kernel_size**3  # A rolling average kernel
rolling_avg_surface = convolve(surface, kernel, mode="same", method="direct")

vertices, faces, _, _ = marching_cubes(rolling_avg_surface, spacing=(1, 1, 1), level=0.5)
mlab.triangular_mesh(vertices[:, 0] + res * 3, vertices[:, 1], vertices[:, 2] + res, faces, colormap="coolwarm")
mlab.points3d(X + res * 3, Y, Z, rolling_avg_surface, mode="cube", scale_factor=1, colormap="coolwarm")
mlab.text3d(res * 4, res * 0.5, res * 2.5, "Rolling block average\nsmoothened object")

# Isolate corner vertices and plot them
mask, vertices = isolate_vertices(surface, block_size=2)
mlab.points3d(
    vertices[:, 0] + res,
    vertices[:, 1],
    vertices[:, 2],
    vertices[:, 3] + 1,
    mode="cube",
    scale_factor=0.5,
    colormap="coolwarm",
)


# Interpolate the grid from isolated vertices, only interpolate the points specified by mask
gridded_vertices = griddata(
    vertices[:, 0:3],
    vertices[:, 3],
    (X[mask], Y[mask], Z[mask]),
    method="linear",
)

interpolated_surface = surface.copy()
interpolated_surface[mask] = gridded_vertices

# Fill in nan's from original ellipse
nan_mask = np.isnan(interpolated_surface)
interpolated_surface[nan_mask] = surface[nan_mask]

mlab.points3d(X + res, Y, Z + res, interpolated_surface, mode="cube", scale_factor=1, colormap="coolwarm")

vertices, faces, _, _ = marching_cubes(interpolated_surface, spacing=(1, 1, 1), level=0.5)
mlab.triangular_mesh(vertices[:, 0] + res, vertices[:, 1], vertices[:, 2] + res * 2, faces, colormap="coolwarm")
mlab.text3d(res * 2, res * 0.5, res * 3.5, "Isolation of vertices\nsmoothened object")

mlab.show()
