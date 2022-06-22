import numpy as np
from raster_geometry import ellipsoid as ellipsoid_gen
from mayavi import mlab
from skimage.measure import marching_cubes
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve


# Isolates corner vertices
def isolate_vertices(values, block_size=2):
    vertices = []
    mask = np.zeros_like(values, dtype=bool)
    kernel = np.ones([block_size] * 3)  # A rolling sum kernel
    blocksum = convolve(ellipsoid, kernel, mode="valid", method="direct")
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
                            blocksum[x, y, z] / block_size**3,
                        )
                    )
    vertices = np.vstack(vertices)
    return mask, vertices


# Generate an ellipsoid in Cartesian coordinates
res = 50
ellipsoid = ellipsoid_gen(res, (res / 2 - 1, res / 3, res / 4)).astype(float)
# ellipsoid = np.zeros([res] * 3)
# for xx in range(res):
#     for yy in range(res):
#         for zz in range(res):
#             scale = res / 3
#             x = (xx - res / 2) / scale
#             y = (yy - res / 2) / scale
#             z = (zz - res / 2) / scale
#             if z - x**2 + y**2 < 0:
#                 ellipsoid[xx, yy, zz] = 1
k = 1
res = res * k + 2
if k > 1:
    for dim in range(3):
        ellipsoid = ellipsoid.repeat(k, axis=dim)
ellipsoid = np.pad(ellipsoid, 5, constant_values=0)
X, Y, Z = np.indices(ellipsoid.shape)

# Plot the ellipse voxels and contour
vertices, faces, _, _ = marching_cubes(ellipsoid, spacing=(1, 1, 1))
mlab.points3d(ellipsoid, mode="cube", scale_factor=1, colormap="coolwarm")
mlab.triangular_mesh(vertices[:, 0], vertices[:, 1], vertices[:, 2] + res, faces, colormap="coolwarm")
mlab.text3d(res * 1, res * 0.5, res * 2.5, "Original terraced object")

# Use gaussian kernel to smooth the ellipsoid
gaussian_ellipsoid = gaussian_filter(ellipsoid - 0.5, 2) + 0.5

vert, face, _, _ = marching_cubes(gaussian_ellipsoid, spacing=(1, 1, 1), level=0.5)
mlab.triangular_mesh(vert[:, 0] + res * 2, vert[:, 1], vert[:, 2] + res, face, colormap="coolwarm")
mlab.points3d(X + res * 2, Y, Z, gaussian_ellipsoid, mode="cube", scale_factor=1, colormap="coolwarm")
mlab.text3d(res * 3, res * 0.5, res * 2.5, "Gaussian kernel\nsmoothened object")

# Use block averages to smooth the ellipsoid
kernel_size = 2
kernel = np.ones([kernel_size] * 3) / kernel_size**3  # A rolling average kernel
rolling_avg_ellipsoid = convolve(ellipsoid, kernel, mode="same", method="direct")

vert, face, _, _ = marching_cubes(rolling_avg_ellipsoid, spacing=(1, 1, 1), level=0.5)
mlab.triangular_mesh(vert[:, 0] + res * 3, vert[:, 1], vert[:, 2] + res, face, colormap="coolwarm")
mlab.points3d(X + res * 3, Y, Z, rolling_avg_ellipsoid, mode="cube", scale_factor=1, colormap="coolwarm")
mlab.text3d(res * 4, res * 0.5, res * 2.5, "Rolling block average\nsmoothened object")

# Isolate corner vertices and plot them
mask, vertices = isolate_vertices(ellipsoid, block_size=2)
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

interpolated_ellipsoid = ellipsoid.copy()
interpolated_ellipsoid[mask] = gridded_vertices

# Fill in nan's from original ellipse
nan_mask = np.isnan(interpolated_ellipsoid)
interpolated_ellipsoid[nan_mask] = ellipsoid[nan_mask]

mlab.points3d(X + res, Y, Z + res, interpolated_ellipsoid, mode="cube", scale_factor=1, colormap="coolwarm")

vert, face, _, _ = marching_cubes(interpolated_ellipsoid, spacing=(1, 1, 1), level=0.5)
mlab.triangular_mesh(vert[:, 0] + res, vert[:, 1], vert[:, 2] + res * 2, face, colormap="coolwarm")
mlab.text3d(res * 2, res * 0.5, res * 3.5, "Isolation of vertices\nsmoothened object")

mlab.show()
