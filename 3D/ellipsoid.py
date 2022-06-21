import numpy as np
from raster_geometry import ellipsoid as ellipsoid_gen
from mayavi import mlab
from skimage.measure import marching_cubes
from scipy.interpolate import griddata


# Isolates corner vertices
def isolate_vertices(values, block_size=2):
    vertices = []
    mask = np.zeros_like(values, dtype=bool)
    for x in range(values.shape[0] - block_size + 1):
        for y in range(values.shape[1] - block_size + 1):
            for z in range(values.shape[2] - block_size + 1):
                blocksum = np.sum(values[x : x + block_size, y : y + block_size, z : z + block_size])
                if not (blocksum == 0 or blocksum == block_size**3):
                    mask[x : x + block_size, y : y + block_size, z : z + block_size] = True
                if blocksum == 1 or blocksum == block_size**3 - 1:
                    vertices.append(
                        [
                            x + (block_size - 1) / 2,
                            y + (block_size - 1) / 2,
                            z + (block_size - 1) / 2,
                            blocksum / block_size**3,
                        ]
                    )
    vertices = np.vstack(vertices)
    return mask, vertices


# Generate an ellipse in Cartesian coordinates
res = 30
# ellipsoid = ellipsoid_gen(res, (res / 2 - 1, res / 3, res / 4)).astype(float)
ellipsoid = np.zeros([res] * 3)
for xx in range(res):
    for yy in range(res):
        for zz in range(res):
            shift = res / 2
            x = (xx - shift) / 2
            y = (yy - shift) / 2
            z = (zz - shift) / 2
            if z - x**2 + y**2 < 0:
                ellipsoid[xx, yy, zz] = 1
k = 1
res *= k + 2
for dim in range(3):
    ellipsoid = ellipsoid.repeat(k, axis=dim)
ellipsoid = np.pad(ellipsoid, 5, constant_values=0)
X, Y, Z = np.indices(ellipsoid.shape)


# Plot the ellipse voxels and contour
vertices, faces, _, _ = marching_cubes(ellipsoid, spacing=(1, 1, 1))
mlab.points3d(ellipsoid, mode="cube", scale_factor=1, colormap="coolwarm")
mlab.triangular_mesh(vertices[:, 0] + res, vertices[:, 1], vertices[:, 2], faces, colormap="coolwarm")

# Isolate corner vertices and plot them
mask, vertices = isolate_vertices(ellipsoid, block_size=2)
mlab.points3d(
    vertices[:, 0] + res * 2,
    vertices[:, 1],
    vertices[:, 2],
    vertices[:, 3] + 1,
    mode="cube",
    scale_factor=1,
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

mlab.points3d(X + res * 3, Y, Z + res, interpolated_ellipsoid, mode="cube", scale_factor=1, colormap="coolwarm")

vert, face, _, _ = marching_cubes(interpolated_ellipsoid, spacing=(1, 1, 1), level=0.5)
mlab.triangular_mesh(vert[:, 0] + res * 3, vert[:, 1], vert[:, 2], face, colormap="coolwarm")

mlab.show()
