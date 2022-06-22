import itertools

from mayavi import mlab
from skimage.measure import marching_cubes
from raster_geometry import sphere
import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from functions_3D import remove_duplicate_vertices, linear

res = 11
n = 4
low_res = np.zeros([res] * 3, dtype=bool)
high_res = np.zeros([((res - 1) * n + 1)] * 3, dtype=bool)


def coords_from_indices(x_idx, y_idx, z_idx, shape):
    return np.array(
        [
            linear(0, -res, shape[0] - 1, res, x_idx),
            linear(0, -res, shape[1] - 1, res, y_idx),
            linear(0, -res, shape[2] - 1, res, z_idx),
        ]
    ).T


for i, j, k in itertools.product(*[np.arange(shape) for shape in low_res.shape]):
    x, y, z = coords_from_indices(i, j, k, low_res.shape)
    if x**2 + y**2 + z**2 <= (res - 1) ** 2:
        low_res[i, j, k] = True

for i, j, k in itertools.product(*[np.arange(shape) for shape in high_res.shape]):
    x, y, z = coords_from_indices(i, j, k, high_res.shape)
    if x**2 + y**2 + z**2 <= (res - 1) ** 2:
        high_res[i, j, k] = True

x, y, z = np.indices(low_res.shape)
X, Y, Z = np.indices(high_res.shape)


# Apply gaussian filter and marching squares after merging
# low_res_scaled = low_res.repeat(n, axis=0).repeat(n, axis=1).repeat(n, axis=2)[: -n + 1, : -n + 1, : -n + 1]
low_res_scaled = zoom(
    low_res,
    high_res.shape[0] / low_res.shape[0],
    order=0,
    mode="nearest",
    prefilter=False,
    grid_mode=True,
)
low_res_cut = low_res_scaled[:, :, : high_res.shape[0] // 2]
high_res_cut = high_res[:, :, high_res.shape[0] // 2 :]
merged = np.concatenate([low_res_cut, high_res_cut], axis=2).reshape(high_res.shape)
merged = gaussian_filter(merged - 0.5, 1) + 0.5

vertices, faces, _, _ = marching_cubes(merged)
vertices = coords_from_indices(*vertices.T, merged.shape)
mlab.triangular_mesh(
    vertices[:, 0] + res * 2,
    vertices[:, 1],
    vertices[:, 2],
    faces,
)


# Apply gaussian filter and marching squares before merging
low_res = gaussian_filter(low_res - 0.5, 1) + 0.5
high_res = gaussian_filter(high_res - 0.5, 1) + 0.5

glue_layer = low_res[:, :, low_res.shape[0] // 2]
glue_layer = zoom(  # Interpolates the layer to higher resolution linearly
    glue_layer.astype(float),
    high_res.shape[0] / low_res.shape[0],
    order=1,
    mode="nearest",
    prefilter=False,
    grid_mode=True,
)
high_res[:, :, high_res.shape[0] // 2] = glue_layer


vertices_low, faces_low, _, _ = marching_cubes(low_res[:, :, : low_res.shape[0] // 2 + 1], level=0.5)
vertices_low = coords_from_indices(*vertices_low.T, low_res.shape)
vertices_high, faces_high, _, _ = marching_cubes(high_res[:, :, high_res.shape[0] // 2 :], level=0.5)
vertices_high = coords_from_indices(*(vertices_high + [0, 0, high_res.shape[0] // 2]).T, high_res.shape)
vertices, faces = np.vstack([vertices_low, vertices_high]), np.vstack([faces_low, faces_high + vertices_low.shape[0]])

mlab.triangular_mesh(
    vertices[:, 0],
    vertices[:, 1],
    vertices[:, 2],
    faces,
)


mlab.show()
