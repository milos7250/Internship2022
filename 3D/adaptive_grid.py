import itertools

from mayavi import mlab
from skimage.measure import marching_cubes
import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from functions_3D import linear

res = 15
n = 4
a, b, c = res - 1, res / 2, res / 3
a = b = c = res - 1
low_res = np.zeros([res] * 3, dtype=float)
high_res = np.zeros([((res - 1) * n + 1)] * 3, dtype=float)


def coords_from_indices(x_idx, y_idx, z_idx, shape):
    return np.array(
        [
            linear(0, -res, shape[0] - 1, res, x_idx),
            linear(0, -res, shape[1] - 1, res, y_idx),
            linear(0, -res, shape[2] - 1, res, z_idx),
        ]
    ).T


# indices = (slice(2, 6), slice(2, 6), slice(2, 5))
# low_res[indices] = high_res[indices] = True

for i, j, k in itertools.product(*[np.arange(shape) for shape in low_res.shape]):
    x, y, z = coords_from_indices(i, j, k, low_res.shape)
    if (x / a) ** 2 + (y / b) ** 2 + (z / c) ** 2 <= 1:
        low_res[i, j, k] = 1

for i, j, k in itertools.product(*[np.arange(shape) for shape in high_res.shape]):
    x, y, z = coords_from_indices(i, j, k, high_res.shape)
    if (x / a) ** 2 + (y / b) ** 2 + (z / c) ** 2 <= 1:
        high_res[i, j, k] = 1

x, y, z = coords_from_indices(*np.indices(low_res.shape), low_res.shape).T
X, Y, Z = coords_from_indices(*np.indices(high_res.shape), high_res.shape).T

# Apply gaussian filter and marching squares after merging
# low_res_scaled = low_res.repeat(n, axis=0).repeat(n, axis=1).repeat(n, axis=2)[: -n + 1, : -n + 1, : -n + 1]
low_res_scaled = zoom(low_res, high_res.shape[0] / low_res.shape[0], order=0, mode="nearest")
low_res_cut = low_res_scaled[:, :, : high_res.shape[0] // 2]
high_res_cut = high_res[:, :, high_res.shape[0] // 2 :]
merged = np.concatenate([low_res_cut, high_res_cut], axis=2).reshape(high_res.shape)
merged = gaussian_filter(merged - 0.5, 1) + 0.5

vertices, faces, _, _ = marching_cubes(merged)
vertices = coords_from_indices(*vertices.T, merged.shape)
mlab.triangular_mesh(
    vertices[:, 0],
    vertices[:, 1],
    vertices[:, 2],
    faces,
)


# Apply gaussian filter and marching squares before merging
low_res[:, :, : low_res.shape[0] // 2 + 1] = gaussian_filter(
    low_res[:, :, : low_res.shape[0] // 2 + 1], 2, mode="nearest"
)
high_res[:, :, high_res.shape[0] // 2 :] = gaussian_filter(high_res[:, :, high_res.shape[0] // 2 :], 2, mode="nearest")


# glue_layer_high = high_res[:, :, high_res.shape[0] // 2]
# glue_layer_high = zoom(  # Shrinks the high resolution layer
#     glue_layer_high,
#     low_res.shape[0] / high_res.shape[0],
#     order=0,
#     mode="nearest",
# )
# low_res[:, :, low_res.shape[0] // 2] = glue_layer_high

# plt.subplot(221)
# plt.imshow(low_res[:, :, low_res.shape[0] // 2])
# plt.subplot(222)
# plt.imshow(high_res[:, :, high_res.shape[0] // 2])
# plt.subplot(223)
# plt.imshow(low_res[:, :, low_res.shape[0] // 2] >= 0.5)
# plt.subplot(224)
# plt.imshow(high_res[:, :, high_res.shape[0] // 2] >= 0.5)


def merge_meshes(vertices_hr, faces_hr, vertices_lr, faces_lr, x_range, y_range, z_range):
    # Removing vertices from vertices 1!
    mask_hr1 = np.all([x_range[0], y_range[0], z_range[0]] <= vertices_hr, axis=1)
    mask_hr2 = np.all(vertices_hr <= [x_range[1], y_range[1], z_range[1]], axis=1)
    mask_lr1 = np.all([x_range[0], y_range[0], z_range[0]] <= vertices_lr, axis=1)
    mask_lr2 = np.all(vertices_lr <= [x_range[1], y_range[1], z_range[1]], axis=1)
    mask_hr = np.logical_and(mask_hr1, mask_hr2)
    mask_lr = np.logical_and(mask_lr1, mask_lr2)

    high_to_low_pairing = np.argmin(
        np.linalg.norm(vertices_hr[mask_hr, np.newaxis] - vertices_lr[mask_lr], axis=2), axis=1
    )
    pts_remove = np.nonzero(mask_hr)[0]
    pts_keep = np.nonzero(mask_lr)[0][high_to_low_pairing]

    for p_remove, p_keep in zip(pts_remove, pts_keep):
        faces_hr.flat[faces_hr.flat == p_remove] = p_keep + vertices_hr.shape[0]

    for idx in range(np.count_nonzero(mask_lr)):
        low_to_high_pairing = high_to_low_pairing == idx
        print(idx)
        print(vertices_lr[mask_lr][idx])
        print(vertices_hr[mask_hr][low_to_high_pairing])
        vertices_lr[np.nonzero(mask_lr)[0][idx]] = (
            vertices_lr[mask_lr][idx] + np.sum(vertices_hr[mask_hr][low_to_high_pairing], axis=0)
        ) / (np.count_nonzero(low_to_high_pairing) + 1)
        print(vertices_lr[mask_lr][idx])

    # Remove repeated points from vertices1 and adjust face numbers accordingly
    for idx in reversed(pts_remove):
        faces_hr.flat[faces_hr.flat > idx] -= 1
    vertices_hr = np.delete(vertices_hr, pts_remove, axis=0)

    # Remove degenerate faces
    delete = []
    for idx, face in enumerate(faces_hr):
        if np.unique(face).size != faces_hr.shape[1]:
            delete.append(idx)
    faces_hr = np.delete(faces_hr, delete, axis=0)
    delete = []
    for idx, face in enumerate(faces_lr):
        if np.unique(face).size != faces_lr.shape[1]:
            delete.append(idx)
    faces_lr = np.delete(faces_lr, delete, axis=0)

    vertices, faces = np.vstack([vertices_hr, vertices_lr]), np.vstack([faces_hr, faces_lr + vertices_hr.shape[0]])

    return vertices, faces


vertices_low, faces_low, _, _ = marching_cubes(low_res[:, :, : low_res.shape[0] // 2 + 1], level=0.5)
vertices_low = coords_from_indices(*vertices_low.T, low_res.shape)
vertices_high, faces_high, _, _ = marching_cubes(high_res[:, :, high_res.shape[0] // 2 :], level=0.5)
vertices_high = coords_from_indices(*(vertices_high + [0, 0, high_res.shape[0] // 2]).T, high_res.shape)

vertices, faces = np.vstack([vertices_high, vertices_low]), np.vstack([faces_high, faces_low + vertices_high.shape[0]])

mlab.triangular_mesh(
    vertices[:, 0] + res * 2,
    vertices[:, 1],
    vertices[:, 2],
    faces,
)

vertices, faces = merge_meshes(
    vertices_high, faces_high, vertices_low, faces_low, [-res, res], [-res, res], [-0.2, 0.2]
)

mlab.triangular_mesh(
    vertices[:, 0] + res * 4,
    vertices[:, 1],
    vertices[:, 2],
    faces,
)

mlab.show()
