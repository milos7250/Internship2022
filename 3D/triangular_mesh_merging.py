from itertools import product

import numpy as np
from mayavi import mlab
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, zoom
from skimage.measure import marching_cubes

from functions_3D import linear

"""
In this script I create and test a method to merge two triangular meshes generated from data with different resolution.
"""


res = 15
res = 2 * res - 1  # Resolution of the low resolution object - has to be odd for this script to work
n = 4  # Multiplier of resolution, the high resolution object will have resolution res * n


# TODO: Very time inefficient, consider replacing the linear() function by something else
def coords_from_indices(
    x_idx: float | NDArray[float],
    y_idx: float | NDArray[float],
    z_idx: float | NDArray[float],
    shape: tuple[int],
) -> NDArray[float]:
    """
    Converts indices of an array to cartesian coordinates. The conversion functions are not general, they have to be
    altered to suit the object.

    :param x_idx: x index or (n) array of x indices
    :param y_idx: y index or (n) array of y indices
    :param z_idx: z index or (n) array of z indices
    :param shape: shape of the array the indices are from
    :return: (1, 3) array of (x, y, z) coordinates or (n, 3) array of x, y and z coordinates of each triplet of indices.
    """
    return np.array(
        [
            linear(0, -res, shape[0] - 1, res, x_idx),
            linear(0, -res, shape[1] - 1, res, y_idx),
            linear(0, -res, shape[2] - 1, res, z_idx),
        ]
    ).T


def merge_meshes(
    vertices_hr: NDArray[float],
    faces_hr: NDArray[float],
    vertices_lr: NDArray[float],
    faces_lr: NDArray[float],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
    average_coordinates: bool = True,
    allow_degenerate: bool = True,
) -> tuple[NDArray[float], NDArray[float]]:
    """
    This function takes the vertices and faces of high and low resolution triangular meshes produced by the
    'skimage.measure._marching_cubes_lewiner.marching_cubes' function and merges them into one. It does so by replacing
    each vertex from the high resolution mesh within the merging region with the closest vertex from the low resolution
    mesh. The vertices and faces arrays are then amended to reflect this replacement and merged together into one to
    produce a single triangular mesh.

    :param vertices_hr: The array of vertices from high resolution mesh.
    :param faces_hr: The array of faces from high resolution mesh.
    :param vertices_lr: The array of vertices from low resolution mesh.
    :param faces_lr: The array of faces from low resolution mesh.
    :param x_range: The (x_min, x_max) tuple specifying the x dimensions of the bounding box for merging region.
    :param y_range: The (y_min, y_max) tuple specifying the y dimensions of the bounding box for merging region.
    :param z_range: The (z_min, z_max) tuple specifying the z dimensions of the bounding box for merging region.
    :param average_coordinates: Whether to average coordinates of the paired vertices. This makes the transition from
    one mesh to another more smooth in the merging region, although it might still leave behind some irregularities.
    Default True.
    :param allow_degenerate: Whether to allow degenerate (i.e. zero-area) triangles in the end-result. Default True.
    If False, degenerate triangles are removed, at the cost of making the algorithm slower.
    :return: The arrays of vertices and faces of the merged triangular mesh.
    """
    # Generate masks identifying vertices within the merging region
    mask_hr1 = np.all([x_range[0], y_range[0], z_range[0]] <= vertices_hr, axis=1)
    mask_hr2 = np.all(vertices_hr <= [x_range[1], y_range[1], z_range[1]], axis=1)
    mask_lr1 = np.all([x_range[0], y_range[0], z_range[0]] <= vertices_lr, axis=1)
    mask_lr2 = np.all(vertices_lr <= [x_range[1], y_range[1], z_range[1]], axis=1)
    mask_hr = np.logical_and(mask_hr1, mask_hr2)
    mask_lr = np.logical_and(mask_lr1, mask_lr2)

    # Pair vertices from the high resolution mesh within the merging region with the closest vertex from the low
    # resolution mesh.
    high_to_low_pairing = np.argmin(
        np.linalg.norm(vertices_hr[mask_hr, np.newaxis] - vertices_lr[mask_lr], axis=2), axis=1
    )

    # Replace the vertices from the high resolution mesh by the vertices they got paired with
    pts_remove = np.nonzero(mask_hr)[0]
    pts_replacement = np.nonzero(mask_lr)[0][high_to_low_pairing]
    for p_remove, p_keep in zip(pts_remove, pts_replacement):
        faces_hr.flat[faces_hr.flat == p_remove] = p_keep + vertices_hr.shape[0]

    # Average the coordinates of paired vertices
    if average_coordinates:
        for idx in range(np.count_nonzero(mask_lr)):
            low_to_high_pairing = high_to_low_pairing == idx
            vertices_lr[np.nonzero(mask_lr)[0][idx]] = (
                vertices_lr[mask_lr][idx] + np.sum(vertices_hr[mask_hr][low_to_high_pairing], axis=0)
            ) / (np.count_nonzero(low_to_high_pairing) + 1)

    # Remove vertices within the merging region from the high resolution mesh and adjust face numbers accordingly
    for idx in reversed(pts_remove):
        faces_hr.flat[faces_hr.flat > idx] -= 1
    vertices_hr = np.delete(vertices_hr, pts_remove, axis=0)

    # Remove degenerate faces if they are not allowed
    if not allow_degenerate:
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

    # Merge the vertices and faces arrays of both meshes into one, adjusting face numbers accordingly
    vertices, faces = np.vstack([vertices_hr, vertices_lr]), np.vstack([faces_hr, faces_lr + vertices_hr.shape[0]])

    return vertices, faces


"""
Generate low and high resolution ellipses
"""
a, b, c = res - 1, res / 1.5, res / 2  # The axis lengths of the ellipse
low_res = np.zeros([res] * 3, dtype=float)
high_res = np.zeros([((res - 1) * n + 1)] * 3, dtype=float)
for i, j, k in product(*[np.arange(shape) for shape in low_res.shape]):
    x, y, z = coords_from_indices(i, j, k, low_res.shape)
    if (x / a) ** 2 + (y / b) ** 2 + (z / c) ** 2 <= 1:
        low_res[i, j, k] = 1
for i, j, k in product(*[np.arange(shape) for shape in high_res.shape]):
    x, y, z = coords_from_indices(i, j, k, high_res.shape)
    if (x / a) ** 2 + (y / b) ** 2 + (z / c) ** 2 <= 1:
        high_res[i, j, k] = 1

# Generate low and high resolution meshgrids with proper coordinates
x, y, z = coords_from_indices(*np.indices(low_res.shape), low_res.shape).T
X, Y, Z = coords_from_indices(*np.indices(high_res.shape), high_res.shape).T


"""
Apply gaussian filter and marching squares after merging
"""
low_res_scaled = zoom(low_res, high_res.shape[0] / low_res.shape[0], order=0, mode="nearest")
low_res_bottom = low_res_scaled[:, :, : high_res.shape[0] // 2]
high_res_top = high_res[:, :, high_res.shape[0] // 2 :]
merged = np.concatenate([low_res_bottom, high_res_top], axis=2).reshape(high_res.shape)
merged = gaussian_filter(merged, 1, mode="nearest")

# Generate and plot the triangular mesh
vertices, faces, _, _ = marching_cubes(merged)
vertices = coords_from_indices(*vertices.T, merged.shape)
mlab.triangular_mesh(
    vertices[:, 0],
    vertices[:, 1],
    vertices[:, 2],
    faces,
    colormap="coolwarm",
)


"""
Apply gaussian filter and marching squares before merging
"""
low_res[:, :, : low_res.shape[0] // 2 + 1] = gaussian_filter(
    low_res[:, :, : low_res.shape[0] // 2 + 1], 2, mode="nearest"
)
high_res[:, :, high_res.shape[0] // 2 :] = gaussian_filter(high_res[:, :, high_res.shape[0] // 2 :], 2, mode="nearest")


"""
Average the layers where low resolution and high resolution objects meet

This is not used anymore as it does not produce nice results. Technically, this also shouldn't be needed as the function
generating both objects should evaluate the same on both objects, regardless of resolution
"""
# glue_layer_high = high_res[:, :, high_res.shape[0] // 2]
# glue_layer_high = zoom(  # Shrinks the high resolution layer
#     glue_layer_high,
#     low_res.shape[0] / high_res.shape[0],
#     order=0,
#     mode="nearest",
# )
# low_res[:, :, low_res.shape[0] // 2] = glue_layer_high
#
# import matplotlib.pyplot as plt
# plt.subplot(221)
# plt.imshow(low_res[:, :, low_res.shape[0] // 2])
# plt.subplot(222)
# plt.imshow(high_res[:, :, high_res.shape[0] // 2])
# plt.subplot(223)
# plt.imshow(low_res[:, :, low_res.shape[0] // 2] >= 0.5)
# plt.subplot(224)
# plt.imshow(high_res[:, :, high_res.shape[0] // 2] >= 0.5)
# plt.show()

"""
Generate meshes from low and high resolution objects and plot them, without and with using the merging function
"""
# Generate the vertices and faces of the meshes
vertices_low, faces_low, _, _ = marching_cubes(low_res[:, :, : low_res.shape[0] // 2 + 1], level=0.5)
vertices_low = coords_from_indices(*vertices_low.T, low_res.shape)
vertices_high, faces_high, _, _ = marching_cubes(high_res[:, :, high_res.shape[0] // 2 :], level=0.5)
vertices_high = coords_from_indices(*(vertices_high + [0, 0, high_res.shape[0] // 2]).T, high_res.shape)

# Combine and plot the meshes without any post-processing
vertices, faces = np.vstack([vertices_high, vertices_low]), np.vstack([faces_high, faces_low + vertices_high.shape[0]])
mlab.triangular_mesh(
    vertices[:, 0] + res * 2,
    vertices[:, 1],
    vertices[:, 2],
    faces,
    colormap="coolwarm",
)

# Combine and plot the meshes using the merge_meshes function
vertices, faces = merge_meshes(
    vertices_high, faces_high, vertices_low, faces_low, (-res, res), (-res, res), (-1 / n, 1 / n)
)
mlab.triangular_mesh(
    vertices[:, 0] + res * 4,
    vertices[:, 1],
    vertices[:, 2],
    faces,
    colormap="coolwarm",
)

mlab.show()
