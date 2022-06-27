import numpy as np
from numpy.typing import NDArray

"""
This file contains various functions useful for working with triangular meshes in 3D
"""


# TODO: Very time inefficient, consider replacing by something else
def linear(x0, y0, x1, y1, x):
    return (y1 - y0) / (x1 - x0) * (x - x0) + y0


# TODO: Definitely not memory friendly, apparently uses about 1GiB of memory for 100 points???
"""
File "/home/milosmicik/Documents/Internship/functions_3D.py", line 38, in remove_duplicate_vertices
    duplicate = np.linalg.norm(vertices - vertices[:, np.newaxis], axis=2) < tolerance
numpy.core._exceptions._ArrayMemoryError: Unable to allocate 4.62 TiB for an array with shape (459865, 459865, 3) and data type float64
"""


def remove_duplicate_vertices(
    vertices: NDArray[float],
    faces: NDArray[float],
    tolerance=1e-3,
    average_coordinates: bool = False,
    allow_degenerate: bool = True,
):
    """
    Used to remove duplicate vertices from output returned by marching cube algorithm.

    :param vertices: (N, 3) array of vertices
    :param faces: (M, 3) array of faces
    :param tolerance: The maximum euclidean distance for points to be considered identical
    :param average_coordinates: Whether to average coordinates of the duplicate vertices. This has the potential to
    produce smoother looking mesh, but is not noticeable and should not be used when tolerance is low. Default False.
    :param allow_degenerate: Whether to allow degenerate (i.e. zero-area) triangles in the end-result. Default True.
    If False, degenerate triangles are removed, at the cost of making the algorithm slower.
    :return: ((N - n), (M - m)) tuple of arrays, with n removed vertices and m removed degenerate faces
    """
    # Reassign faces to the smallest index of a point and average their coordinates
    duplicate = np.linalg.norm(vertices - vertices[:, np.newaxis], axis=2) < tolerance
    duplicate[np.triu_indices_from(duplicate)] = False
    nonzero = duplicate.nonzero()
    if nonzero[0].size == 0:
        return vertices, faces
    pts_remove, pts_replacement = nonzero
    for p_remove, p_replace in zip(pts_remove, pts_replacement):
        faces.flat[faces.flat == p_remove] = p_replace

    # Average the coordinates of paired vertices
    if average_coordinates:
        for idx in pts_replacement:
            duplicated = np.count_nonzero(duplicate[:, idx])
            if duplicated > 0:
                vertices[idx] = (vertices[idx] + np.sum(vertices[np.nonzero(duplicate[:, idx])], axis=0)) / (
                    duplicated + 1
                )

    # Remove repeated points from vertices and adjust face numbers accordingly
    delete = np.unique(pts_remove)
    for idx in reversed(delete):
        faces[np.unravel_index(np.nonzero(faces.flatten() > idx)[0], faces.shape)] -= 1
    vertices = np.delete(vertices, delete, axis=0)

    # Remove degenerate faces
    if not allow_degenerate:
        delete = []
        for idx, face in enumerate(faces):
            if np.unique(face).size != faces.shape[1]:
                delete.append(idx)
        faces = np.delete(faces, delete, axis=0)

    return vertices, faces


def indices_to_coords_1d(ids, vals):
    ids_vals = np.empty_like(ids, dtype=float)
    whole_part = np.floor(ids).astype(int)
    decimal_part = ids - whole_part
    integer_indices = decimal_part == 0
    ids_vals[integer_indices] = vals[whole_part[integer_indices]]
    ids_vals[~integer_indices] = linear(
        x0=whole_part[~integer_indices],
        y0=vals[whole_part[~integer_indices]],
        x1=whole_part[~integer_indices] + 1,
        y1=vals[whole_part[~integer_indices] + 1],
        x=ids[~integer_indices],
    )
    return ids_vals


def indices_to_coords_nd(ids, vals):
    return np.array([indices_to_coords_1d(xi_ids, xi_vals) for xi_ids, xi_vals in zip(ids, vals)]).T
