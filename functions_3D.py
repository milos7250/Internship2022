import numpy as np
from numpy.typing import NDArray

"""
This file contains various functions useful for working with triangular meshes in 3D
"""


# TODO: Very time inefficient, consider replacing by something else
def linear(x0, y0, x1, y1, x):
    return (y1 - y0) / (x1 - x0) * (x - x0) + y0


def sph_to_car(r: float | NDArray[float], theta: float | NDArray[float], phi: float | NDArray[float]) -> NDArray[float]:
    """
    Converts spherical coordinates to cartesian.
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z]).T


def car_to_sph(x: float | NDArray[float], y: float | NDArray[float], z: float | NDArray[float]) -> NDArray[float]:
    """
    Converts cartesian coordinates to spherical.
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y / x)
    return np.array([r, theta, phi]).T


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
    duplicate = np.linalg.norm(vertices - vertices[:, np.newaxis], axis=2) < tolerance
    duplicate[np.triu_indices_from(duplicate)] = False
    nonzero = duplicate.nonzero()
    if nonzero[0].size == 0:
        return vertices, faces
    # Reassign faces to the common point with the smallest index
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
        faces.flat[faces.flat > idx] -= 1
    vertices = np.delete(vertices, delete, axis=0)

    # Remove degenerate faces
    if not allow_degenerate:
        delete = []
        for idx, face in enumerate(faces):
            if np.unique(face).size != faces.shape[1]:
                delete.append(idx)
        faces = np.delete(faces, delete, axis=0)

    return vertices, faces
