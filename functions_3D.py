import numpy as np
from raster_geometry import sphere as sphere_gen


def linear(x0, y0, x1, y1, x):
    return (y1 - y0) / (x1 - x0) * (x - x0) + y0


def remove_duplicate_vertices(vertices: np.ndarray, faces: np.ndarray, tolerance=1e-3):
    """
    Used to remove duplicate vertices from output returned by marching cube algorithm.

    :param vertices: (N, 3) array of vertices
    :param faces: (M, 3) array of faces
    :param tolerance: The maximum euclidean distance for points to be considered identical
    :return: ((N - n), (M - m)) tuple of arrays, with n removed vertices and m removed degenerate faces
    """
    # Reassign faces to the smallest index of a point and average their coordinates
    same = np.linalg.norm(vertices - vertices[:, np.newaxis], axis=2) < tolerance
    same[np.triu_indices_from(same)] = False
    pts_remove, pts_keep = same.nonzero()
    for p_remove, p_keep in zip(pts_remove, pts_keep):
        faces[np.unravel_index(np.nonzero(faces.flatten() == p_remove)[0], faces.shape)] = p_keep
        vertices[p_keep] = (vertices[p_remove] + vertices[p_keep]) / 2

    # Remove repeated points from vertices and adjust face numbers accordingly
    delete = np.unique(pts_remove)
    for idx in reversed(delete):
        faces[np.unravel_index(np.nonzero(faces.flatten() > idx)[0], faces.shape)] -= 1
    vertices = np.delete(vertices, delete, axis=0)

    # Remove degenerate faces
    delete = []
    for idx, face in enumerate(faces):
        if np.unique(face).size != faces.shape[1]:
            delete.append(idx)
    faces = np.delete(faces, delete, axis=0)

    return vertices, faces


class CartesianSphere:
    def __init__(self, resolution: int, radius=1, pad: int = 1, centre=(0, 0, 0)):
        self.resolution = resolution
        self.radius = radius
        self.pad = pad
        self.centre = centre
        self.values = sphere_gen(resolution + 2 * pad, resolution / 2).astype(int)
        self.spacing = 2 * radius / (resolution - 1)
        points = np.linspace(
            -radius - pad * self.spacing,
            +radius + pad * self.spacing,
            resolution + 2 * pad,
        )
        self.grid = (points.copy() + self.centre[0], points.copy() + self.centre[1], points.copy() + self.centre[2])
        self.meshgrid = np.meshgrid(*self.grid)
        self.x_from_index = (
            lambda index: linear(
                0, -radius - pad * self.spacing, resolution + 2 * pad, +radius + pad * self.spacing, index
            )
            + self.centre[0]
        )
        self.y_from_index = (
            lambda index: linear(
                0, -radius - pad * self.spacing, resolution + 2 * pad, +radius + pad * self.spacing, index
            )
            + self.centre[1]
        )
        self.z_from_index = (
            lambda index: linear(
                0, -radius - pad * self.spacing, resolution + 2 * pad, +radius + pad * self.spacing, index
            )
            + self.centre[2]
        )

    def __str__(self):
        return self.__dict__.__str__()


class SphericalSphere:
    def __init__(self, resolution: int or [int, int, int], radius=1, pad: int = 1):
        if type(resolution) is int:
            self.r_resolution = self.theta_resolution = self.phi_resolution = resolution
            self.resolution = [resolution] * 3
        else:
            self.r_resolution, self.theta_resolution, self.phi_resolution = resolution
            self.resolution = resolution
        self.radius = radius
        self.pad = pad
        self.values = np.zeros((self.r_resolution + pad, self.theta_resolution, self.phi_resolution + 1))
        self.values[0 : self.r_resolution] = 1
        # self.r_from_index = lambda r: linear(0, radius / self.r_resolution, self.r_resolution - 1, radius, r)
        self.r_from_index = lambda r: linear(0, 0, self.r_resolution - 1, radius, r)
        self.theta_from_index = lambda theta: linear(0, 0, self.theta_resolution - 1, np.pi, theta)
        self.phi_from_index = lambda phi: linear(0, 0, self.phi_resolution, 2 * np.pi, phi)
