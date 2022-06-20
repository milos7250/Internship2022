from mayavi import mlab
from skimage.measure import marching_cubes
import numpy as np

from functions_3D import remove_duplicate_vertices, linear, SphericalSphere

small = 7
r = theta = small * 2 - 1
phi = small * 2

sphere = SphericalSphere([r, theta, phi])
sphere2 = SphericalSphere(small)


def sph_to_car(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z]).T


def car_to_sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y / x)
    return np.array([r, theta, phi]).T


def sphere_to_car(sphere: SphericalSphere):
    car_points = np.empty((sphere.values.size, 4))
    i = 0
    for r_idx in range(sphere.values.shape[0]):
        for theta_idx in range(sphere.values.shape[1]):
            for phi_idx in range(sphere.values.shape[2]):
                car_points[i] = [
                    *sph_to_car(
                        sphere.r_from_index(r_idx), sphere.theta_from_index(theta_idx), sphere.phi_from_index(phi_idx)
                    ),
                    sphere.values[r_idx, theta_idx, phi_idx],
                ]
                i += 1
    return car_points


vertices, faces, _, _ = marching_cubes(sphere.values[:, :7, :], allow_degenerate=False, level=0.999)
vertices = sph_to_car(
    sphere.r_from_index(vertices[:, 0]), sphere.theta_from_index(vertices[:, 1]), sphere.phi_from_index(vertices[:, 2])
)
vertices, faces = remove_duplicate_vertices(vertices, faces)
mlab.triangular_mesh(
    vertices[:, 0],
    vertices[:, 1],
    vertices[:, 2],
    faces,
)

vertices2, faces2, _, _ = marching_cubes(
    sphere2.values[:, sphere2.theta_resolution // 2 :, :], allow_degenerate=False, level=0.999
)
vertices2 = sph_to_car(
    sphere2.r_from_index(vertices2[:, 0]),
    sphere2.theta_from_index(vertices2[:, 1] + sphere2.theta_resolution // 2),
    sphere2.phi_from_index(vertices2[:, 2]),
)
vertices2, faces2 = remove_duplicate_vertices(vertices2, faces2)
mlab.triangular_mesh(
    vertices2[:, 0] + 3,
    vertices2[:, 1],
    vertices2[:, 2],
    faces2,
)

# TODO: Close holes when merging
merged_vertices = np.vstack([vertices, vertices2])
merged_faces = np.vstack([faces, faces2 + vertices.shape[0]])
merged_vertices, merged_faces = remove_duplicate_vertices(merged_vertices, merged_faces)
mlab.triangular_mesh(
    merged_vertices[:, 0] + 6,
    merged_vertices[:, 1],
    merged_vertices[:, 2],
    merged_faces,
)

mlab.show()
