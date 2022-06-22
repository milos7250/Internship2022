from mayavi import mlab
from skimage.measure import marching_cubes
import numpy as np

from functions_3D import remove_duplicate_vertices, linear, SphericalSphere

k = 2
small = 7
r = theta = small * k - 1
phi = small * k

sphere = SphericalSphere([r, theta, phi])
sphere_small = SphericalSphere(small)


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


# Show the upper part of high resolution sphere
vertices, faces, _, _ = marching_cubes(
    sphere.values[:, : sphere.theta_resolution // 2 + 1, :], allow_degenerate=False, level=0.999
)
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

# Show the lower part of low resolution sphere
vertices2, faces2, _, _ = marching_cubes(
    sphere_small.values[:, sphere_small.theta_resolution // 2 :, :], allow_degenerate=False, level=0.999
)
vertices2 = sph_to_car(
    sphere_small.r_from_index(vertices2[:, 0]),
    sphere_small.theta_from_index(vertices2[:, 1] + sphere_small.theta_resolution // 2),
    sphere_small.phi_from_index(vertices2[:, 2]),
)
vertices2, faces2 = remove_duplicate_vertices(vertices2, faces2)
mlab.triangular_mesh(
    vertices2[:, 0] + 3,
    vertices2[:, 1],
    vertices2[:, 2],
    faces2,
)

# Merge the two half-spheres together
merged_vertices = np.vstack([vertices, vertices2])
merged_faces = np.vstack([faces, faces2 + vertices.shape[0]])
merged_vertices, merged_faces = remove_duplicate_vertices(merged_vertices, merged_faces)
mlab.triangular_mesh(
    merged_vertices[:, 0] + 6,
    merged_vertices[:, 1],
    merged_vertices[:, 2],
    merged_faces,
)

# sphere_middle = SphericalSphere(sphere.resolution)
# sphere_middle.values[...] = 0
# sphere_middle.values[:, sphere.theta_resolution // 2 + 1, :] = sphere.values[:, sphere.theta_resolution // 2 + 1, :]
# sphere_middle.values[::k, sphere.theta_resolution // 2 + 4, ::k] = sphere_small.values[
#     :-1, sphere_small.theta_resolution // 2, :
# ]
#
# vertices3, faces3, _, _ = marching_cubes(sphere_middle.values, allow_degenerate=False, level=0.999)
# vertices3 = sph_to_car(
#     sphere_middle.r_from_index(vertices3[:, 0]),
#     sphere_middle.theta_from_index(vertices3[:, 1]),
#     sphere_middle.phi_from_index(vertices3[:, 2]),
# )
# vertices3, faces3 = remove_duplicate_vertices(vertices3, faces3)
# mlab.triangular_mesh(
#     vertices3[:, 0] + 9,
#     vertices3[:, 1],
#     vertices3[:, 2],
#     faces3,
# )

mlab.show()
exit()

# Nearest neighbor interpolation of sphere in spherical coordinates produces a smooth sphere
sphere = SphericalSphere(small * k, pad=1)
sphere_small = SphericalSphere(small, pad=0)
sphere_scaled = SphericalSphere(small * k, pad=1)
values = sphere_small.values.repeat(k, axis=0).repeat(k, axis=1).repeat(k, axis=2)[:, :, : -k + 1]
sphere_scaled.values = np.pad(values, [[0, 1], [0, 0], [0, 0]])

merged_sphere = SphericalSphere(small * k, pad=1)
merged_sphere.values = np.concatenate(
    [
        sphere_scaled.values[:, sphere_scaled.theta_resolution // 2 :, :],
        sphere.values[:, : sphere.theta_resolution // 2 + 1, :],
    ],
    axis=1,
)

vertices3, faces3, _, _ = marching_cubes(merged_sphere.values, allow_degenerate=False, level=0.999)
vertices3 = sph_to_car(
    merged_sphere.r_from_index(vertices3[:, 0]),
    merged_sphere.theta_from_index(vertices3[:, 1]),
    merged_sphere.phi_from_index(vertices3[:, 2]),
)
vertices3, faces3 = remove_duplicate_vertices(vertices3, faces3)
mlab.triangular_mesh(
    vertices3[:, 0] + 9,
    vertices3[:, 1],
    vertices3[:, 2],
    faces3,
)
mlab.show()
