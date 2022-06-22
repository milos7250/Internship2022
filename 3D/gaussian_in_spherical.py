import numpy as np
from mayavi import mlab
from skimage.measure import marching_cubes
from scipy.ndimage import gaussian_filter

from functions_3D import remove_duplicate_vertices, linear


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


def coords_from_indices(r_idx, theta_idx, phi_idx, shape):
    return (
        linear(0, 0, shape[0] - 1, shape[0] - 1, r_idx),
        linear(0, 0, shape[1] - 1, np.pi, theta_idx),
        linear(0, 0, shape[2] - 1, 2 * np.pi, phi_idx),
    )


def sph_grid_to_car(grid):
    car_points = np.empty((grid.size, 4))
    i = 0
    for r_idx in range(grid.shape[0]):
        for theta_idx in range(grid.shape[1]):
            for phi_idx in range(grid.shape[2]):
                car_points[i] = [
                    *sph_to_car(*coords_from_indices(r_idx, theta_idx, phi_idx, grid.shape)),
                    grid[r_idx, theta_idx, phi_idx],
                ]
                i += 1
    return car_points


res = 30
surface_kind = ["ellipsoid", "saddle"][0]  # Change number here to change surface type
surface = np.zeros([res] * 3)
for r in range(res):
    for theta in range(res):
        for phi in range(res):
            x, y, z = sph_to_car(*coords_from_indices(r, theta, phi, surface.shape))
            if surface_kind == "ellipsoid":
                a, b, c = res - 1, res / 2, res / 3
                if (x / a) ** 2 + (y / b) ** 2 + (z / c) ** 2 < 1:
                    surface[r, theta, phi] = 1
            elif surface_kind == "saddle":
                x /= res / 2
                y /= res / 2
                z /= res / 2
                if z - x**2 + y**2 < 0:
                    surface[r, theta, phi] = 1
X, Y, Z = np.indices(surface.shape)


# Plot the surface in (r, theta, phi) space
vertices, faces, _, _ = marching_cubes(surface, allow_degenerate=False, level=0.5)
vertices, faces = remove_duplicate_vertices(vertices, faces)
mlab.triangular_mesh(
    vertices[:, 0],
    vertices[:, 1],
    vertices[:, 2],
    faces,
)
mlab.orientation_axes(xlabel="r", ylabel="theta", zlabel="phi")

# Plot the surface in cartesian coordinates
vertices = sph_to_car(*coords_from_indices(vertices[:, 0], vertices[:, 1], vertices[:, 2], surface.shape))
vertices, faces = remove_duplicate_vertices(vertices, faces)
mlab.triangular_mesh(
    vertices[:, 0],
    vertices[:, 1],
    vertices[:, 2] + res * 2.5,
    faces,
)

# Use gaussian kernel to smoothen the surface
gaussian_surface = gaussian_filter(surface - 0.5, 1) + 0.5


# Plot the surface in (r, theta, phi) space
vertices, faces, _, _ = marching_cubes(gaussian_surface, allow_degenerate=False, level=0.5)
vertices, faces = remove_duplicate_vertices(vertices, faces)
mlab.triangular_mesh(
    vertices[:, 0] + res * 2.5,
    vertices[:, 1],
    vertices[:, 2],
    faces,
)

# Plot the surface in cartesian coordinates
vertices, faces, _, _ = marching_cubes(gaussian_surface, allow_degenerate=False, level=0.5)
vertices = sph_to_car(*coords_from_indices(vertices[:, 0], vertices[:, 1], vertices[:, 2], gaussian_surface.shape))
vertices, faces = remove_duplicate_vertices(vertices, faces)
mlab.triangular_mesh(
    vertices[:, 0] + res * 2.5,
    vertices[:, 1],
    vertices[:, 2] + res * 2.5,
    faces,
)
mlab.show()
