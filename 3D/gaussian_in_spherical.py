import numpy as np
from mayavi import mlab
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes

from functions_3D import remove_duplicate_vertices, sph_to_car, indices_to_coords_nd

"""
This script investigates how marching cubes and gaussian filter work in spherical coordinates.
"""


res = 30
surface_kind = ["ellipsoid", "saddle"][0]  # Change number here to change surface type
surface = np.zeros([res] * 3)
r = np.linspace(0, res - 1, res)
theta = np.linspace(0, np.pi, res)
phi = np.linspace(0, 2 * np.pi, res)
for r_idx, r_val in enumerate(r):
    for theta_idx, theta_val in enumerate(theta):
        for phi_idx, phi_val in enumerate(phi):
            x, y, z = sph_to_car(r_val, theta_val, phi_val)
            if surface_kind == "ellipsoid":
                a, b, c = res - 1, res / 2, res / 3
                if (x / a) ** 2 + (y / b) ** 2 + (z / c) ** 2 < 1:
                    surface[r_idx, theta_idx, phi_idx] = 1
            elif surface_kind == "saddle":
                x /= res / 2
                y /= res / 2
                z /= res / 2
                if z - x**2 + y**2 < 0:
                    surface[r_idx, theta_idx, phi_idx] = 1


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
vertices = indices_to_coords_nd(vertices.T, [r, theta, phi])
vertices = sph_to_car(*vertices.T)
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
vertices = indices_to_coords_nd(vertices.T, [r, theta, phi])
vertices = sph_to_car(*vertices.T)
vertices, faces = remove_duplicate_vertices(vertices, faces)
mlab.triangular_mesh(
    vertices[:, 0] + res * 2.5,
    vertices[:, 1],
    vertices[:, 2] + res * 2.5,
    faces,
)
mlab.show()
