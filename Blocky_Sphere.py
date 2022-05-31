R_actual = 1.0
X_offset = 3.0
block_dimensions = [11, 11, 11]
scaling_factor = 4


import numpy as np
from mayavi import mlab
from skimage import measure

# Calculate low resolution isosurface on a grid of block_dimensions
lowres_X = np.linspace(-1.1 * R_actual, 1.1 * R_actual, num=block_dimensions[0])
lowres_Y = np.linspace(-1.1 * R_actual, 1.1 * R_actual, num=block_dimensions[1])
lowres_Z = np.linspace(-1.1 * R_actual, 1.1 * R_actual, num=block_dimensions[2])
lowres_field = np.zeros((len(lowres_X), len(lowres_Y), len(lowres_Z)))
for idx_x in range(len(lowres_X)):
    for idx_y in range(len(lowres_Y)):
        for idx_z in range(len(lowres_Z)):
            if lowres_X[idx_x] ** 2 + lowres_Y[idx_y] ** 2 + lowres_Z[idx_z] ** 2 <= R_actual:
                lowres_field[idx_x, idx_y, idx_z] = 1.0

lowres_verts, lowres_faces, lowres_normals, lowres_values = measure.marching_cubes(
    lowres_field, 0.5, spacing=(lowres_X[1] - lowres_X[0], lowres_Y[1] - lowres_Y[0], lowres_Z[1] - lowres_Z[0])
)
# Shift vertices to where grid actually starts
lowres_verts[:, 0] = lowres_verts[:, 0] + lowres_X[0]
lowres_verts[:, 1] = lowres_verts[:, 1] + lowres_Y[0]
lowres_verts[:, 2] = lowres_verts[:, 2] + lowres_Z[0]


# Calculate high resolution, but very "blocky" isosurface
blocky_X = np.linspace(-1.1 * R_actual, 1.1 * R_actual, num=block_dimensions[0] * scaling_factor)
blocky_Y = np.linspace(-1.1 * R_actual, 1.1 * R_actual, num=block_dimensions[1] * scaling_factor)
blocky_Z = np.linspace(-1.1 * R_actual, 1.1 * R_actual, num=block_dimensions[2] * scaling_factor)
blocky_field = np.zeros((len(blocky_X), len(blocky_Y), len(blocky_Z)))
for idx_x in range(len(lowres_X)):
    for idx_y in range(len(lowres_Y)):
        for idx_z in range(len(lowres_Z)):
            blocky_field[
                idx_x * scaling_factor : (idx_x + 1) * scaling_factor,
                idx_y * scaling_factor : (idx_y + 1) * scaling_factor,
                idx_z * scaling_factor : (idx_z + 1) * scaling_factor,
            ] = lowres_field[idx_x, idx_y, idx_z]
blocky_verts, blocky_faces, blocky_normals, blocky_values = measure.marching_cubes(
    blocky_field, 0.5, spacing=(blocky_X[1] - blocky_X[0], blocky_Y[1] - blocky_Y[0], blocky_Z[1] - blocky_Z[0])
)
# Shift vertices to where grid actually starts
blocky_verts[:, 0] = blocky_verts[:, 0] + blocky_X[0]
blocky_verts[:, 1] = blocky_verts[:, 1] + blocky_Y[0]
blocky_verts[:, 2] = blocky_verts[:, 2] + blocky_Z[0]


# Create actual sphere
theta = np.linspace(0.0, np.pi, num=250)
phi = np.linspace(-np.pi, np.pi, num=500)
phi, theta = np.meshgrid(phi, theta)
sphere_X = R_actual * np.sin(theta) * np.cos(phi)
sphere_Y = R_actual * np.sin(theta) * np.sin(phi)
sphere_Z = R_actual * np.cos(theta)
sphere_S = R_actual * np.ones(np.shape(theta))


# Display lowres isosurface
ocb_mesh = mlab.triangular_mesh(
    lowres_verts[:, 0],
    lowres_verts[:, 1],
    lowres_verts[:, 2],
    lowres_faces,
    scalars=lowres_values,
    colormap="PuOr",
    vmin=0.0,
    vmax=2.0,
)

# Display blocky isosurface
ocb_mesh = mlab.triangular_mesh(
    blocky_verts[:, 0] + X_offset,
    blocky_verts[:, 1],
    blocky_verts[:, 2],
    blocky_faces,
    scalars=blocky_values,
    colormap="PuOr",
    vmin=0.0,
    vmax=2.0,
)

# Display sphere
mlab.mesh(sphere_X + 2 * X_offset, sphere_Y, sphere_Z, scalars=sphere_S, colormap="PuOr", vmin=0.0, vmax=2.0)


# Required to show window, can also save as .png
mlab.show()
