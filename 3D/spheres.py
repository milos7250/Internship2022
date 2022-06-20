from mayavi import mlab
from skimage.measure import marching_cubes

from functions_3D import CartesianSphere

for i in range(3, 6):
    sphere = CartesianSphere(i, centre=(3 * i, 0, 0))
    vert, face, _, _ = marching_cubes(sphere.values, spacing=[sphere.spacing] * 3)
    mlab.triangular_mesh(
        vert[:, 0] + sphere.grid[0][0],
        vert[:, 1] + sphere.grid[1][0],
        vert[:, 2] + sphere.grid[2][0] + 3,
        face,
    )
    mlab.points3d(*sphere.meshgrid, sphere.values, mode="cube", scale_factor=sphere.spacing)

mlab.show()

for i in range(3, 9):
    sphere = CartesianSphere(i, centre=(3 * i, 0, 0))
    k = 3
    # x = np.linspace(sphere.grid[0][0], sphere.grid[0][-1], (sphere.resolution + 2) * k)
    # y = np.linspace(sphere.grid[1][0], sphere.grid[1][-1], (sphere.resolution + 2) * k)
    # z = np.linspace(sphere.grid[2][0], sphere.grid[2][-1], (sphere.resolution + 2) * k)
    values = sphere.values.repeat(k, axis=0).repeat(k, axis=1).repeat(k, axis=2)
    vert, face, _, _ = marching_cubes(values, spacing=[2 / sphere.resolution / k] * 3)
    mlab.triangular_mesh(
        vert[:, 0] + sphere.grid[0][0],
        vert[:, 1] + sphere.grid[1][0],
        vert[:, 2] + sphere.grid[2][0] + 3,
        face,
    )
    mlab.points3d(*sphere.meshgrid, sphere.values, mode="cube", scale_factor=sphere.spacing)

mlab.show()
