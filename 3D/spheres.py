from mayavi import mlab
from skimage.measure import marching_cubes
import numpy as np
from raster_geometry import sphere as sphere_gen


class Sphere:
    def __init__(self, resolution: int, radius=1, centre=(0, 0, 0)):
        self.resolution = resolution
        self.radius = radius
        self.centre = centre
        self.values = sphere_gen(resolution + 2, resolution / 2).astype(int)
        if resolution == 1:
            self.spacing = 1
            points = np.linspace(-radius, radius, 3)
        else:
            self.spacing = 2 * radius / (resolution - 1)
            points = np.linspace(
                -radius - self.spacing,
                +radius + self.spacing,
                resolution + 2,
            )
        self.grid = (points.copy() + self.centre[0], points.copy() + self.centre[1], points.copy() + self.centre[2])
        self.meshgrid = np.meshgrid(*self.grid)

    def __str__(self):
        return self.__dict__.__str__()


if __name__ == "__main__":
    for i in range(3, 6):
        sphere = Sphere(i, centre=(3 * i, 0, 0))
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
        sphere = Sphere(i, centre=(3 * i, 0, 0))
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
