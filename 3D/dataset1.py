from copy import deepcopy
from os.path import exists
from sys import getsizeof

import numpy as np
from matplotlib import pyplot as plt
from mayavi import mlab
from scipy.ndimage import gaussian_filter1d
from skimage.measure import marching_cubes

from functions_3D import linear, sph_to_car


class Grid:
    """
    This is a class which provides utilities for importing, visualizing and stitching grids from TODO: add grid origin
    The grid uses spherical coordinates.
    """

    def __init__(self, r, theta, phi, grid):
        """
        Initiate an instance.

        :param r: R coordinates corresponding to the first axis of 'grid'.
        :param theta: Theta coordinates corresponding to the second axis of 'grid'.
        :param phi: Phi coordinates corresponding to the third axis of 'grid'.
        :param grid: The values of the grid at locations defined by 'r', 'theta' and 'phi'.
        """
        # Coordinates have to be increasing
        assert np.all(r[1:] >= r[:-1])
        assert np.all(theta[1:] >= theta[:-1])
        assert np.all(phi[1:] >= phi[:-1])
        self.r = r
        self.theta = theta
        self.phi = phi
        self.grid = grid

    @staticmethod
    def from_file(grid_no, folder="../data/Dataset1"):
        """
        Initiate an instance by loading the data from a file. The files are stored in the specified folder with
        filenames grid_binary_#.npy for the grid and grid_R_#.npy, grid_t_#.npy, grid_p_#.npy for the R, t and p
        coordinates respectively

        :param grid_no: Number of the grid.
        :param folder: The folder where the grid and coordinate arrays are stored.
        :return: Grid instance.
        """
        if exists(f"{folder}/grid_{grid_no}.npz"):
            r, t, p, grid = np.load(f"{folder}/grid_{grid_no}.npz").values()
        else:
            # Round the values to 10 decimal places to avoid precision issues in the 'stitch' method.
            r = np.load(f"{folder}/grid_R_{grid_no}.npy").astype(float).round(10)
            t = np.load(f"{folder}/grid_t_{grid_no}.npy").astype(float).round(10)
            p = np.load(f"{folder}/grid_p_{grid_no}.npy").astype(float).round(10)
            grid = np.load(f"{folder}/grid_binary_{grid_no}.npy").astype(bool)
            np.savez_compressed(f"{folder}/grid_{grid_no}.npz", r=r, t=t, p=p, grid=grid)
        theta = (90 - t) / 180 * np.pi
        phi = p / 180 * np.pi
        # Fix theta values so that they are increasing
        theta, grid = theta[::-1], grid[:, ::-1, :]
        # Remove overlap in phi coordinates
        phi, grid = phi[3:-3], grid[:, :, 3:-3]

        return Grid(r, theta, phi, grid)

    def coords_from_indices_1d(self, ids, axis: str):
        """
        Converts indices of an array to the spherical coordinates in the specified axis direction.
        If the index is a whole number, it's converted by directly looking up the corresponding value in the axis'
        coordinate array.
        If the index is a decimal number, it's coordinates are interpolated by the nearest integer indices' coordinates
        linearly.

        :param ids: (n) array of indices
        :param axis: Either 'r', 'theta' or 'phi', which specifies the axis direction the indices are from.
        :return: (n) array of coordinates generated from indices.
        """
        r_vals = np.empty_like(ids, dtype=self.__getattribute__(axis).dtype)
        whole_part = np.floor(ids).astype(int)
        decimal_part = ids - whole_part
        integer_indices = decimal_part == 0

        # Get values for integer indices directly
        r_vals[integer_indices] = self.__getattribute__(axis)[whole_part[integer_indices]]

        # Get values for non-integer indices by linear interpolation
        r_vals[~integer_indices] = linear(
            x0=whole_part[~integer_indices],
            y0=self.__getattribute__(axis)[whole_part[~integer_indices]],
            x1=whole_part[~integer_indices] + 1,
            y1=self.__getattribute__(axis)[whole_part[~integer_indices] + 1],
            x=ids[~integer_indices],
        )
        return r_vals

    def coords_from_indices_nd(self, r_ids, theta_ids, phi_ids):
        """
        Converts indices of an array to the spherical coordinates.
        If the index is a whole number, it's converted by directly looking up the corresponding value in the coordinate
        arrays.
        If the index is a decimal number, it's coordinates are interpolated by the nearest integer indices' coordinates
        linearly.

        :param r_ids: (n) array of r indices
        :param theta_ids: (n) array of theta indices
        :param phi_ids: (n) array of phi indices
        :return: (n, 3) array of coordinates generated from indices.
        """
        return np.array(
            [
                self.coords_from_indices_1d(r_ids, "r"),
                self.coords_from_indices_1d(theta_ids, "theta"),
                self.coords_from_indices_1d(phi_ids, "phi"),
            ]
        ).T

    def stitch(self, other: "Grid"):
        """
        The stitching assumes that the spatial resolution is the same in both grids in the region where the two grids
        overlap. Moreover, the grid points of the overlapping region have to have the same coordinates in both grids.
        In the overlapping region, values from 'other' take precedence.

        :param other: The grid to stitch with.
        """
        # Merge the coordinate arrays of the two grids
        new_r = np.union1d(self.r, other.r)
        new_theta = np.union1d(self.theta, other.theta)
        new_phi = np.union1d(self.phi, other.phi)
        new_grid = np.zeros((new_r.size, new_theta.size, new_phi.size), dtype=self.grid.dtype)

        # Set values of the original array
        r0 = np.nonzero(self.r[0] == new_r)[0][0]
        theta0 = np.nonzero(self.theta[0] == new_theta)[0][0]
        phi0 = np.nonzero(self.phi[0] == new_phi)[0][0]
        new_grid[r0 : r0 + self.r.size, theta0 : theta0 + self.theta.size, phi0 : phi0 + self.phi.size] = self.grid

        # Set values of the 'other' array
        r0 = np.nonzero(other.r[0] == new_r)[0][0]
        theta0 = np.nonzero(other.theta[0] == new_theta)[0][0]
        phi0 = np.nonzero(other.phi[0] == new_phi)[0][0]
        new_grid[r0 : r0 + other.r.size, theta0 : theta0 + other.theta.size, phi0 : phi0 + other.phi.size] = other.grid

        # Set new arrays
        self.r = new_r
        self.theta = new_theta
        self.phi = new_phi
        self.grid = new_grid


# grid = Grid.from_file(0, folder="../data/Dataset1")
# for i in range(1, 2):
#     grid.stitch(Grid.from_file(i, folder="../data/Dataset1"))

grid = Grid.from_file(10, folder="../data/Dataset2")
for i in range(11, 25):
    grid.stitch(Grid.from_file(i, folder="../data/Dataset2"))

print(getsizeof(grid.grid) / 1e6, "MB")
print(grid.grid.shape)

"""
Using unsigned 8-bit integer allows to use 1/8 of the memory required to store the grid after applying the gaussian
filter in compared to 64-bit float, while having the same results when creating the triangular mesh.
"""

grid.grid = grid.grid.astype(np.uint8) * 255
# Apply gaussian filter axis by axis in order to make sure that on the phi axis, the values at -pi and +pi are equal
original = deepcopy(grid)
truncates = [10, 2, 5]
sigmas = [3, 0.75, 2]
"""
Uncomment to show the kernel function in each direction
"""
# labels = ["r", "theta", "phi"]
# for i in range(3):
#     plt.subplot(131 + i)
#     x = np.arange(-truncates[i], truncates[i] + 1)
#     y = np.exp(-0.5 * np.square(x) / np.square(sigmas[i]))
#     y = y / np.sum(y)
#     plt.plot(x, y)
#     plt.xticks(x)
#     plt.ylim(0, np.max(y))
#     plt.title(labels[i])
# plt.show()

grid.grid = gaussian_filter1d(
    grid.grid, axis=0, sigma=sigmas[0], truncate=truncates[0] / sigmas[0], mode="nearest", output=grid.grid
)
grid.grid = gaussian_filter1d(
    grid.grid, axis=1, sigma=sigmas[1], truncate=truncates[1] / sigmas[1], mode="nearest", output=grid.grid
)
grid.grid[..., :-1] = gaussian_filter1d(
    grid.grid[..., :-1],
    axis=2,
    sigma=sigmas[2],
    truncate=truncates[2] / sigmas[2],
    mode="wrap",
    output=grid.grid[..., :-1],
)
grid.grid[..., -1] = grid.grid[..., 0]

# Plot in spherical coordinate space
vertices, faces, _, _ = marching_cubes(grid.grid, allow_degenerate=False, level=128)
vertices = grid.coords_from_indices_nd(*vertices.T)
# mlab.triangular_mesh(
#     vertices[:, 0],
#     vertices[:, 1],
#     vertices[:, 2],
#     faces,
#     color=(0.5, 0.5, 0.5),
# )
#
# mlab.surf(*np.ogrid[1:1:2j, 0 : np.pi : 2j], [[-np.pi, -np.pi], [np.pi, np.pi]], color=(0.7, 0.7, 0))
# mlab.orientation_axes(xlabel="r", ylabel="theta", zlabel="phi")
# mlab.show()

# Plot in cartesian coordinates
vertices = sph_to_car(*vertices.T)
mlab.triangular_mesh(
    vertices[:, 0],
    vertices[:, 1],
    vertices[:, 2],
    faces,
    color=(0.5, 0.5, 0.5),
)

"""
Uncomment for overlay of original contour
"""
# vertices, faces, _, _ = marching_cubes(original.grid, allow_degenerate=False, level=0.5)
# vertices = original.coords_from_indices_nd(*vertices.T)
# vertices = sph_to_car(*vertices.T)
# mlab.triangular_mesh(
#     vertices[:, 0],
#     vertices[:, 1],
#     vertices[:, 2],
#     faces,
#     color=(0.0, 0.3, 0.3),
#     opacity=0.1,
# )


mlab.points3d(0, 0, 0, scale_factor=2 * 1, resolution=64, color=(0.7, 0.7, 0))
mlab.orientation_axes(xlabel="x", ylabel="y", zlabel="z")

mlab.show()
