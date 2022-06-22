from functions_3D import CartesianSphere
from mayavi import mlab
from skimage.measure import marching_cubes
import numpy as np
from scipy.ndimage import gaussian_filter

res = 10
k = 10
low_res = CartesianSphere(res, pad=2)
high_res = CartesianSphere(res * k, pad=2)

low_res_scaled = low_res.values[1:-1, 1:-1, 1:-1].repeat(k, axis=0).repeat(k, axis=1).repeat(k, axis=2)
low_res_scaled = np.pad(low_res_scaled, 2, constant_values=0)
low_res_scaled = gaussian_filter(low_res_scaled - 0.5, 5, truncate=8) + 0.5

vert, face, _, _ = marching_cubes(low_res_scaled, spacing=[high_res.spacing] * 3, level=0.5)
mlab.triangular_mesh(
    vert[:, 0] + high_res.grid[0][0],
    vert[:, 1] + high_res.grid[1][0],
    vert[:, 2] + high_res.grid[2][0],
    face,
)
# mlab.points3d(*high_res.meshgrid, high_res.values, mode="cube", scale_factor=high_res.spacing)


mlab.show()
