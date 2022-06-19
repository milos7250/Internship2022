from spheres import Sphere
from mayavi import mlab
from skimage.measure import marching_cubes
import numpy as np

res = 10
k = 10
low_res = Sphere(res)
high_res = Sphere(res * k)

# for idx, radius in enumerate([5, 50]):
#     sphere = Sphere(radius, centre=(idx * 3, 0, 0))
#     vert, face, _, _ = marching_cubes(sphere.values, spacing=[sphere.spacing] * 3)
#     mlab.triangular_mesh(
#         vert[:, 0] + sphere.grid[0][0],
#         vert[:, 1] + sphere.grid[1][0],
#         vert[:, 2] + sphere.grid[2][0] + 3,
#         face,
#     )
#     mlab.points3d(*sphere.meshgrid, sphere.values, mode="cube", scale_factor=sphere.spacing)

low_res_scaled = low_res.values[1:-1, 1:-1, 1:-1].repeat(k, axis=0).repeat(k, axis=1).repeat(k, axis=2)
low_res_scaled = np.pad(low_res_scaled, 1, constant_values=0)
low_res_cut = low_res_scaled[:, :, high_res.grid[2] < 0]
high_res_cut = high_res.values[:, :, high_res.grid[2] > 0]
merged = np.stack([low_res_cut, high_res_cut], axis=2).reshape(high_res.values.shape)

vert, face, _, _ = marching_cubes(merged, spacing=[high_res.spacing] * 3)
mlab.triangular_mesh(
    vert[:, 0] + high_res.grid[0][0],
    vert[:, 1] + high_res.grid[1][0],
    vert[:, 2] + high_res.grid[2][0],
    face,
)
# mlab.points3d(*high_res.meshgrid, high_res.values, mode="cube", scale_factor=high_res.spacing)


mlab.show()
