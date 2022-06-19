from spheres import Sphere
from mayavi import mlab
from skimage.measure import marching_cubes
import numpy as np
from scipy.interpolate import griddata, RBFInterpolator
import tricubic

res = 10
k = 5
low_res = Sphere(res)

vert, face, _, _ = marching_cubes(low_res.values, spacing=[low_res.spacing] * 3)
mlab.triangular_mesh(
    vert[:, 0] + low_res.grid[0][0],
    vert[:, 1] + low_res.grid[1][0],
    vert[:, 2] + low_res.grid[2][0],
    face,
)

spline = tricubic.tricubic(low_res.values.tolist(), list(low_res.values.shape))
interpolated = np.ndarray([i * k for i in low_res.values.shape])
for z in range(interpolated.shape[2]):
    for y in range(interpolated.shape[1]):
        for x in range(interpolated.shape[0]):
            interpolated[z, y, x] = spline.ip([x / k, y / k, z / k])
vert, face, _, _ = marching_cubes(interpolated, spacing=[low_res.spacing / k] * 3)
mlab.triangular_mesh(
    vert[:, 0] + low_res.grid[0][0] + 3,
    vert[:, 1] + low_res.grid[1][0],
    vert[:, 2] + low_res.grid[2][0],
    face,
)

points = np.array(low_res.meshgrid).reshape(3, -1).T

X, Y, Z = np.mgrid[
    low_res.grid[0][0] : low_res.grid[0][-1] : complex(low_res.grid[0].size * k),
    low_res.grid[1][0] : low_res.grid[1][-1] : complex(low_res.grid[1].size * k),
    low_res.grid[2][0] : low_res.grid[2][-1] : complex(low_res.grid[2].size * k),
]
interpolate_points = np.array([X, Y, Z]).reshape(3, -1).T
interpolated = RBFInterpolator(points, low_res.values.flatten(), kernel="thin_plate_spline", smoothing=1e-1)(
    interpolate_points
).reshape(
    low_res.grid[2].size * k,
    low_res.grid[1].size * k,
    low_res.grid[0].size * k,
)

vert, face, _, _ = marching_cubes(interpolated, spacing=[low_res.spacing / k] * 3)
mlab.triangular_mesh(
    vert[:, 0] + low_res.grid[0][0] + 6,
    vert[:, 1] + low_res.grid[1][0],
    vert[:, 2] + low_res.grid[2][0],
    face,
)

mlab.show()
