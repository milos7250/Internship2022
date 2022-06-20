from functions_3D import CartesianSphere
from mayavi import mlab
from skimage.measure import marching_cubes
import numpy as np
from scipy.ndimage import gaussian_filter, rank_filter
from scipy.interpolate import RBFInterpolator
import tricubic

res = 10
k = 2
pad = 2
low_res = CartesianSphere(res, pad=pad)
high_res = CartesianSphere(res * k, pad=pad)

vert, face, _, _ = marching_cubes(low_res.values, spacing=[low_res.spacing] * 3)
mlab.triangular_mesh(
    vert[:, 0] + low_res.grid[0][0] - 3,
    vert[:, 1] + low_res.grid[1][0],
    vert[:, 2] + low_res.grid[2][0],
    face,
)

low_res_scaled = low_res.values[pad:-pad, pad:-pad, pad:-pad].repeat(k, axis=0).repeat(k, axis=1).repeat(k, axis=2)
low_res_scaled = np.pad(low_res_scaled, pad, constant_values=0)


def isolate_vertices(x, y, z, values, block_size=2):
    vertices = []
    removed_mask = np.zeros_like(values)
    centering = (x[block_size - 1] - x[0]) / block_size
    for z_idx, z_val in enumerate(z[: -block_size + 1]):
        for y_idx, y_val in enumerate(y[: -block_size + 1]):
            for x_idx, x_val in enumerate(x[: -block_size + 1]):
                blocksum = np.sum(values[z_idx : z_idx + 2, y_idx : y_idx + 2, x_idx : x_idx + 2])
                if blocksum != 0 and blocksum != block_size**3:
                    removed_mask[z_idx : z_idx + 2, y_idx : y_idx + 2, x_idx : x_idx + 2] = 1
                if blocksum == 1 or blocksum == block_size**3 - 1:
                    vertices.append(
                        [
                            x_val + centering,
                            y_val + centering,
                            z_val + centering,
                            blocksum / block_size**3,
                        ]
                    )
    vertices = np.vstack(vertices)
    return removed_mask, vertices


removed_mask, vertices = isolate_vertices(*high_res.grid, low_res_scaled)
mlab.points3d(*vertices[:, 0:3].T, vertices[:, 3] + 1, scale_factor=high_res.spacing, colormap="coolwarm")
mlab.points3d(
    high_res.meshgrid[0],
    high_res.meshgrid[1],
    high_res.meshgrid[2] + 3,
    removed_mask,
    scale_factor=high_res.spacing / 2,
    colormap="terrain",
)


interpolator = RBFInterpolator(vertices[:, 0:3], vertices[:, 3], kernel="thin_plate_spline", smoothing=2)
interpolated = low_res_scaled.copy()


for z_idx, z_val in enumerate(high_res.grid[2]):
    for y_idx, y_val in enumerate(high_res.grid[1]):
        for x_idx, x_val in enumerate(high_res.grid[0]):
            if removed_mask[z_idx, y_idx, x_idx] == 1:
                interpolated[z_idx, y_idx, x_idx] = interpolator(np.array([[x_val, y_val, z_val]]))

# print(*interpolated)

vert, face, _, _ = marching_cubes(interpolated, spacing=[high_res.spacing] * 3)
mlab.triangular_mesh(
    vert[:, 0] + low_res.grid[0][0] + 3,
    vert[:, 1] + low_res.grid[1][0],
    vert[:, 2] + low_res.grid[2][0],
    face,
)


# spline = tricubic.tricubic(interpolated.tolist(), list(interpolated.shape))
# interpolated = np.ndarray([i * k for i in interpolated.shape])
# for z in range(interpolated.shape[2]):
#     for y in range(interpolated.shape[1]):
#         for x in range(interpolated.shape[0]):
#             interpolated[z, y, x] = spline.ip([x / k, y / k, z / k])
#
# vert, face, _, _ = marching_cubes(interpolated, spacing=[low_res.spacing / k] * 3)
# mlab.triangular_mesh(
#     vert[:, 0] + low_res.grid[0][0] + 6,
#     vert[:, 1] + low_res.grid[1][0],
#     vert[:, 2] + low_res.grid[2][0],
#     face,
# )


mlab.show()
