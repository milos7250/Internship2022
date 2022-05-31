import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate

from bezier import evaluate_bezier

# USE FOR FUNCTIONS

# Load and plot the original function
line = np.loadtxt("data/Simple_Line1A.txt")
plt.figure(figsize=(11, 8))
plt.plot(line[:, 0], line[:, 1])


def check_half_plane(collinear_points, points):
    """
    Tests if points lie in the same half-plane
    """
    x1 = collinear_points[0, 0]
    x2 = collinear_points[1, 0]
    y1 = collinear_points[0, 1]
    y2 = collinear_points[1, 1]
    line_eq = lambda point: (x2 - x1) * (point[1] - y1) - (y2 - y1) * (point[0] - x1)

    return np.sign(line_eq(points[0])) == np.sign(line_eq(points[1]))


def isolate_points(line):
    """
    Isolate points from step segments. The 'kind' input argument changes which part of the step line is chosen as the
    reference point for interpolation.
    """
    points = np.ndarray((0, 2))
    firstpoint = 0
    lastpoint = 1
    i = 2

    while i < line.shape[0]:
        if line[i, 1] == line[firstpoint, 1]:
            lastpoint = i
            i += 1
            continue
        if (
            firstpoint != 0
            and lastpoint != line.shape[0]
            and check_half_plane(line[[firstpoint, lastpoint]], line[[firstpoint - 1, lastpoint + 1]])
        ):
            points = np.append(
                points,
                [
                    2 / 3 * line[firstpoint] + 1 / 3 * line[lastpoint],
                    1 / 3 * line[firstpoint] + 2 / 3 * line[lastpoint],
                ],
                axis=0,
            )
        else:
            points = np.append(points, [(line[firstpoint] + line[lastpoint]) / 2], axis=0)
        firstpoint = lastpoint
        lastpoint = i
        i += 1

    points = np.append(points, [(line[firstpoint] + line[lastpoint]) / 2], axis=0)
    return points


isolated_points = isolate_points(line)

# CUBIC SPLINE
model = interpolate.CubicSpline(isolated_points[:, 0], isolated_points[:, 1], bc_type="natural", extrapolate=False)
new_x = np.linspace(line[0, 0], line[-1, 0], 1000)
new_y = model(new_x)
plt.plot(new_x, new_y)

# BEZIER
path = evaluate_bezier(isolated_points, 50)
plt.plot(path[:, 0], path[:, 1])

plt.legend(["Original line", "Cubic spline", "Bezier spline"])

# Plot line points and isolated points
# plt.plot(line[:, 0], line[:, 1], "gx", markersize=10)
# plt.plot(isolated_points[:, 0], isolated_points[:, 1], 'ro')

plt.show()
