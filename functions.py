import sys

import numpy as np
from scipy.interpolate import splev, splprep


def are_collinear(points, tol=None):
    """
    Tests if points are collinear. Generalized to n dimensions.
    """
    points = np.array(points)
    points -= points.mean(axis=0)[np.newaxis, :]
    rank = np.linalg.matrix_rank(points, tol=tol)
    return rank == 1


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


def isolate_collinear(line, closed=False, collinearity_tol=1e-5):
    """
    Isolates points from step segments.
    """
    points = np.ndarray((0, 2))
    firstpoint = 0
    lastpoint = 1
    i = 2

    def add_point(points, point):
        if points.size != 0 and np.all(point == points[-1]):
            pass
        else:
            points = np.append(points, [point], axis=0)
        return points

    while i < line.shape[0]:
        if are_collinear(line[firstpoint : i + 1], tol=collinearity_tol):  # Set tolerance to match point distances
            lastpoint = i
            i += 1
            continue

        if check_half_plane(
            line[[firstpoint, lastpoint]], line[[firstpoint - 1, min(lastpoint + 1, line.shape[0] - 1)]]
        ):
            points = add_point(points, 2 / 3 * line[firstpoint] + 1 / 3 * line[lastpoint])
            points = add_point(points, 1 / 3 * line[firstpoint] + 2 / 3 * line[lastpoint])
        else:
            points = add_point(points, (line[firstpoint] + line[lastpoint]) / 2)
        firstpoint = lastpoint
        lastpoint = i
        i += 1

    if closed and check_half_plane(
        line[[firstpoint, lastpoint]], line[[firstpoint - 1, (lastpoint + 1) % (line.shape[0] - 1)]]
    ):

        points = add_point(points, 2 / 3 * line[firstpoint] + 1 / 3 * line[lastpoint])
        points = add_point(points, 1 / 3 * line[firstpoint] + 2 / 3 * line[lastpoint])
    else:
        points = add_point(points, (line[firstpoint] + line[lastpoint]) / 2)
    if closed:
        points = add_point(points, points[0])
    return points


def smooth_contour(contour, smoothness=20, closed=None, collinearity_tol=1e-5):
    if closed is None:
        closed = np.all(contour[0] == contour[-1])

    isolated_contour = isolate_collinear(contour, closed, collinearity_tol)
    try:
        if isolated_contour.shape[0] >= 4:
            smoothness = isolated_contour.shape[0] * smoothness
            model, u = splprep(isolated_contour.T, s=0, k=3, per=1 if closed else 0)
        else:
            smoothness = contour.shape[0] * smoothness
            model, u = splprep(contour.T, s=0, k=min(3, contour.shape[0] - 1), per=1 if closed else 0)
        return np.array(splev(np.linspace(0, 1, smoothness), model)).T
    except ValueError as e:
        print("Failed to smoothen contour: " + e, file=sys.stderr)
        return isolated_contour
