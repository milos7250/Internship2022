import warnings

import numpy as np
from scipy.interpolate import splev, splprep


def are_collinear(points: np.ndarray, collinearity_tol: float = 1e-5):
    """
    Tests if n points are collinear. Generalized to D dimensions.

    :param points: (n, D) ndarray - points to test.
    :param collinearity_tol: Needs to be adjusted to fit size of data. Decrease if false colinear points are found,
    increase if collinear points are not distinguished.
    :return: True or False.
    """
    points = np.array(points)
    points = points - points[0]
    rank = np.linalg.matrix_rank(points, tol=collinearity_tol)
    return rank == 1


def check_half_plane(collinear_points: np.ndarray, points: np.ndarray):
    """
    Tests if points lie in the same half-plane.

    :param collinear_points: (2, 2) ndarray - 2 points which define a line.
    :param points: (2, 2) ndarray - 2 points of which location in reference to the line is to be determined.
    :return: True if points line in the same half-plane, False otherwise.
    """
    x1 = collinear_points[0, 0]
    x2 = collinear_points[1, 0]
    y1 = collinear_points[0, 1]
    y2 = collinear_points[1, 1]
    line_eq = lambda point: (x2 - x1) * (point[1] - y1) - (y2 - y1) * (point[0] - x1)

    return np.sign(line_eq(points[0])) == np.sign(line_eq(points[1]))


def isolate_from_collinear(path: np.ndarray, closed: bool = False, collinearity_tol: float = None):
    """
    Isolates points from step segments. In 2D, midpoints are selected if a step is an inflection point, two points along
    the step are selected if the step is local minimum/maximum. In more dimensions, only midpoints are selected.

    :param path: (n, 2) ndarray - the path to isolate points from.
    :param closed: Whether we want the isolated path to be closed
    :param collinearity_tol: Tolerance for 'functions_1D.are_collinear'. Needs to be adjusted to fit size of data.
    :return: (n, 2) ndarray - the path with isolated points.
    """
    points = np.ndarray((0, 2))
    firstpoint = 0
    lastpoint = 1
    i = 2

    def add_point(points: np.ndarray, point: np.ndarray):
        """
        Adds a point to points collection if last point is not identical.

        :param points: (n, D) ndarray of points
        :param point: Point to add
        :return: (n + 1, D) ndarray of the new collection of points
        """
        if points.size != 0 and np.all(point == points[-1]):
            pass
        else:
            points = np.append(points, [point], axis=0)
        return points

    while i < path.shape[0]:
        if are_collinear(path[firstpoint : i + 1], collinearity_tol):  # TODO: Estimate tolerance automatically
            lastpoint = i
            i += 1
            continue

        # Only consider adding two points per segment if the line is in 2D
        if points.shape[1] == 2 and check_half_plane(
            path[[firstpoint, lastpoint]], path[[firstpoint - 1, min(lastpoint + 1, path.shape[0] - 1)]]
        ):
            points = add_point(points, 2 / 3 * path[firstpoint] + 1 / 3 * path[lastpoint])
            points = add_point(points, 1 / 3 * path[firstpoint] + 2 / 3 * path[lastpoint])
        else:
            points = add_point(points, (path[firstpoint] + path[lastpoint]) / 2)
        firstpoint = lastpoint
        lastpoint = i
        i += 1

    # Handle the addition of last point
    if closed and check_half_plane(
        path[[firstpoint, lastpoint]], path[[firstpoint - 1, (lastpoint + 1) % (path.shape[0] - 1)]]
    ):

        points = add_point(points, 2 / 3 * path[firstpoint] + 1 / 3 * path[lastpoint])
        points = add_point(points, 1 / 3 * path[firstpoint] + 2 / 3 * path[lastpoint])
    else:
        points = add_point(points, (path[firstpoint] + path[lastpoint]) / 2)
    if closed:
        points = add_point(points, points[0])
    return points


def smooth_contour(contour: np.ndarray, smoothness: int = 20, collinearity_tol: float = None):
    """
    Smoothens a 2D contour by isolating points from linear segments and interpolating these points by cubic splines.

    :param contour: (n, 2) ndarray - points of the contour to smoothen.
    :param smoothness: The number of segments by which the spline should be approximated in between two consecutive
    points.
    :param collinearity_tol: Tolerance for 'functions_1D.are_collinear'. Needs to be adjusted to fit size of data.
    :return: (n, 2) ndarray - the smoothened contour.
    """
    closed = np.all(contour[0] == contour[-1])
    smoothness += 1  # Need to add 1 to change from no of segments to no of points

    isolated_contour = isolate_from_collinear(contour, closed, collinearity_tol)
    try:
        if isolated_contour.shape[0] >= 4:
            smoothness = isolated_contour.shape[0] * smoothness
            model, u = splprep(isolated_contour.T, s=0, k=3, per=1 if closed else 0)
        else:
            smoothness = contour.shape[0] * smoothness
            model, u = splprep(contour.T, s=0, k=min(3, contour.shape[0] - 1), per=1 if closed else 0)
        return np.array(splev(np.linspace(0, 1, smoothness), model)).T
    except ValueError as e:
        warnings.warn("An error occurred when smoothing contour. Points might be collinear/coincident.", RuntimeWarning)
        return isolated_contour
