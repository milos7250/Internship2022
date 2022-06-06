import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from skimage.measure import find_contours
from functions_1D import smooth_contour, isolate_collinear
from scipy import interpolate


def plot_contours(
    x: np.ndarray,
    y: np.ndarray,
    datagrid: np.ndarray,
    interpolate: bool,
    ax: Axes,
    colors: ScalarMappable,
    levels: list = None,
    low_output_resolution: bool = False,
    plot_title: str = None,
):
    """
    Finds contours of data at specified levels using 'skimage.measure._find_contours.find_contours' and plots them on
    the specified axis.

    :param x: (m) ndarray - The x coordinates of the datagrid
    :param y: (n) ndarray - The y coordinates of the datagrid
    :param datagrid: (n, m) ndarray - the datagrid to plot contours from.
    :param interpolate: Whether to interpolate the found contours using 'functions_1D.smooth_contour'.
    :param ax: The pyplot axis to plot to.
    :param colors: ScalarMappable instance used to determine contour colors.
    :param levels: Values at which to plot contours. Only necessary when low_output_resolution=False.
    :param low_output_resolution: If set to True, the values of datagrid are expected to be discrete. This is the case
    when the output resolution of the function is low (the output is binned).
    :param plot_title: The title of the plot.
    :return: None
    """
    if low_output_resolution:
        levels = np.unique(datagrid)
        levels = np.append(levels, 2 * levels[-1] - levels[-2])
    elif levels is None:
        raise TypeError("Either low_output_resolution=True or levels must be specified.")

    # Find contours for each level
    for contour_datapoints in isolate_contour_datapoints(
        x, y, datagrid, levels=levels, low_output_resolution=low_output_resolution, return_separate_contours=True
    ):
        level = contour_datapoints[0, 2]
        if interpolate:
            contour_datapoints = smooth_contour(contour_datapoints[:, 0:2])
        ax.plot(contour_datapoints[:, 0], contour_datapoints[:, 1], color=colors.to_rgba(level))
    title = ""
    if plot_title:
        title += f"{plot_title}\n"
    if interpolate:
        title += "Interpolated "
    ax.set_title(f"{title}Contours")


# TODO This method is no longer used, it does not produce satisfiable results.
def isolate_datapoints(datagrid: np.ndarray, borders: bool, collinearity_tol: float = None):
    """
    Isolates points from data in a grid using methods for 1D interpolation of step-like functions.

    :param datagrid: (n, m) ndarray - the datagrid to isolate points from.
    :param borders: Whether to include border points in the output.
    :param collinearity_tol: Tolerance for 'functions_1D.are_collinear'. Needs to be adjusted to fit size of data.
    :return: (n, 3) ndarray, where each line is a point (x, y, f(x, y)) isolated from the grid.
    """
    isolated_datapoints = np.ndarray((0, 3))
    x = np.arange(datagrid.shape[0])
    y = np.arange(datagrid.shape[1])
    # Isolate points along x axis
    for yi, y_stripe in enumerate(datagrid):
        isolated = isolate_collinear(np.column_stack([x, y_stripe]), closed=False, collinearity_tol=collinearity_tol)
        if borders:
            isolated = np.append(isolated, [[x[0], y_stripe[0]], [x[-1], y_stripe[-1]]], axis=0)
        isolated_datapoints = np.append(
            isolated_datapoints, np.insert(isolated, 1, np.repeat(yi, isolated.shape[0]), axis=1), axis=0
        )
    # Isolate points along y axis
    for xi, x_stripe in enumerate(datagrid.T):
        isolated = isolate_collinear(np.column_stack([y, x_stripe]), closed=False, collinearity_tol=collinearity_tol)
        if borders:
            isolated = np.append(isolated, [[y[0], x_stripe[0]], [y[-1], x_stripe[-1]]], axis=0)
        isolated_datapoints = np.append(
            isolated_datapoints, np.insert(isolated, 0, np.repeat(xi, isolated.shape[0]), axis=1), axis=0
        )
    return isolated_datapoints


def isolate_contour_datapoints(
    x: np.ndarray,
    y: np.ndarray,
    datagrid: np.ndarray,
    levels: list = None,
    low_output_resolution: bool = False,
    return_separate_contours=False,
):
    """
    Finds contours of data at specified levels using 'skimage.measure._find_contours.find_contours'.

    :param x: (m) ndarray - The x coordinates of the datagrid
    :param y: (n) ndarray - The y coordinates of the datagrid
    :param datagrid: (n, m) ndarray - the datagrid to isolate contours from.
    :param levels: Values at which to plot contours. Only necessary when low_output_resolution=False.
    :param low_output_resolution: If set to True, the values of datagrid are expected to be discrete. This is the case
    when the output resolution of the function is low (the output is binned).
    :return: list of (k_i, 3) ndarrays - isolated contours
    """
    if low_output_resolution and levels is None:
        levels = np.unique(datagrid)
        levels = np.append(levels, 2 * levels[-1] - levels[-2])
    elif levels is None:
        raise TypeError("Either low_output_resolution=True or levels must be specified.")

    contours = list()
    # Find contours
    for idx, level in enumerate(levels):
        for contour in find_contours(datagrid.T, level):
            contours.append(
                np.concatenate(
                    [
                        x[contour[:, 0].astype(int)].T[:, np.newaxis],
                        y[contour[:, 1].astype(int)].T[:, np.newaxis],
                        np.array([levels[idx + 1 if low_output_resolution else idx]] * contour.shape[0]).T[
                            :, np.newaxis
                        ],
                    ],
                    axis=1,
                ),
            )
    if return_separate_contours:
        return contours
    else:
        return np.concatenate(contours, axis=0)


def __test_elementwise(original, interpolated, levels):
    min_index = np.where(original == np.array(levels))[0][-1]
    minimum = levels[min_index]
    maximum = levels[min_index + 1]
    res = minimum <= interpolated <= maximum
    return res


__test_vectorized = np.vectorize(__test_elementwise, excluded=[2])


def test(datagrid, interpolated_datagrid):
    levels = np.unique(datagrid)
    levels = np.append(levels, 2 * levels[-1] - levels[-2])
    return __test_vectorized(datagrid, interpolated_datagrid, levels)


def __clip_elementwise(original, interpolated, levels):
    min_index = np.where(original == np.array(levels))[0][-1]
    minimum = levels[min_index]
    maximum = levels[min_index + 1]
    if interpolated <= minimum:
        return minimum
    elif maximum <= interpolated:
        return maximum
    else:
        return interpolated


__clip_vectorized = np.vectorize(__clip_elementwise, excluded=[2])


def clip_to_data(datagrid, interpolated_datagrid):
    interpolated_datagrid[np.isnan(interpolated_datagrid)] = datagrid[np.isnan(interpolated_datagrid)]
    levels = np.unique(datagrid)
    levels = np.append(levels, 2 * levels[-1] - levels[-2])
    return __clip_vectorized(datagrid, interpolated_datagrid, levels)


def fix_contours(x: np.ndarray, y: np.ndarray, datagrid, interpolated_datagrid):
    """

    :param x: (m) ndarray - The x coordinates of the datagrid.
    :param y: (n) ndarray - The y coordinates of the datagrid.
    :param datagrid:(n, m) ndarray - The datagrid before interpolation.
    :param interpolated_datagrid: (n, m) ndarray - The datagrid after interpolation.
    :return: (n, m) ndarray - The interpolated datagrid combined with contours from the original datagrid.
    """
    contours = isolate_contour_datapoints(x, y, datagrid, low_output_resolution=True)
    for point in contours:
        try:
            interpolated_datagrid[np.where(y == point[1])[0][0], np.where(x == point[0])[0][0]] = point[2]
        except IndexError:
            pass
    return interpolated_datagrid


def interpolate_low_output_resolution(
    x: np.ndarray,
    y: np.ndarray,
    datagrid: np.ndarray,
    method: str = "rfb_thin_plate_spline",
    smoothing=0.01,
    use_fix_contours: bool = True,
    use_clip_to_data: bool = True,
    **kwargs,
):
    """
    Interpolates data in a grid with low output resolution (i.e. the data values are step-like, not smooth) by first
    isolating the contours of the data and then using methods available in the 'scipy.interpolate' module.

    :param x: (m) ndarray - The x coordinates of the datagrid
    :param y: (n) ndarray - The y coordinates of the datagrid
    :param datagrid:(n, m) ndarray - the datagrid to interpolate
    :param method: One of "nearest, linear, cubic" (using scipy.interpolate._ndgriddata.griddata) or any methods
    available in 'scipy.interpolate._rbfinterp.RBFInterpolator' prefixed by "rfb_". In case RFB is used, also smoothing
    and epsilon parameters can/need to be specified when appropriate. Other arguments for RBFInterpolator can be passed
    on, see their respective documentation for further information.
    :param fix_and_clip: #TODO
    :return: (n, m) ndarray - the interpolated datagrid
    """

    X, Y = np.meshgrid(x, y)
    contours_datapoints = isolate_contour_datapoints(x, y, datagrid, low_output_resolution=True)
    match method:
        case "nearest" | "linear" | "cubic":
            interpolated_datagrid = interpolate.griddata(
                contours_datapoints[:, 0:2],
                contours_datapoints[:, 2],
                (X, Y),
                method=method,
                fill_value=np.nan,
                **kwargs,
            )
        case _ if "rfb" in method:
            inputpoints = np.array([X, Y]).reshape(2, -1).T
            interpolated_datagrid = interpolate.RBFInterpolator(
                contours_datapoints[:, 0:2],
                contours_datapoints[:, 2],
                kernel=method[4:],
                smoothing=smoothing,
                **kwargs,
            )(inputpoints).reshape(y.size, x.size)
        case _:
            raise TypeError(f"Invalid method selected: {method}")
    if use_fix_contours:
        interpolated_datagrid = fix_contours(x, y, datagrid, interpolated_datagrid)
    if use_clip_to_data:
        interpolated_datagrid = clip_to_data(datagrid, interpolated_datagrid)
    return interpolated_datagrid
