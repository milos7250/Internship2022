import warnings

import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from skimage.measure import find_contours
from functions_1D import smooth_contour
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
    if low_output_resolution and levels is None:
        levels = np.unique(datagrid)
        levels = np.append(levels, 2 * levels[-1] - levels[-2])
        if levels.size > 100:
            raise RuntimeWarning("More than 100 levels were detected. Did you mean to pass low_output_resolution=True?")
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

    :param x: (m) ndarray - The x coordinates of the datagrid.
    :param y: (n) ndarray - The y coordinates of the datagrid.
    :param datagrid: (n, m) ndarray - The datagrid to isolate contours from.
    :param levels: Values at which to plot contours. Only necessary when low_output_resolution=False.
    :param low_output_resolution: If set to True, the values of datagrid are expected to be discrete. This is the case
    when the output resolution of the function is low (the output is binned).
    :param return_separate_contours: If set to True, the contour points are separated into a list by contours.
    :return: list of (k_i, 3) ndarrays - The isolated contours.
    """
    if low_output_resolution and levels is None:
        levels = np.unique(datagrid)
        levels = np.append(levels, 2 * levels[-1] - levels[-2])
        if levels.size > 100:
            raise RuntimeWarning("More than 100 levels were detected. Did you mean to pass low_output_resolution=True?")
        look_for_levels = (levels[0:-1] + levels[1:]) / 2
    elif levels is None:
        raise TypeError("Either low_output_resolution=True or levels must be specified.")
    else:
        look_for_levels = levels

    contours = list()
    # Find contours
    for idx, level in enumerate(look_for_levels):
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
    """
    Element-wise version of 'test' function.
    """
    min_index = np.where(original == np.array(levels))[0][-1]
    minimum = levels[min_index]
    maximum = levels[min_index + 1]
    res = minimum <= interpolated <= maximum
    return res


# Turns element-wise function into a vector function
__test_vectorized = np.vectorize(__test_elementwise, excluded=[2])


def test(datagrid, interpolated_datagrid):
    """
    Tests whether interpolated data is within the range suggested by original low output resolution data.

    :param datagrid: (n, m) ndarray - Original low output resolution data.
    :param interpolated_datagrid: (n, m) ndarray - Interpolated data.
    :return: (n, m) ndarray - Boolean array, True indicated the data in a given entry was inside the range, False
    otherwise.
    """
    levels = np.unique(datagrid)
    levels = np.append(levels, 2 * levels[-1] - levels[-2])
    return __test_vectorized(datagrid, interpolated_datagrid, levels)


def __clip_elementwise(original, interpolated, levels):
    """
    Element-wise version of 'clip' function.
    """
    min_index = np.where(original == np.array(levels))[0][-1]
    minimum = levels[min_index]
    maximum = levels[min_index + 1]
    if interpolated <= minimum:
        return minimum
    elif maximum <= interpolated:
        return maximum
    else:
        return interpolated


# Turns element-wise function into a vector function
__clip_vectorized = np.vectorize(__clip_elementwise, excluded=[2])


def clip_to_data(x, y, datagrid, xi, yi, interpolated_datagrid):
    """
    Clips interpolated data to the range suggested by original low output resolution data.

    :param datagrid: (n, m) ndarray - Original low output resolution data.
    :param interpolated_datagrid: (n, m) ndarray - Interpolated data.
    :return: (n, m) ndarray - The interpolated data. The data that were outside the suggested range are clipped to the
    endpoints of the range.
    """
    # TODO: Consider implementing case when the shapes are not the same
    if datagrid.shape != interpolated_datagrid.shape:
        warnings.warn("Different data shapes. Data not clipped.", RuntimeWarning)
        return interpolated_datagrid
    #     XI, YI = np.meshgrid(xi, yi)
    #     inputpoints = np.array([XI, YI]).reshape(2, -1).T
    #     datagrid = (
    #         interpolate.RegularGridInterpolator((x, y), datagrid, method="nearest")(inputpoints)
    #         .reshape(yi.size, xi.size)
    #         .T
    #     )
    interpolated_datagrid[np.isnan(interpolated_datagrid)] = datagrid[np.isnan(interpolated_datagrid)]
    levels = np.unique(datagrid)
    levels = np.append(levels, 2 * levels[-1] - levels[-2])
    return __clip_vectorized(datagrid, interpolated_datagrid, levels)


def fix_contours(x: np.ndarray, y: np.ndarray, datagrid, interpolated_datagrid):
    """
    Replaces contours from original data into the interpolated data. This makes sure that the original information about
    contours is perserved.

    :param x: (m) ndarray - The x coordinates of the datagrid.
    :param y: (n) ndarray - The y coordinates of the datagrid.
    :param datagrid: (n, m) ndarray - The datagrid before interpolation.
    :param interpolated_datagrid: (n, m) ndarray - The datagrid after interpolation.
    :return: (n, m) ndarray - The interpolated datagrid combined with contours from the original datagrid.
    """
    # TODO: Consider implementing case when the shapes are not the same
    if datagrid.shape != interpolated_datagrid.shape:
        warnings.warn("Different data shapes. Contours not fixed.", RuntimeWarning)
        return interpolated_datagrid
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
    xi: np.ndarray = None,
    yi: np.ndarray = None,
    method: str = "rfb_thin_plate_spline",
    smoothing: float = 1e-3,
    use_fix_contours: bool = True,
    use_clip_to_data: bool = True,
    **kwargs,
):
    """
    Interpolates data in a grid with low output resolution (i.e. the data values are step-like, not smooth) by first
    isolating the contours of the data and then using methods available in the 'scipy.interpolate' module.

    :param x: (m) ndarray - The x coordinates of the datagrid.
    :param y: (n) ndarray - The y coordinates of the datagrid.
    :param datagrid:(n, m) ndarray - The datagrid to interpolate.
    :param xi: (m) ndarray - The x coordinates to interpolate the datagrid to.
    :param yi: (n) ndarray - The x coordinates to interpolate the datagrid to.
    :param method: One of "nearest, linear, cubic" (using scipy.interpolate._ndgriddata.griddata) or any methods
    available in 'scipy.interpolate._rbfinterp.RBFInterpolator' prefixed by "rfb_". In case RFB is used, also smoothing
    and epsilon parameters can/need to be specified when appropriate. Other arguments for RBFInterpolator can be passed
    on, see their respective documentation for further information.
    :param smoothing: The smoothing applied to RFB interpolated data. If contours are wobbly, increase smoothing.
    :param use_fix_contours: If True, applies 'fix_contours' to interpolated data.
    :param use_clip_to_data: If True, applies 'clip_to_data' to interpolated data.
    :return: (n, m) ndarray - the interpolated datagrid
    """
    if xi is None and yi is None:
        xi, yi = x, y
    XI, YI = np.meshgrid(xi, yi)
    contours_datapoints = isolate_contour_datapoints(x, y, datagrid, low_output_resolution=True)
    match method:
        case "nearest" | "linear" | "cubic":
            interpolated_datagrid = interpolate.griddata(
                contours_datapoints[:, 0:2],
                contours_datapoints[:, 2],
                (XI, YI),
                method=method,
                fill_value=np.nan,
            )
        case _ if "rfb" in method:
            """
            Limit the number of points passed to RBFInterpolator in order to cut down time. Taking points in regular
            intervals from contours should not affect the results much, as it removes points which are close to
            unremoved points, and these can be interpolated well.
            """
            if contours_datapoints.shape[0] > 5000:
                old = contours_datapoints.shape[0]
                step = contours_datapoints.shape[0] // 5000 + 1
                contours_datapoints = contours_datapoints[::step]
                # print(f"The number of contour datapoints was shrunk from {old} to {contours_datapoints.shape[0]}.")
            # else:
            #     print(f"The number of contour datapoints is {contours_datapoints.shape[0]}.")

            inputpoints = np.array([XI, YI]).reshape(2, -1).T
            interpolated_datagrid = interpolate.RBFInterpolator(
                contours_datapoints[:, 0:2],
                contours_datapoints[:, 2],
                kernel=method[4:],
                smoothing=smoothing,
                **kwargs,
            )(inputpoints).reshape(yi.size, xi.size)
        case _:
            raise TypeError(f"Invalid method selected: {method}")
    if use_fix_contours:
        interpolated_datagrid = fix_contours(xi, yi, datagrid, interpolated_datagrid)
    if use_clip_to_data:
        interpolated_datagrid = clip_to_data(x, y, datagrid, xi, yi, interpolated_datagrid)
    return interpolated_datagrid
