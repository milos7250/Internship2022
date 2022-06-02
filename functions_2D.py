import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from skimage.measure import find_contours
from functions_1D import smooth_contour, isolate_collinear
from scipy import interpolate


def plot_contours(
    datagrid: np.ndarray, levels: list, interpolate: bool, ax: Axes, colors: ScalarMappable, plot_title: str = None
):
    """
    Finds contours of data at specified levels using 'skimage.measure._find_contours.find_contours' and plots them on
    the specified axis.

    :param datagrid: (n, m) ndarray - the datagrid to plot contours from.
    :param levels: Values at which to plot contours.
    :param interpolate: Whether to interpolate the found contours using 'functions_1D.smooth_contour'.
    :param ax: The pyplot axis to plot to.
    :param colors: ScalarMappable instance used to determine contour colors.
    :param plot_title: The title of the plot
    :return: None
    """
    # Find contours for each level
    for level in levels:
        for contour in find_contours(datagrid.T, level):
            # Plot smoothened contour in second row
            if interpolate:
                contour = smooth_contour(contour)
            ax.plot(contour[:, 0], contour[:, 1], color=colors.to_rgba(level))
        title = ""
        if plot_title:
            title += f"{plot_title}\n"
        if interpolate:
            title += "Interpolated "
        ax.set_title(f"{title}Contours")


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
        isolated = isolate_collinear(np.column_stack([y, x_stripe]), closed=False)
        if borders:
            isolated = np.append(isolated, [[y[0], x_stripe[0]], [y[-1], x_stripe[-1]]], axis=0)
        isolated_datapoints = np.append(
            isolated_datapoints, np.insert(isolated, 0, np.repeat(xi, isolated.shape[0]), axis=1), axis=0
        )
    return isolated_datapoints


def interpolate_low_output_resolution(
    datagrid: np.ndarray, method: str = "linear", collinearity_tol: float = None, **kwargs
):
    """
    Interpolates data in a grid with low output resolution (i.e. the data values are step-like, not smooth). It uses
    methods available in the 'scipy.interpolate' module. Can be used for non-regularly spaced inputs.

    :param datagrid:(n, m) ndarray - the datagrid to interpolate
    :param method: One of "nearest, linear, cubic" (using scipy.interpolate._ndgriddata.griddata) or any methods
    available in 'scipy.interpolate._rbfinterp.RBFInterpolator' prefixed by "rfb_". In case RFB is used, also smoothing
    and epsilon parameters need to be specified when appropriate. Other arguments for RBFInterpolator can be passed on,
    see their respective documentation for further information.
    :param collinearity_tol: Tolerance for 'functions_1D.are_collinear'. Needs to be adjusted to fit size of data.
    :return: (n, m) ndarray - the interpolated datagrid
    """

    x = np.arange(datagrid.shape[0])
    y = np.arange(datagrid.shape[1])
    X, Y = np.meshgrid(x, y)
    isolated_datapoints = isolate_datapoints(datagrid, borders=True, collinearity_tol=collinearity_tol)
    match method:
        case "nearest" | "linear" | "cubic":
            interpolated_datagrid = interpolate.griddata(
                isolated_datapoints[:, 0:2], isolated_datapoints[:, 2], (X, Y), method=method
            )
        case _ if "rfb" in method:
            inputpoints = np.array([X, Y]).reshape(2, -1).T
            interpolated_datagrid = interpolate.RBFInterpolator(
                isolated_datapoints[:, 0:2], isolated_datapoints[:, 2], kernel=method[4:], **kwargs
            )(inputpoints).reshape(x.size, y.size)
        case _:
            raise TypeError(f"Invalid method selected: {method}")

    return interpolated_datagrid
