import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from functions_2D import interpolate_discretized_data, plot_contours
from helpers.save_figure_position import ShowLastPos

"""
This script implements a comparative test for 2D interpolation techniques
"""


block_stride = 1
height_levels = 50


def mean_square_error(original_datagrid, interpolated_datagrid):
    return np.average(np.nan_to_num(original_datagrid - interpolated_datagrid) ** 2)


def test_epsilons_for_method(xo, yo, original_datagrid, x, y, datagrid, epsilons):
    mses = list()
    for epsilon in epsilons:
        mse = mean_square_error(
            original_datagrid,
            interpolate_discretized_data(x, y, datagrid, xo, yo, method="rbf_multiquadric", epsilon=epsilon),
        )
        mses.append(mse)
        print(f"Epsilon: {epsilon}, Mean Square Error: {mse}")

    plt.figure()
    plt.plot(epsilons, mses)
    plt.show()


def test_methods(xo, yo, original_datagrid, x, y, datagrid, plot):
    def plot_result():
        fig = plt.figure(figsize=(16, 9))
        axes = fig.subplots(1, 4, subplot_kw={"aspect": "equal"})
        plt.tight_layout()
        axes[0].pcolormesh(xo, yo, original_datagrid, cmap=cmap, norm=norm)
        axes[1].pcolormesh(x, y, datagrid, cmap=cmap, norm=norm)
        axes[2].pcolormesh(xo, yo, interpolated_datagrid, cmap=cmap, norm=norm)
        plt.colorbar(colors, ticks=levels, ax=axes[0:3].ravel().tolist())

        diff = np.nan_to_num(interpolated_datagrid - original_datagrid)
        cmap_diff = plt.get_cmap("coolwarm")
        norm_diff = Normalize(-np.max(abs(diff)), np.max(abs(diff)))
        colors_diff = ScalarMappable(cmap=cmap_diff, norm=norm_diff)
        axes[3].pcolormesh(xo, yo, diff, cmap=cmap_diff, norm=norm_diff)
        plt.colorbar(colors_diff, ax=axes[3])
        plt.show()

    for method in [
        "linear",  # The spline interpolator don't work well
        "cubic",
        "rbf_linear",
        "rbf_thin_plate_spline",
        "rbf_cubic",
        "rbf_quintic",
    ]:
        interpolated_datagrid = interpolate_discretized_data(
            x,
            y,
            datagrid,
            xo,
            yo,
            method=method,
        )
        print(f"Method: {method}, Mean Error: {mean_square_error(original_datagrid, interpolated_datagrid)}.")
        if plot:
            plot_result()
    for epsilon, method in [
        [3, "rbf_multiquadric"],  # Still not sure what value for epsilon is the best
        [10, "rbf_multiquadric"],
        [30, "rbf_multiquadric"],
        [100, "rbf_multiquadric"],
        [10, "rbf_inverse_multiquadric"],  # Inverse methods don't work well in sparse places.
        [3.5, "rbf_inverse_quadratic"],
    ]:
        interpolated_datagrid = interpolate_discretized_data(
            x,
            y,
            datagrid,
            xo,
            yo,
            method=method,
            epsilon=epsilon,
        )
        print(
            f"Method: {method}, Epsilon: {epsilon}, Mean Error: {mean_square_error(original_datagrid, interpolated_datagrid)}."
        )
        if plot:
            plot_result()


# for tile in ["NN17"]:
for tile in ["NN17", "NO33", "NO44", "NO51"]:
    print(f"Tile name: {tile}.")
    original_datagrid = np.loadtxt(f"data/{tile}.asc", skiprows=5)[::-1, :]
    # Each tile is of dimension 10km x 10km, sampled by 50m, thus we have 200 x 200 samples
    xo = np.linspace(0, 10, original_datagrid.shape[1])
    yo = np.linspace(0, 10, original_datagrid.shape[0])
    maximum = int(np.ceil(np.max(original_datagrid)))
    levels = [i for i in range(0, maximum + height_levels, height_levels)]
    maximum = levels[-1]

    cmap = plt.get_cmap("terrain")
    norm = Normalize(
        min(np.min(original_datagrid) * 1.1, -10), maximum * 1.1
    )  # Leave extra 10% for interpolation overshoot
    colors = ScalarMappable(norm=norm, cmap=cmap)
    datagrid = height_levels * np.floor(original_datagrid[::block_stride, ::block_stride] / height_levels)
    x = np.linspace(0, 10, datagrid.shape[1])
    y = np.linspace(0, 10, datagrid.shape[0])
    test_methods(xo, yo, original_datagrid, x, y, datagrid, plot=False)
