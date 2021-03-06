import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from functions_2D import interpolate_discretized_data
from mayavi import mlab
from helpers.colors import DEMScalarMappable

"""
This script implements a comparative test for 2D interpolation techniques
"""


block_stride = 1
height_levels = 50
dim_x = 10 * 1e3
dim_y = 10 * 1e3


def mean_square_error(original_datagrid, interpolated_datagrid):
    """
    Returns the mean square error of the interpolated datagrid.
    """
    return np.average(np.nan_to_num(original_datagrid - interpolated_datagrid) ** 2)


def test_epsilons_for_method(xo, yo, original_datagrid, x, y, datagrid, epsilons, method="rbf_multiquadric"):
    """
    Interpolates the datagrid using the specified method once for each specified epsilon and plots the corresponding
    mean square errors of the interpolated datagrid.

    :param xo: The x coordinates of the original datagrid.
    :param yo: The y coordinates of the original datagrid.
    :param original_datagrid: The original datagrid.
    :param x: The x coordinates of the datagrid to interpolate.
    :param y: The y coordinates of the datagrid to interpolate.
    :param datagrid: The datagrid to interpolate.
    :param epsilons: A list of epsilons to test.
    :param method: The method to use for interpolation. Needs to be supported by
    'functions_2D.interpolate_discretized_data'.
    """
    mses = list()
    for epsilon in epsilons:
        mse = mean_square_error(
            original_datagrid,
            interpolate_discretized_data(x, y, datagrid, xo, yo, method=method, epsilon=epsilon),
        )
        mses.append(mse)
        print(f"Epsilon: {epsilon}, Mean Square Error: {mse}")

    plt.figure()
    plt.plot(epsilons, mses)
    plt.show()


def test_methods(show: bool = True, save: bool = False):
    """
    Interpolates the loaded datagrid using methods available in 'functions_2D.interpolate_discretized_data' and
    calculates the mean square errors of the interpolation. Optionally produces comparison plots of the interpolation.

    :param show: Show plots of the interpolated datagrid.
    :param save: Save the plots of the interpolated datagrid to file.
    """

    def plot_result(show, save):
        """
        Creates plots from the interpolation
        """
        # Set up Figure
        plt.rc("text", usetex=True)
        plt.rc("font", family="serif")
        size = 1
        fig = plt.figure(figsize=(16 * size, 9 * size))
        axes = fig.subplots(
            1,
            4,
            sharex="all",
            sharey="all",
            subplot_kw={
                "adjustable": "box",
                "aspect": "equal",
                "xticks": [i for i in np.linspace(0, dim_x, 6)],
                "yticks": [i for i in np.linspace(0, dim_y, 6)],
                "xticklabels": [f"{i:d} km" for i in np.linspace(0, dim_x / 1e3, 6, dtype=int)],
                "yticklabels": [f"{i:d} km" for i in np.linspace(0, dim_x / 1e3, 6, dtype=int)],
                "xlim": (0, dim_x),
                "ylim": (0, dim_y),
            },
            gridspec_kw={"hspace": 0.3},
        )
        filename = method_name.replace("\n", " ").replace(" ", "_")

        axes[0].pcolormesh(xo, yo, original_datagrid, cmap=colors.cmap, norm=colors.norm, rasterized=True)
        axes[0].set_title("Original Raster")
        axes[0].set_ylabel("Northing")
        axes[1].pcolormesh(x, y, datagrid, cmap=colors.cmap, norm=colors.norm, rasterized=True)
        axes[1].set_title("Discretized Raster")
        axes[1].set_xlabel("Easting")
        axes[2].pcolormesh(xo, yo, interpolated_datagrid, cmap=colors.cmap, norm=colors.norm, rasterized=True)
        axes[2].set_title(f"Interpolated Raster, using\n{method_name}")
        plt.colorbar(colors, ticks=levels, ax=axes[0:3].ravel().tolist(), shrink=0.4, aspect=15)
        if not show:
            mlab.options.offscreen = True
        mlab.figure(bgcolor=(1, 1, 1))
        surf = mlab.surf(
            yo,
            xo,
            np.rot90(interpolated_datagrid.T),
            warp_scale=5,
            colormap="terrain",
            vmin=vmin,
            vmax=vmax,
        )
        surf.module_manager.scalar_lut_manager.lut.table = colors.lut
        if save:
            mlab.savefig(f"images/differences/{filename}_3D.png", magnification=10)
            # Use ImageMagick to remove background from image.
            os.system(
                f"convert images/differences/{filename}_3D.png -transparent white -trim +repage images/differences/{filename}_3D.png"
            )
        if show:
            mlab.show()

        diff = np.nan_to_num(interpolated_datagrid - original_datagrid)
        cmap_diff = plt.get_cmap("coolwarm")
        norm_diff = Normalize(-np.max(abs(diff)), np.max(abs(diff)))
        colors_diff = ScalarMappable(cmap=cmap_diff, norm=norm_diff)
        axes[3].pcolormesh(xo, yo, diff, cmap=cmap_diff, norm=norm_diff, rasterized=True)
        axes[3].set_title("Difference of Interpolated\nand Original Raster")
        axes[3].set_xlabel("Easting")
        plt.colorbar(colors_diff, ax=axes[3], fraction=0.05, pad=0.1)
        if save:
            plt.savefig(f"images/differences/{filename}_2D.svg", transparent=True, dpi=300, bbox_inches="tight")
        if show:
            plt.show()

    # Comment / uncomment lines with methods to test
    for method, method_name in [
        [
            "linear",
            "Delaunay Triangulation\nand Barycentric Interpolation",
        ],  # The spline interpolators don't work well
        ["cubic", "Delaunay Triangulation\nand Clough-Tocher scheme"],
        # ["rbf_linear", "Radial Basis\nLinear Function"],
        ["rbf_thin_plate_spline", "Radial Basis\nThin Plate Spline"],
        # ["rbf_cubic", "Radial Basis\nCubic Function"],
        # ["rbf_quintic", "Radial Basis\nQuintic Function"],
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
        if show or save:
            plot_result(show, save)

    # Comment / uncomment lines with methods to test
    for epsilon, method, method_name in [
        # Still not sure what value for epsilon is the best
        # [3, "rbf_multiquadric", "Radial Basis\nMultiquadric Function"],
        # [10, "rbf_multiquadric", "Radial Basis\nMultiquadric Function"],
        # [30, "rbf_multiquadric", "Radial Basis\nMultiquadric Function"],
        # [100, "rbf_multiquadric", "Radial Basis\nMultiquadric Function"],
        # Inverse methods don't work well in sparse places.
        # [10, "rbf_inverse_multiquadric", "Radial Basis\nInverse Multiquadric Function"],
        # [3.5, "rbf_inverse_quadratic", "Radial Basis\nInverse Quadratic Function"],
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
        if show or save:
            plot_result(show, save)


for tile in ["NO44"]:
    print(f"Tile name: {tile}.")
    original_datagrid = np.loadtxt(f"../data/{tile}.asc", skiprows=5)[::-1, :]
    # Each tile is of dimension 10km x 10km, sampled by 50m, thus we have 200 x 200 samples
    xo = np.linspace(0, dim_x, original_datagrid.shape[1])
    yo = np.linspace(0, dim_y, original_datagrid.shape[0])
    minimum = int(np.floor(np.min(original_datagrid) / height_levels) * height_levels)
    maximum = int(np.ceil(np.max(original_datagrid) / height_levels) * height_levels)
    levels = np.arange(minimum, maximum + 1e-10, height_levels, dtype=int)

    # Set up Colormaps
    vmin = minimum / 1.1 if minimum > 0 else minimum * 1.1  # Leave extra 10% for interpolation overshoot
    vmax = maximum * 1.1  # Leave extra 10% for interpolation overshoot
    colors = DEMScalarMappable(vmin, vmax)

    # Discretize datagrid
    datagrid = height_levels * np.floor(original_datagrid[::block_stride, ::block_stride] / height_levels)
    x = np.linspace(0, dim_x, datagrid.shape[1])
    y = np.linspace(0, dim_y, datagrid.shape[0])
    test_methods(show=True, save=False)
