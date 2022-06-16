from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, TwoSlopeNorm, BoundaryNorm, Normalize
from matplotlib.cm import ScalarMappable


class DEMScalarMappable(ScalarMappable):
    colors_sea = plt.cm.terrain(np.linspace(0, 0.17, 256))
    colors_land = plt.cm.terrain(np.linspace(0.25, 1, 256))
    all_colors = np.vstack((colors_sea, colors_land))

    def __init__(self, vmin, vmax, zero=0, zero_is_water=False):
        if vmax <= zero:
            norm = Normalize(vmin, vmax, clip=True)
            cmap = LinearSegmentedColormap.from_list("terrain_map", self.colors_sea)
        elif vmin == zero and zero_is_water:
            norm = Normalize(vmin, vmax, clip=True)
            cmap = LinearSegmentedColormap.from_list(
                "terrain_map", np.vstack((self.colors_sea[-1], self.colors_land[1:]))
            )
        elif (vmin == zero and zero_is_water is False) or vmin > zero:
            norm = Normalize(vmin, vmax, clip=True)
            cmap = LinearSegmentedColormap.from_list("terrain_map", self.colors_land)
        elif zero_is_water:
            norm = TwoSlopeNorm(vmin=vmin, vcenter=zero, vmax=vmax)
            cmap = LinearSegmentedColormap.from_list(
                "terrain_map", np.vstack((self.colors_sea, self.colors_sea[-1], self.colors_land[1:]))
            )
        else:
            norm = TwoSlopeNorm(vmin=vmin, vcenter=zero, vmax=vmax)
            cmap = LinearSegmentedColormap.from_list("terrain_map", self.all_colors)
        super().__init__(norm, cmap)

    def segmented(self, bin_bounds, kind):
        if kind == "bottom":
            norm = BoundaryNorm(bin_bounds, bin_bounds.size - 1, clip=True)
            cmap = LinearSegmentedColormap.from_list(
                "terrain_map_segmented", self.to_rgba(bin_bounds[:-1]), N=bin_bounds.size - 1
            )
            return ScalarMappable(norm, cmap)
        elif kind == "middle":
            norm = BoundaryNorm(
                [
                    1.5 * bin_bounds[0] - 0.5 * bin_bounds[1],
                    *(bin_bounds[0:-1] + bin_bounds[1:]) / 2,
                    1.5 * bin_bounds[-1] - 0.5 * bin_bounds[-2],
                ],
                bin_bounds.size,
            )
            cmap = LinearSegmentedColormap.from_list(
                "terrain_map_segmented", self.to_rgba(bin_bounds), N=bin_bounds.size
            )
            return ScalarMappable(norm, cmap)
        elif kind == "top":
            norm = BoundaryNorm(bin_bounds, bin_bounds.size - 1, clip=True)
            cmap = LinearSegmentedColormap.from_list(
                "terrain_map_segmented", self.to_rgba(bin_bounds[1:]), N=bin_bounds.size - 1
            )
            return ScalarMappable(norm, cmap)
        else:
            raise TypeError("Type can only be one of 'bottom', 'middle' or 'top'.")

    @property
    def lut(self):
        return self.__get_lut(self)

    def segmented_lut(self, bin_bounds, kind):
        return self.__get_lut(self.segmented(bin_bounds, kind))

    @staticmethod
    def __get_lut(scalarmappable: ScalarMappable):
        return (
            scalarmappable.to_rgba(np.linspace(scalarmappable.get_clim()[0], scalarmappable.get_clim()[1], 256)) * 255
        )
