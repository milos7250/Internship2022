import pathlib

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy import interpolate
from PIL import Image

import helpers.save_figure_position
from functions_2D import plot_contours, interpolate_low_output_resolution

image = Image.open("data/FictionalData.png")
colordict = {0: 7, 1: 0, 2: 6, 3: 1, 4: 5, 5: 2, 6: 4, 7: 3}
change_color = np.vectorize(lambda x: colordict[x])

cmap = "terrain"
norm = Normalize(0, 7)
colors = ScalarMappable(norm=norm, cmap=cmap)
# noinspection PyTypeChecker
image = np.array(image)
image = change_color(image)
x = np.arange(0, image.shape[1])
y = np.arange(0, image.shape[0])


plt.subplot(121)
plt.imshow(image, cmap=cmap, norm=norm, interpolation="none", resample=False)
plt.colorbar()


stride = 5
interpolated_image = interpolate_low_output_resolution(x, y, image - 1, x[::stride], y[::stride])
interpolated_image = np.where(image[::stride, ::stride] == 0, image[::stride, ::stride], interpolated_image)
plt.subplot(122)
plt.imshow(interpolated_image, cmap=cmap, norm=norm, interpolation="none", resample=False)

helpers.save_figure_position.ShowLastPos(plt)
