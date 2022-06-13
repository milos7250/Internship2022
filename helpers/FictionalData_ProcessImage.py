import numpy as np
from PIL import Image

image = Image.open("../data/FictionalData.png")
colordict = {0: 7, 1: 0, 2: 6, 3: 1, 4: 5, 5: 2, 6: 4, 7: 3}
change_color = np.vectorize(lambda x: colordict[x])
# noinspection PyTypeChecker
image = np.array(image)
image = change_color(image)
