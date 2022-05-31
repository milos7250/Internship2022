import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import splev, splprep

import functions
from bezier import evaluate_bezier
from functions import isolate_collinear

# USE FOR CLOSED CONTOURS

# Load and plot the original contour
line = np.loadtxt("data/RealContour1A.txt")
# Set last point equal to first to close the curve
if not np.all(line[0] == line[-1]):
    line = np.append(line, [line[0]], axis=0)

plt.figure(figsize=(11, 8))
plt.plot(line[:, 0], line[:, 1])


isolated_points = isolate_collinear(line, collinearity_tol=1e-2)

# Cubic splines
model, u = splprep(isolated_points.T, s=0, k=3, per=1)
interpolated = splev(np.linspace(0, 1, 10000), model)
plt.plot(interpolated[0], interpolated[1])

# Bezier
# path = evaluate_bezier(isolated_points, 50)
# plt.plot(path[:, 0], path[:, 1])
# plt.legend(["Original line", "Cubic spline", "Bezier spline"])

# Plot line points and isolated points
# plt.plot(line[:, 0], line[:, 1], "gx", markersize=10)
# plt.plot(isolated_points[:, 0], isolated_points[:, 1], "ro")

plt.show()
