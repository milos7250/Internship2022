import numpy as np
from matplotlib import pyplot as plt

from external.bezier import evaluate_bezier
from functions_1D import isolate_collinear, smooth_contour

# USE FOR CLOSED CONTOURS

# Load and plot the original contour
contour = np.loadtxt("data/RealContour1A.txt")
# Set last point equal to first to close the curve
if not np.all(contour[0] == contour[-1]):
    contour = np.append(contour, [contour[0]], axis=0)

plt.figure(figsize=(11, 8))
plt.plot(contour[:, 0], contour[:, 1])

interpolated_contour = smooth_contour(contour, collinearity_tol=1e-2)
plt.plot(interpolated_contour[:, 0], interpolated_contour[:, 1])

# Bezier
contour_points = isolate_collinear(contour, closed=True, collinearity_tol=1e-2)
path = evaluate_bezier(contour_points, 20)
plt.plot(path[:, 0], path[:, 1])
plt.legend(["Original line", "Cubic spline", "Bezier spline"])

# Plot line points and isolated points
# plt.plot(line[:, 0], line[:, 1], "gx", markersize=10)
# plt.plot(isolated_points[:, 0], isolated_points[:, 1], "ro")

plt.show()
