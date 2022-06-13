import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate

from external.bezier import evaluate_bezier
from functions_1D import isolate_collinear

# USE FOR FUNCTIONS

# Load and plot the original function
line = np.loadtxt("data/Simple_Line1A.txt")
plt.figure(figsize=(11, 8))
plt.plot(line[:, 0], line[:, 1])


isolated_points = isolate_collinear(line)

# CUBIC SPLINE
model = interpolate.CubicSpline(isolated_points[:, 0], isolated_points[:, 1], bc_type="natural", extrapolate=False)
new_x = np.linspace(line[0, 0], line[-1, 0], 1000)
new_y = model(new_x)
plt.plot(new_x, new_y)

# BEZIER
path = evaluate_bezier(isolated_points, 50)
plt.plot(path[:, 0], path[:, 1])

plt.legend(["Original line", "Cubic spline", "Bezier spline"])

# Plot line points and isolated points
# plt.plot(line[:, 0], line[:, 1], "gx", markersize=10)
# plt.plot(isolated_points[:, 0], isolated_points[:, 1], 'ro')

plt.show()
