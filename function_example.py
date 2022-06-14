import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate

from external.bezier import evaluate_bezier
from functions_1D import isolate_from_collinear

"""
Script interpolates 1D functions by using Cubic splines and Bezier curves.
"""

# Set up Figure
plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=16)
size = 0.5
fig = plt.figure(figsize=(16 * size, 9 * size))
fig.tight_layout()
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

# Load and plot the original function
line = np.loadtxt("data/Simple_Line1A.txt")
plt.plot(line[:100, 0], line[:100, 1], "o", markersize=2)
plt.legend(["Original line"], frameon=False)
plt.savefig("images/1D_Function.png")
plt.show()

fig = plt.figure(figsize=(16 * size, 9 * size))

# Isolate points from collinear segments
isolated_points = isolate_from_collinear(line)

# CUBIC SPLINE
cubic_spline = interpolate.CubicSpline(
    isolated_points[:, 0], isolated_points[:, 1], bc_type="natural", extrapolate=False
)
x = np.linspace(line[0, 0], line[-1, 0], 1000)
y = cubic_spline(x)
plt.plot(line[:, 0], line[:, 1], "o", markersize=2)
plt.plot(x, y)

# BEZIER CURVE
bezier_curve = evaluate_bezier(isolated_points, 20)
plt.plot(bezier_curve[:, 0], bezier_curve[:, 1])

plt.legend(["Original line", "Cubic spline", "Bezier curve"], frameon=False)

# Plot line points and isolated points
# plt.plot(line[:, 0], line[:, 1], "gx", markersize=10)
plt.plot(isolated_points[:, 0], isolated_points[:, 1], "ro", markersize=3)

plt.savefig("images/1D_Function_Interpolated.png")
plt.show()
