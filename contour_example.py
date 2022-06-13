import numpy as np
from matplotlib import pyplot as plt

from external.bezier import evaluate_bezier
from functions_1D import isolate_from_collinear, smooth_contour

"""
Script interpolates 1D parametric paths in 2D by using Cubic splines.
"""

# Set up Figure
plt.rc("text", usetex=True)
plt.rc("font", family="serif")
size = 0.7
fig = plt.figure(figsize=(16 * size, 9 * size))
fig.tight_layout()
axes = fig.subplots(1, 3, sharex="all", sharey="all", subplot_kw={"frame_on": False, "xticks": [], "yticks": []})

# Load and plot the original contour
contour = np.loadtxt("data/RealContour1A.txt")
axes[0].plot(contour[:, 0], contour[:, 1], color=plt.get_cmap("tab10")(0))
axes[0].set_title("Terraced contour")

cubic_spline = smooth_contour(contour, collinearity_tol=1e-2)
axes[1].plot(cubic_spline[:, 0], cubic_spline[:, 1], color=plt.get_cmap("tab10")(1))
axes[1].set_title("Cubic spline")

# Bezier
contour_points = isolate_from_collinear(contour, collinearity_tol=1e-2)
bezier_curve = evaluate_bezier(contour_points, 20)
axes[2].plot(bezier_curve[:, 0], bezier_curve[:, 1], color=plt.get_cmap("tab10")(2))
axes[2].set_title("Bezier curve")

# Plot line points and isolated points
# plt.plot(line[:, 0], line[:, 1], "gx", markersize=10)
# plt.plot(isolated_points[:, 0], isolated_points[:, 1], "ro")

plt.savefig("images/2D_Contour.png")
plt.show()
