# Bin altitudes to this many meters
block_height = 50.0
# Sample original grid once every block_stride
block_stride = 8


import matplotlib.pyplot as plt
import numpy as np

# Comment the next two lines if you do not have LaTeX
plt.rc("text", usetex=True)
plt.rc("font", family="serif")


# Taken from Ordnance Survey (OS) data, each grid with 10km side
# Note - Y-axis must be inverted for plotting
data = np.genfromtxt("../data/NO44.asc", skip_header=5)


# Create 10km x 10km coordinates
north = np.linspace(0, 10, num=np.shape(data)[0])
east = np.linspace(0, 10, num=np.shape(data)[1])
north, east = np.meshgrid(north, east)


# Create grids with heights binned and strided
data_block_height = block_height * np.floor(data / block_height)
data_block_stride = data_block_height[::block_stride, ::block_stride]


# Create a handle for the figure and axes
fig = plt.figure("AltitudeReal", figsize=(9, 8))
ax = fig.gca()

# Plot 2D function; rasterized=True arguments is good for saving as PDF
color_plot = plt.pcolormesh(
    north, east, data[::-1, :], vmin=0, vmax=max(data.flatten()), cmap="gist_earth", rasterized=True
)

# Ticks and labels
plt.tick_params(axis="both", which="major", labelsize=20, direction="in", bottom=True, top=True, left=True, right=True)
plt.ylabel(r"Northing [km]", fontsize=22)
plt.xlabel(r"Easting [km]", fontsize=22)

# Side colorbar
cbar = fig.colorbar(color_plot)
cbar.ax.tick_params(labelsize=20, direction="in", left=True, right=True)
cbar.set_label(label=r"Altitude [m]", fontsize=22)

# Save figure, can be .pdf or .png
plt.savefig("AltitudeReal.pdf", dpi=100, bbox_inches="tight", pad_inches=0.1)

fig2 = plt.figure("AltitudeBlockHeight", figsize=(9, 8))
ax2 = fig2.gca()

color_plot2 = plt.pcolormesh(
    north, east, data_block_height[::-1, :], vmin=0, vmax=max(data.flatten()), cmap="gist_earth", rasterized=True
)

plt.tick_params(axis="both", which="major", labelsize=20, direction="in", bottom=True, top=True, left=True, right=True)
cbar2 = fig2.colorbar(color_plot2)
cbar2.ax.tick_params(labelsize=20, direction="in", left=True, right=True)
cbar2.set_label(label=r"Altitude [m]", fontsize=22)
plt.ylabel(r"Northing [km]", fontsize=22)
plt.xlabel(r"Easting [km]", fontsize=22)

plt.savefig("AltitudeBlockHeight.pdf", format="pdf", dpi=100, bbox_inches="tight", pad_inches=0.1)


fig3 = plt.figure("AltitudeBlockBoth", figsize=(9, 8))
ax3 = fig3.gca()

color_plot3 = plt.pcolormesh(
    north[::block_stride, ::block_stride],
    east[::block_stride, ::block_stride],
    data_block_stride[::-1, :],
    vmin=0,
    vmax=max(data.flatten()),
    cmap="gist_earth",
    rasterized=True,
)

plt.tick_params(axis="both", which="major", labelsize=20, direction="in", bottom=True, top=True, left=True, right=True)
cbar3 = fig3.colorbar(color_plot3)
cbar3.ax.tick_params(labelsize=20, direction="in", left=True, right=True)
cbar3.set_label(label=r"Altitude [m]", fontsize=22)
plt.ylabel(r"Northing [km]", fontsize=22)
plt.xlabel(r"Easting [km]", fontsize=22)

plt.savefig("AltitudeBlockBoth.pdf", format="pdf", dpi=100, bbox_inches="tight", pad_inches=0.1)

plt.show()
