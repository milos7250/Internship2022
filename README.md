This repository contains Python scripts, Unity apps and report created by me during my EMS Bursary studentship at the University of Dundee in 2022.

## Extract
My research involved investigating various methods of interpolating functions and contours in one, two and three dimensions. The interpolation considered real-valued functions and their contours in two different settings. The first one was interpolating functions whose range is discretized, where the goal was to interpolate it back into a continuous range. The second was interpolating functions whose domain resolution is low, with the aim to interpolate the range to a domain with higher resolution. Various methods, such as fitting piecewise polynomials, radial basis function interpolation and convolutions were applied to functions and the results were compared on digital elevation maps in 2D and binary data from solar wind simulation in 3D.

-> ![Discretized elevation map](images/newsletter/Newsletter.png =x250) <-

Digital elevation maps are mathematically real-valued bivariate functions, so one part of my research included comparing bivariate and multivariate interpolation methods. The terraced image in the middle is the result of creating a surface from an elevation map with its range discretized into 50 m steps. The smooth image on the right was produced by isolating the contour lines from the discretized range and using them as the input points in thin-plate spline interpolation.

-> ![Blocky isosurface](images/3D/Dataset1.png =x250) ![Interpolated isosurface](images/3D/Dataset1_Gaussian.png =x250) <-

The second part of the research involved interpolating binary data from simulations of the Sun's corona. The images here represent a boundary between open flux and closed flux magnetic fields. The image on the left is produced by applying marching cubes algorithm without any interpolation of the data, whereas the image on the right was produced after convolution of the data with a Gaussian kernel.

-> ![Unity visualization](images/3D/Dataset1_Unity.png =x250) <-

The research also included using the results of the interpolated surfaces in Unity, a popular game engine, to create more advanced visualizations and animations for outreach purposes. The goal was to visualize the magnetic field as a "force field" around the Sun.

## Outcomes
The full outcomes of the studentship are summarized in the [report](report/Report.pdf). Along with the images and animations produced in Unity, I made a reusable template for creating similar visualizations from future simulations. The template and instructions for it's usage are located in the [Unity folder](Unity).
