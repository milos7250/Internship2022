import matplotlib.axis


def zoom_figure(plt, x0, y0, dx, aspect=None):
    if aspect is matplotlib.axis.Axis:
        ax = aspect
        aspect = (ax.get_ylim()[1] - ax.get_ylim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0])
    else:
        plt.gcf().set_figwidth(plt.gcf().get_figheight() * aspect)
    dy = aspect * dx
    plt.xlim(x0 - dx, x0 + dx)
    plt.ylim(y0 - dy, y0 + dy)
