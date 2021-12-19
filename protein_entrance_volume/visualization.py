"""
Created by: Mitchell Walls
Email: miwalls@siue.edu
"""
import matplotlib.pyplot as plt
from protein_entrance_volume import exception


def coordinates(*coords, plot_type="scatter"):
    """
    Takes a set of 3D coords and plots them depending on the type.
    """
    if plot_type == "scatter":
        scatter(*coords)
    else:
        raise exception.InvalidPlotType(("scatter",))


def scatter(*coords):
    """
    Simple matplotlib function to plot any number of arrays of cartesian
    (x,y,z) coords each will be colored different by default.
    """
    fig = plt.figure()
    ax_subplot = fig.add_subplot(111, projection='3d')
    for coord in coords:
        ax_subplot.scatter3D(*coord.T)
    plt.show()
