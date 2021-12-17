import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits import mplot3d


def matplotlib_points(*points):
    """
    Simple matplotlib function to plot any number of arrays of cartesian
    (x,y,z) points.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for p in points:
        ax.scatter3D(*p.T)
    plt.show()
