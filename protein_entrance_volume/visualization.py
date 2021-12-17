"""
Created by: Mitchell Walls
Email: miwalls@siue.edu
"""
import matplotlib.pyplot as plt


def matplotlib_points(*points):
    """
    Simple matplotlib function to plot any number of arrays of cartesian
    (x,y,z) points each will be colored different by default.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for p in points:
        ax.scatter3D(*p.T)
    plt.show()
