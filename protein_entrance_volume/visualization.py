import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits import mplot3d


def matplotlib_mesh(verts, faces, grid_shape):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces])
    ax.add_collection3d(mesh)
    ax.set_xlim(0, grid_shape[0])
    ax.set_ylim(0, grid_shape[1])
    ax.set_zlim(0, grid_shape[2])
    plt.tight_layout()
    plt.show()


def matplotlib_points(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    mins = points.min(axis=0)
    maxes = points.max(axis=0)
    ax.set_xlim(mins[0], maxes[0])
    ax.set_ylim(mins[1], maxes[1])
    ax.set_zlim(mins[2], maxes[2])
    ax.scatter3D(*points.T)
    plt.show()
