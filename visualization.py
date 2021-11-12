"""
Visualization of the cavity and atoms.
Created by: Mitchell Walls
Email: miwalls@siue.edu
"""
from numba import njit, prange
from numba.experimental import jitclass
from numba.typed import Dict, List
from numba import types
import numpy as np
import matplotlib.cm
from operator import itemgetter
from os import getcwd
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
# import open3d as o3d
from time import time_ns


def generate_xyz(xyzs):
    """
    Generate xyz of all of the caviy voxels use grid size for visualization
    """
    with open('plot.xyz', 'w+') as fh:
        for xyz in xyzs:
            fh.write('{},{},{}\n'.format(*xyz))


def generate_html(all_spheres, cavity_voxels, cavity_center, cavity_shift, grid_size):
    """
    Generates visualization using html file.
    """
    # spheres = all_spheres
    spheres = all_spheres[all_spheres['id'] != 'boundary']
    spheres = spheres[spheres['id'] != 'inner']
    spheres = spheres[spheres['id'] != 'outer']
    shift = spheres[['x', 'y', 'z']].min().to_numpy()
    spheres[['x', 'y', 'z']] -= shift
    cmap = matplotlib.cm.get_cmap('Greens')
    cavity_voxels -= shift
    radii_cm = {}
    segment = 1 / len(spheres.radius.unique())
    for i, radius in enumerate(spheres.radius.unique()):
        color = np.array(cmap(i * segment))
        radii_cm[radius] = (color)

    spheres_colors = []
    for index, sphere in spheres.iterrows():
        color = radii_cm[sphere.radius]
        if sphere.outer_residue:
            c1 = color[1]
            c2 = color[0]
            c3 = color[2]
        elif sphere.inner_residue:
            c1 = color[0]
            c2 = color[2]
            c3 = color[1]
        else:
            c1 = color[0]
            c2 = color[1]
            c3 = color[2]
        spheres_colors.append(np.array([c1, c2, c3]))
    spheres_colors = np.array(spheres_colors).T

    # Changing to marking boundary residues.
    # f = itemgetter(*spheres.radius.to_list())
    # spheres_colors = np.array(f(radii_cm)).T

    centroid = cavity_center - shift

    maxes = spheres[['x', 'y', 'z']].max()
    camera_position = tuple(centroid + [0, maxes.y, maxes.z * 1.5])

    cmap = matplotlib.cm.get_cmap('Greens')
    placeholders = {}
    placeholders["POCKETS_X_PLACEHOLDER"] = list(cavity_voxels.T[0])
    placeholders["POCKETS_Y_PLACEHOLDER"] = list(cavity_voxels.T[1])
    placeholders["POCKETS_Z_PLACEHOLDER"] = list(cavity_voxels.T[2])
    placeholders["POCKETS_S_PLACEHOLDER"] = grid_size

    placeholders["ATOMS_X_PLACEHOLDER"] = spheres.x.tolist()
    placeholders["ATOMS_Y_PLACEHOLDER"] = spheres.y.tolist()
    placeholders["ATOMS_Z_PLACEHOLDER"] = spheres.z.tolist()
    placeholders["ATOMS_r_PLACEHOLDER"] = (spheres.radius).tolist()

    placeholders["ATOMS_R_PLACEHOLDER"] = spheres_colors[0].tolist()
    placeholders["ATOMS_G_PLACEHOLDER"] = spheres_colors[1].tolist()
    placeholders["ATOMS_B_PLACEHOLDER"] = spheres_colors[2].tolist()

    placeholders["CAMERA_X_PLACEHOLDER"] = camera_position[0]
    placeholders["CAMERA_Y_PLACEHOLDER"] = camera_position[1]
    placeholders["CAMERA_Z_PLACEHOLDER"] = camera_position[2]

    placeholders["LOOK_X_PLACEHOLDER"] = centroid[0]
    placeholders["LOOK_Y_PLACEHOLDER"] = centroid[1]
    placeholders["LOOK_Z_PLACEHOLDER"] = centroid[2]

    placeholders["AXIS_SIZE_PLACEHOLDER"] = spheres[['x', 'y', 'z']].to_numpy().ptp() * 1.5

    placeholders["N_POCKETS_PLACEHOLDER"] = len(cavity_voxels)
    placeholders["N_VOXELS_PLACEHOLDER"] = spheres.shape[0]

    output_name = "plot.html"

    src = "{}".format("lib/template.html")
    dst = "{}".format(output_name)
    with open(src, "r") as inp, open(dst, "w") as out:
        for line in inp:
            for key, val in placeholders.items():
                if key in line:
                    line = line.replace(key, str(val))
            out.write(line)

    print('open in web browser:\n{}/{}'.format(getcwd(), output_name))

# def generate_mesh(xyzs, normals, centroid=np.array([0, 0, 0])):
#     pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyzs))
#     pcd.normals = o3d.utility.Vector3dVector(normals)
#     pcd.estimate_normals()
#     mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd)
#     return pcd, mesh[0]
