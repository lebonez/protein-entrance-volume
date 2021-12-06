"""
Created by: Mitchell Walls
Email: miwalls@siue.edu
"""
import numpy as np
from numba import njit, prange
from numba.typed import Dict
from numba import types


def side_point(plane, point):
    """
    Finds which side a point lies on a plane. Returns a -1 or 1.
    Plane should be [center, normal].
    """
    side_vector = point - plane[0]
    magnitude = np.linalg.norm(side_vector)
    unit_vector = side_vector / magnitude
    return int(np.sign(np.dot(plane[1], unit_vector)))


def best_fit_plane(points):
    """
    Find the best fit plane given a set of points.
    """
    centroid = np.mean(points, axis=0)
    points_centered = (points - centroid)
    u = np.linalg.svd(points_centered.T)[0]
    normal = u[:, 2]
    return centroid, normal


@njit(parallel=True, nogil=True, cache=True)
def mesh_area(vertices, triangles):
    """
    Given a triangle mesh with vertices and triangles we can
    calculate the area using formula from origin:
    V = 1/2 * |(v1 • v2) x (v1 • v3)|
    Then adding up result for every triangle we get the mesh surface area.
    """
    area = 0
    for i in prange(triangles.shape[0]):
        triangle = vertices[triangles[i]]
        area += np.linalg.norm(np.cross((triangle[0] - triangle[1]), (triangle[0] - triangle[2])))
    return area / 2.0


@njit(parallel=True, nogil=True, cache=True)
def mesh_volume(vertices, triangles):
    """
    Given a triangle mesh with vertices and triangles we can
    calculate the volume using formula from origin:
    V = 1 / 6 * (v1 x v2) • v3
    Then adding up result for every triangle we get absolute value of
    mesh volume.
    """
    volume = 0
    for i in prange(triangles.shape[0]):
        triangle = vertices[triangles[i]]
        volume += np.dot(np.cross(triangle[0], triangle[1]), triangle[2])
    return np.abs(volume / 6.0)


def closest_node(node, nodes):
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)


def round_decimal(n, down=False):
    """
    Rounds up (default) or down first nonzero decimal places.
    """
    if n == 0:
        return 0
    sgn = -1 if n < 0 else 1
    scale = int(-np.floor(np.log10(abs(n))))
    if scale <= 0:
        scale = 1
    factor = 10 ** scale
    if down:
        result = np.floor(abs(n)*factor)
    else:
        result = np.ceil(abs(n)*factor)
    return sgn * result / factor


def inside_mbr(coords, mins, maxes):
    """
    Returns indices of coords array that are inside a minimum bounding
    rectangle (MBR).
    """
    return ((coords > mins) & (coords < maxes)).all(axis=1)


def generate_sphere_points(num_points=100):
    """
    Generate some unit cartesian points on the surface of a sphere.
    """
    indices = np.arange(0, num_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / num_points)
    theta = np.pi * (1 + 5 ** 0.5) * indices

    x = (np.cos(theta) * np.sin(phi))
    y = (np.sin(theta) * np.sin(phi))
    z = (np.cos(phi))

    return np.transpose([x, y, z])
