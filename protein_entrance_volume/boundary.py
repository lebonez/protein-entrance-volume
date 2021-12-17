"""
Created by: Mitchell Walls
Email: miwalls@siue.edu
"""
import numpy as np
from protein_entrance_volume import utils


def sphere(radius, centroid, num_points=1000):
    """
    Generates a sphere using the unit sphere from utils scaling to radius and
    moving to centroid.
    """
    unit_sphere = utils.generate_sphere_points(num_points=num_points)
    return unit_sphere * radius + centroid


def half_sphere(plane_coords, radius, centroid, opposing_point, num_points=1000):
    """
    Generates a half sphere using the unit sphere from utils scaling to radius
    and moving to centroid. Then find a best fit plane given plane coords then
    return array of sphere points on the opposite side of opposing points.
    """
    unit_sphere = utils.generate_sphere_points(num_points=num_points)

    plane = utils.best_fit_plane(plane_coords)
    side_point = utils.side_point(plane, opposing_point)
    sphere_points = unit_sphere * radius + centroid
    half_sphere_points = []
    for point in sphere_points:
        if utils.side_point(plane, point) != side_point:
            half_sphere_points.append(point)
    return np.array(half_sphere_points)
