"""
Created by: Mitchell Walls
Email: miwalls@siue.edu
"""
import numpy as np
from protein_entrance_volume import utils


def sphere(distance, centroid, num_points=1000):
    # TODO: Need to determine the number of points by the distance
    unit_sphere = utils.generate_sphere_points(num_points=num_points)
    return unit_sphere * distance + centroid


def half_sphere(plane_coords, distance, centroid, opposing_point, num_points=1000):
    # TODO: Need to determine the number of points by the distance
    unit_sphere = utils.generate_sphere_points(num_points=num_points)

    plane = utils.best_fit_plane(plane_coords)
    side_point = utils.side_point(plane, opposing_point)
    sphere_points = unit_sphere * distance + centroid
    half_sphere_points = []
    for point in sphere_points:
        if utils.side_point(plane, point) != side_point:
            half_sphere_points.append(point)
    return np.array(half_sphere_points)
