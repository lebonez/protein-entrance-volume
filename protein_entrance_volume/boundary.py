"""
Created by: Mitchell Walls
Email: miwalls@siue.edu
"""
import numpy as np
from protein_entrance_volume import utils


class Sphere:
    def __init__(self, distance, centroid, extension=0, num_points=20000):
        # TODO: Need to determine the number of points by the distance
        distance -= extension
        unit_sphere = utils.generate_sphere_points(num_points=num_points)
        self._coords = list(unit_sphere * distance + centroid)

    @property
    def coords(self):
        """
        Convenience property to return an 2D array of coordinate pairs
        """
        return self._coords


class HalfSphere:
    def __init__(self, plane_coords, distance, centroid, opposing_point, extension=0, num_points=1000):
        # TODO: Need to determine the number of points by the distance
        distance -= extension
        unit_sphere = utils.generate_sphere_points(num_points=num_points)

        plane = utils.best_fit_plane(plane_coords)
        side_point = utils.side_point(plane, opposing_point)

        sphere_points = unit_sphere * distance + centroid
        half_sphere_points = []
        for point in sphere_points:
            if utils.side_point(plane, point) != side_point:
                half_sphere_points.append(point)

        self._coords = half_sphere_points

    @property
    def coords(self):
        """
        Convenience property to return an 2D array of coordinate pairs
        """
        return self._coords
