"""
Created by: Mitchell Walls
Email: miwalls@siue.edu
"""
import numpy as np
from protein_entrance_volume import utils


class Spheres:

    def __init__(self, coords, radii, distance, centroid, extension=1):
        # TODO: Need to determine the number of points by the distance
        num_points = 1000
        unit_sphere = utils.generate_sphere_points(num_points=num_points)
        self._coords = np.append(coords, unit_sphere * distance + centroid, axis=0)

        # Extend boundary spheres radii by extension and make a boundary fake
        # sphere boundary with extension radius
        self._radii = np.append(radii, np.full(num_points, extension))

    @property
    def coords(self):
        """
        Convenience property to return an 2D array of coordinate pairs
        """
        return self._coords

    @property
    def radii(self):
        """
        Convenience property to return an 1D array of atom radii
        """
        return self._radii


class HalfSpheres:
    def __init__(self, plane_coords, coords, radii, distance, centroid, opposing_point, extension=0):
        # TODO: Need to determine the number of points by the distance
        num_points = 1000
        unit_sphere = utils.generate_sphere_points(num_points=num_points)

        plane = utils.best_fit_plane(plane_coords)
        side_point = utils.side_point(plane, opposing_point)

        sphere_points = unit_sphere * distance + centroid
        half_sphere_points = []
        for point in sphere_points:
            if utils.side_point(plane, point) != side_point:
                half_sphere_points.append(point)

        self._coords = np.append(coords, half_sphere_points, axis=0)
        self._radii = np.append(radii, np.full(num_points, extension))

    @property
    def coords(self):
        """
        Convenience property to return an 2D array of coordinate pairs
        """
        return self._coords

    @property
    def radii(self):
        """
        Convenience property to return an 1D array of atom radii
        """
        return self._radii
