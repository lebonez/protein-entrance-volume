"""
Created by: Mitchell Walls
Email: miwalls@siue.edu
"""
from scipy.spatial import KDTree
from protein_entrance_volume import utils
import numpy as np


class SAS:
    """
    Surface class for generating the surface of the object
    Note that the centroid is used to (hopefully) determine the correct
    surface by being nearest to the desired surface.
    """
    _surface_points = []

    def __init__(self, coords, radii, boundary_points=None, num_points=100):
        self._coords = coords
        self._radii = radii

        # Generate a unit sphere with evenly spaced points
        self._unit_sphere = utils.generate_sphere_points(num_points=num_points)

        # Build kdtree for spheres
        kdt = KDTree(self.coords, 10)

        # Build kdtree for boundary_points
        if boundary_points is not None:
            kdt_boundary = KDTree(boundary_points, 10)
            # Possible points of the boundary
            boundary_set = set(range(len(boundary_points)))

        # The limit of how far to look for spheres in the kdtree
        twice_maxradii = self.radii.max() * 2

        # Create a set representing indices of s_on_i
        points_set = set(range(num_points))


        # Start with None for the array
        surface_points = None

        # Loop through spheres and generate points that are on the surface of
        # spheres that are not intersecting.
        append_time = 0
        for i in range(self.coords.shape[0]):
            r_i = self.radii[i]
            c_i = self.coords[i]

            # Make spherical surface
            s_on_i = np.array(self._unit_sphere, copy=True) * r_i + c_i
            available_set = points_set.copy()

            # Build kdtree of spherical surface
            kdt_sphere = KDTree(s_on_i, 10)

            # Loop through all spheres (j) that are close enough to possibly
            # intersect with the sphere (i).
            nearby_spheres = kdt.query_ball_point(c_i, twice_maxradii)

            # Pre compute sphere distances
            s_distances = np.linalg.norm(self.coords[nearby_spheres] - c_i, axis=1)

            for k, j in enumerate(nearby_spheres):
                r_j = self.radii[j]
                c_j = self.coords[j]

                # Skip itself since that would generate nothing
                if i == j:
                    continue

                # Make sure the jth sphere is close enough to intersect with
                # the ith sphere.
                if s_distances[k] <= (r_i + r_j):
                    # Remove all intersecting ith sphere surface points.
                    available_set -= set(kdt_sphere.query_ball_point(c_j, r_j))

            self._surface_points.extend(s_on_i[list(available_set)])

            # Filter out any boundary points within the current sphere.
            if boundary_points is not None:
                boundary_set -= set(kdt_boundary.query_ball_point(c_i, r_i))

        if boundary_points is not None:
            self._boundary_points = np.array(boundary_points)[list(boundary_set)]
            
        # surface_points.extend(boundary_points)
        # FIXME: there is a chance we get the wrong surface so find a way to
        # verify if we have or not then loop through possible other starting
        # indexes.

        # The surface set should contain our distinct surface so index the
        # surface points using it.
        self._surface_points = np.array(self._surface_points)

    @property
    def coords(self):
        """
        Convenience property to return an 2D array of coordinate pairs
        """
        return self._coords

    @property
    def radii(self):
        """
        Convenience property to return an 1D array of radii pairs
        """
        return self._radii

    @property
    def boundary_points(self):
        """
        Convenience property to return an 2D array of boundary coord pairs
        """
        return self._boundary_points

    @property
    def surface_points(self):
        """
        Convenience property to return an 2D array of surface coord pairs
        """
        return self._surface_points
