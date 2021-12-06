"""
Created by: Mitchell Walls
Email: miwalls@siue.edu
"""
from scipy.spatial import KDTree
from protein_entrance_volume import utils
import numpy as np


class Surface:
    """
    Surface class for generating the surface of the object
    Note that the centroid is used to (hopefully) determine the correct
    surface by being nearest to the desired surface.
    """
    _max_unit_distance = None

    def __init__(self, coords, radii, centroid, num_points=100):
        self._coords = coords
        self._radii = radii

        # Generate a unit sphere with evenly spaced points
        self._unit_sphere = utils.generate_sphere_points(num_points=num_points)

        # Build kdtree for spheres
        kdt = KDTree(self.coords, 10)

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

            # Get the surface points that are non-intersecting.
            available_coords = s_on_i[list(available_set)]
            if surface_points is None:
                surface_points = available_coords
            else:
                surface_points = np.append(surface_points, available_coords, axis=0)

        # FIXME: there is a chance we get the wrong surface so find a way to
        # verify if we have or not then loop through possible other starting
        # indexes.

        # Find the nearest surface point to our supplied centroid to orient
        # ourself in hopes of finding the correct surface in the next steps.
        starting_index = utils.closest_node(centroid, surface_points)

        # Build KDTree of surface points.
        kdt = KDTree(surface_points, 10)

        # Surface points very likely has multiple distinct surfaces so given a
        # starting index lets search "recursively" for connected points
        # within the calculated max unit sphere distance using the max radii.
        surface = set()
        queue = set([starting_index])
        while queue:
            i = queue.pop()
            surface.add(i)
            # Get points that are close enough to be considered adjacently
            # connected and queue them up if we haven't marked them as the
            # surface.
            for j in kdt.query_ball_point(surface_points[i], self.max_unit_distance):
                if j not in queue and j not in surface:
                    queue.add(j)

        # The surface set should contain our distinct surface so index the
        # surface points using it.
        self._surface_points = surface_points[list(surface)]

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
    def surface_points(self):
        """
        Convenience property to return an 2D array of coordinate pairs
        """
        return self._surface_points

    @property
    def max_unit_distance(self, scale=1.2):
        """
        Convenience property to return the maximum unit spherical distance
        between points on the unit sphere with a scale to catch slightly
        further points.
        """
        if self._max_unit_distance is None:
            # Calculate largest possible unit sphere
            max_unit_sphere = np.array(self._unit_sphere, copy=True) * self.radii.max()
            # Find closest coordinate index to the first coordinate
            max_closest_index = utils.closest_node(
                max_unit_sphere[0], max_unit_sphere[1:]
            )
            # Calculate distance betwee the closest and first coordinate
            max_unit_distance = np.linalg.norm(
                max_unit_sphere[max_closest_index] - max_unit_sphere[0]
            )
            # Round up to the first non-zero decimal and scale
            self._max_unit_distance = utils.round_decimal(max_unit_distance) * scale
        return self._max_unit_distance
