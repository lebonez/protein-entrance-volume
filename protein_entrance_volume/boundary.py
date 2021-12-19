"""
Created by: Mitchell Walls
Email: miwalls@siue.edu
"""
from protein_entrance_volume import utils


class Hemisphere:
    """
    Generates and holds the properties of the hemisphere.
    """
    _tip = None
    _normal = None

    def __init__(self, centroid, coords, opposing_point, distance):
        """
        Generates a hemisphere using the unit sphere from utils scaling to
        radius and moving to centroid. Then find a best fit plane given plane
        coords then return array of sphere points on the opposite side of
        opposing point.
        """
        self._centroid = centroid
        self._radius = utils.average_distance(self._centroid, coords)
        num_points = utils.sphere_num_points(self._radius, distance)
        unit_sphere = utils.generate_sphere_points(num_points=num_points)

        self.plane = utils.best_fit_plane(coords)
        self.opposing_side = utils.side_point(self.plane, opposing_point)
        sphere_points = unit_sphere * self._radius + self._centroid
        self._coords = sphere_points[
            utils.side_points(self.plane, sphere_points) != self.opposing_side
        ]

    @property
    def centroid(self):
        """
        Convenience property to return the coordinates of the hemisphere
        """
        return self._centroid

    @property
    def coords(self):
        """
        Convenience property to return the coordinates of the hemisphere
        """
        return self._coords

    @property
    def radius(self):
        """
        Convenience property to return the radius of the hemisphere
        """
        return self._radius

    @property
    def normal(self):
        """
        Convenience property to return the normal of the plane making up the
        hemisphere. Property also makes sure the normal being referenced
        points towards the side of the "sphere" where the hemispherical points
        are located. Singular value decomposition is random in the
        side the normal points for the best fit plane.
        """
        if self._normal is None:
            self._normal = self.plane[1]
            # Make sure the normal is not pointing at the opposing side if it
            # is swap it.
            if utils.side_point(self.plane, self.plane[1] + self.centroid) \
                    == self.opposing_side:
                # Swap normal to point away from the opposing side if it wasn't
                # already.
                self._normal *= -1
        return self._normal

    @property
    def tip(self):
        """
        Convenience property to return the coordinates of the tip of the
        hemisphere which is basically the point on the surface from the
        centroid in the direction of the normal.
        """
        if self._tip is None:
            self._tip = (self.radius) * self.normal + self.centroid
        return self._tip


class Sphere:
    """
    Generates and holds the properties of the sphere.
    """
    def __init__(self, centroid, coords, distance):
        """
        Generates a sphere using the unit sphere from utils scaling to radius
        and moving to centroid. Radius is calculated using the average distance
        between the centroid of the spheres and the coordinates near the outer
        bounds of the sphere. Number of points is calculated using the radius
        described above and the ideal "distance" between points.
        """
        self._centroid = centroid
        # This should probably rethought
        self._radius = utils.average_distance(self._centroid, coords)
        num_points = utils.sphere_num_points(self._radius, distance)
        unit_sphere = utils.generate_sphere_points(num_points=num_points)
        self._coords = unit_sphere * self._radius + self._centroid

    @property
    def centroid(self):
        """
        Convenience property to return the centroid of the sphere
        """
        return self._centroid

    @property
    def coords(self):
        """
        Convenience property to return the coordinates of the sphere
        """
        return self._coords

    @property
    def radius(self):
        """
        Convenience property to return the radius of the sphere
        """
        return self._radius
