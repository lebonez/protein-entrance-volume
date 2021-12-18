"""
Created by: Mitchell Walls
Email: miwalls@siue.edu
"""
from protein_entrance_volume import utils


def sphere(centroid, coords, distance):
    """
    Generates a sphere using the unit sphere from utils scaling to radius and
    moving to centroid. Radius is calculated using the average distance
    between the centroid of the spheres and the coordinates near the outer
    bounds of the sphere. Number of points is calculated using the radius
    described above and the ideal "distance" between points.
    """
    radius = utils.average_distance(centroid, coords)
    num_points = utils.sphere_num_points(radius, distance)
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
    half_sphere_points = sphere_points[utils.side_points(plane, sphere_points) != side_point]
    return half_sphere_points
