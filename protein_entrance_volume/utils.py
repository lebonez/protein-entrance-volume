"""
Created by: Mitchell Walls
Email: miwalls@siue.edu
"""
import numpy as np


def parse_frames(frames):
    """
    Parse frames from a string of comma separated or single (-) dash ranges or
    integer frames.
    """
    frame_results = []
    frames = frames.split(',')
    for frame in frames:
        if '-' in frame:
            frame_range = frame.split('-')
            if len(frame_range) > 2:
                raise ValueError(f"Frame ranges should be like (a-b) was "
                                 f"{frame}.")
            if int(frame_range[0]) >= int(frame_range[1]):
                raise ValueError("Frame range (a-b) 'a' must be less than "
                                 "'b'.")
            try:
                frame_results.extend(
                    range(int(frame_range[0]), int(frame_range[1]) + 1)
                )
            except ValueError as value_error:
                raise ValueError(
                    f"Frame range must both be integers was {frame}."
                ) from value_error
        else:
            try:
                frame_results.append(int(frame))
            except ValueError as value_error:
                raise ValueError(
                    f"Frame must be an integer was {frame}."
                ) from value_error
    return frame_results


def distance(first, second):
    """
    Return cartesian distance between two points.
    """
    return np.linalg.norm(first - second)


def average_distance(point, points):
    """
    Return average cartesian distance between point and points.
    """
    return np.linalg.norm(points - point, axis=1).mean()


def side_points(plane, points):
    """
    Finds which side a set of points lie on a plane. Returns a -1 or 1
    depending on the side the points are on.
    Plane should be [centroid, normal].
    """
    side_vectors = points - plane[0]
    magnitudes = np.linalg.norm(side_vectors, axis=1)
    unit_vectors = []
    for i in range(3):
        unit_vectors.append(np.divide(side_vectors.T[i], magnitudes))
    unit_vectors = np.array(unit_vectors).T
    signs = np.sign(np.dot(unit_vectors, plane[1].reshape((3, 1)))).flatten()
    return signs.astype(int)


def side_point(plane, point):
    """
    Finds which side a point lies on a plane. Returns a -1 or 1 depending on
    the side the point is on.
    Plane should be [centroid, normal].
    """
    side_vector = point - plane[0]
    magnitude = np.linalg.norm(side_vector)
    unit_vector = side_vector / magnitude
    return int(np.sign(np.dot(plane[1], unit_vector)))


def best_fit_plane(points):
    """
    Find the best fit plane given a set of points using singular value
    decomposition (SVD).
    """
    centroid = np.mean(points, axis=0)
    points_centered = (points - centroid)
    # Get left singular matrix of the SVD of points centered.
    left_matrix = np.linalg.svd(points_centered.T)[0]
    # Right most column of left singular matrix is normal of best fit plane.
    normal = left_matrix[:, 2]
    # Return centroid and the normal
    return centroid, normal


def furthest_node(point, points):
    """
    The furthest "points" index from point to an array of points
    """
    deltas = points - point
    dist = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmax(dist)


def closest_node(node, nodes):
    """
    The closest "points" index from point to an array of points
    """
    deltas = nodes - node
    dist = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist)


def round_decimal(number, down=False):
    """
    Rounds up (default) or down first nonzero decimal places.
    """
    if number == 0:
        return 0

    sign = -1 if number < 0 else 1
    # How many decimal places are zero
    scale = int(-np.floor(np.log10(abs(number))))
    if scale <= 0:
        scale = 1
    # What should we multiply the n value by in order to round up or down.
    factor = 10 ** scale
    if down:
        # Round up after scaling by factor
        result = np.floor(abs(number)*factor)
    else:
        # Round down after scaling by factor
        result = np.ceil(abs(number)*factor)
    # Return result scale back by factor and signed by sign.
    return sign * result / factor


def inside_mbr(coords, mins, maxes):
    """
    Returns indices of coords array that are inside a minimum bounding
    rectangle (MBR).
    """
    return ((coords > mins) & (coords < maxes)).all(axis=1)


def sphere_num_points(radius, dist):
    """
    Calculate the optimum number of points given the radius of sphere and the
    distance between the points on that sphere.
    """
    ratio = np.cos(4 * np.pi / (1 + np.sqrt(5)))
    radius2 = radius ** 2
    distance2 = dist ** 2
    # solve for N, d = r * sqrt((cos(b)*sqrt(6/N-9/N^2)-cos(a)*sqrt(
    # 2/N-1/N^2))^2+(sin(b)*sqrt(6/N-9/N^2)-sin(a)*sqrt(2/N-1/N^2))^2+4/N^2)
    # This is the approximation of the ugly equation above. 1.85 gives a very
    # close value to the true N compared to using a brute force while loop
    # method.
    return np.int64(-1.85 * (4 * np.sqrt(3) * radius2 * ratio - 2 * radius2) /
                    distance2)


def generate_sphere_points(num_points=100):
    """
    Generate some unit cartesian points on the surface of a sphere.
    """
    # Generate evenly space set of indices from 0 to number of points stepping
    # by one.
    indices = np.arange(0, num_points, dtype=float) + 0.5
    # Calculate all of phi values for every index.
    phi = np.arccos(1 - 2 * indices / num_points)
    # Golden ratio to generate evenly spaced points along a surface of
    # sphere.
    golden_ratio = (1 + 5 ** 0.5) / 2
    # Theta calculated using golden ratio and indices array.
    theta = 2 * np.pi * indices / golden_ratio

    # Calculate our x, y, and z using spherical coordinates from indices.
    x_coords = (np.cos(theta) * np.sin(phi))
    y_coords = (np.sin(theta) * np.sin(phi))
    z_coords = (np.cos(phi))

    # Return the coordinates are x, y, z pairs.
    return np.transpose([x_coords, y_coords, z_coords])
