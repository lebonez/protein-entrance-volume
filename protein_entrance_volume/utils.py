"""
Created by: Mitchell Walls
Email: miwalls@siue.edu
"""
import numpy as np
from numba import njit, prange
from numba.typed import Dict
from numba import types
from protein_entrance_volume import exception

# Can't reference this type inside a numba compiled function.
int_array = types.int64[:]

# Set of components to generate an equation of directly adjacent coordinates.
components = np.array(
    [[-1,  0,  0], [ 0, -1,  0], [ 0,  0, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
)

def connected_components(grid, starting_voxel, border_only=False):
    """
    Using components above we can calculate adjacent indices using the equation
    calculated by raveling the voxels and subtracting the starting voxel
    index. This drastically speeds up connected components by using 1D
    arrays.
    """
    # Find raveled 1D index of starting voxel coordinate
    starting_index = np.ravel_multi_index(starting_voxel, grid.shape)
    # Find equation of 1D indices of adjacent voxels.
    eqn = np.ravel_multi_index((starting_voxel + components).T, grid.shape) - starting_index
    # Run the meat of the algorithm.
    was_out_of_bounds, nodes, borders = calculate_components(starting_index, eqn, grid.flatten(), border_only)
    # Out of bounds is bad probably means we were outside of the bounding object.
    if was_out_of_bounds:
        raise exception.OutOfBounds
    return nodes, borders


@njit(nogil=True, cache=True)
def calculate_components(starting_index, eqn, grid, border_only):
    """
    Process connected components marking if grid points are on border where
    number of empty grid points near it is not equal to 6.
    """
    # Build a seen numba dict that allows us to mark indices that are true on
    # the border (False values)
    seen = Dict.empty(types.int64, types.b1)
    # Queue numba dict for tracking indices that need checked and store their
    # adjacent indices as the value.
    queue = Dict.empty(types.int64, int_array)
    # Dict to hold indices to check in the first while loop.
    check = Dict.empty(types.int64, int_array)
    # What is the maximum possible grid point so we can check for out of bounds.
    limit = len(grid) - 1
    was_out_of_bounds = False

    check[starting_index] = eqn + starting_index
    # Build starting index's indices queue and make sure the starting index is
    # valid and at the border if border only.
    while True:
        # Get adjacent indices using equation.
        index, indices = check.popitem()
        # Check if out of bounds where indices can never be less than zero or
        # greater than limit.
        if (indices > limit).any() or (indices < 0).any():
            return was_out_of_bounds, np.array(list(seen)), np.array([k for k, v in seen.items() if not v])
        # Filter the indices to include only ones that are False on the boolean
        # grid.
        ies = indices[~grid[indices]]
        # Don't need to be border starting index but make sure it has at least
        # one False adjacent index.
        if not border_only and ies.shape[0] > 0:
            queue[index] = ies
            break
        # Need to be border starting index and make sure it has at least
        # one False adjacent index.
        if ies.shape[0] != 6 and ies.shape[0] > 0:
            queue[index] = ies
            break
        # Didn't find what we needed time to add current indices to check further.
        for i in indices:
            check[i] = eqn + i

    while queue:
        # Remove item from queue assigning the key and value as below.
        index, indices = queue.popitem()
        # Add index to seen and mark if border or not where equal to six and is
        # not a border index. Also index should always be False on the grid.
        seen[index] = indices.shape[0] == 6
        # Loop through all of the False grid indices adjacent to index.
        for i in indices:
            # Get adjacent indices of the current i index.
            ies = eqn + i
            # Check out of bounds again because that is bad if it happens.
            if (ies > limit).any() or (ies < 0).any():
                was_out_of_bounds = True
                return was_out_of_bounds, np.array(list(seen)), np.array([k for k, v in seen.items() if not v])
            # Get adjacents that are False on the grid.
            ies = ies[~grid[ies]]
            # Most cases border only is better but sometimes marking non-border
            # components as well can be useful.
            if border_only:
                # Ignore any points that are obviously not near or on the border
                # Reduces this algorithms run time by ten times in most cases.
                if ies.shape[0] == 6 and seen[index]:
                    continue
            # Add index to key queue if it hasn't been seen or already in the
            # queue. Also add the adjacent indices of the index to the value.
            if i not in seen and i not in queue:
                queue[i] = ies
    # Border only just returns indices if the value of indices shape was not
    # six or on the border.
    if border_only:
        return was_out_of_bounds, np.array(list(seen)), np.array([k for k, v in seen.items() if not v])
    # Return all seen indices including ones not on the border.
    return was_out_of_bounds, np.array(list(seen)), np.array([k for k, v in seen.items() if not v])


@njit(parallel=True, nogil=True, cache=True)
def eqn_grid(nodes, eqn, grid):
    """
    This function takes a 1D boolean grid applies an eqn to
    every node in parallel (haven't found a way to do this without numba that
    is as fast.)
    """
    for i in prange(nodes.shape[0]):
        grid[nodes[i] + eqn] = True
    return grid


def average_distance(point, points):
    """
    Return average cartesian distance between point and points.
    """
    return np.linalg.norm(points - point, axis=1).mean()


def side_point(plane, point):
    """
    Finds which side a point lies on a plane. Returns a -1 or 1.
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


def round_decimal(n, down=False):
    """
    Rounds up (default) or down first nonzero decimal places.
    """
    if n == 0:
        return 0

    sign = -1 if n < 0 else 1
    # How many decimal places are zero
    scale = int(-np.floor(np.log10(abs(n))))
    if scale <= 0:
        scale = 1
    # What should we multiply the n value by in order to round up or down.
    factor = 10 ** scale
    if down:
        # Round up after scaling by factor
        result = np.floor(abs(n)*factor)
    else:
        # Round down after scaling by factor
        result = np.ceil(abs(n)*factor)
    # Return result scale back by factor and signed by sign.
    return sign * result / factor


def inside_mbr(coords, mins, maxes):
    """
    Returns indices of coords array that are inside a minimum bounding
    rectangle (MBR).
    """
    return ((coords > mins) & (coords < maxes)).all(axis=1)


def sphere_num_points(radius, distance):
    """
    Calculate the optimum number of points given the radius of sphere and the
    distance between the points on that sphere.
    """
    golden_ratio = 4 * np.pi / (1 + np.sqrt(5))
    r2 = radius ** 2
    d2 = distance ** 2
    # solve for N, d = r * sqrt((cos(b)*sqrt(6/N-9/N^2)-cos(a)*sqrt(2/N-1/N^2))^2+(sin(b)*sqrt(6/N-9/N^2)-sin(a)*sqrt(2/N-1/N^2))^2+4/N^2)
    # This is the approximation of the ugly equation above. 1.85 gives a very
    # close value to the true N compared to using a brute force while loop method.
    return np.int64(-1.85 * (4 * np.sqrt(3) * r2 * np.cos(golden_ratio) - 2 * r2) / d2)


def generate_sphere_points(num_points=100):
    """
    Generate some unit cartesian points on the surface of a sphere.
    """
    # Generate evenly space set of indices from 0 to number of points stepping
    # by one.
    indices = np.arange(0, num_points, dtype=float) + 0.5
    # Calculate all of phi values for every index.
    phi = np.arccos(1 - 2 * indices / num_points)
    # Golden ratio to generate evenly spaced points in a along surface of
    # sphere.
    golden_ratio = (1 + 5 ** 0.5) / 2
    # Theta calculated using golden ratio and indices array.
    theta = 2 * np.pi * indices / golden_ratio

    # Calculate our x, y, and z using spherical coordinates from indices.
    x = (np.cos(theta) * np.sin(phi))
    y = (np.sin(theta) * np.sin(phi))
    z = (np.cos(phi))

    # Return the coordinates are x, y, z pairs.
    return np.transpose([x, y, z])
