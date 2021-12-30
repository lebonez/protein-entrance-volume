"""
Created by: Mitchell Walls
Email: miwalls@siue.edu
"""
import numpy as np
from numba import njit, prange
from numba.typed import Dict
from numba import types
from protein_entrance_volume import rasterize
from protein_entrance_volume import exception


# Can't reference this type inside a numba compiled function.
int_array = types.int64[:]


# Set of components to generate an equation of directly adjacent coordinates.
components = np.array([[-1,  0,  0], [0, -1,  0], [0,  0, -1], [1, 0, 0],
                       [0, 1, 0], [0, 0, 1]])


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
    eqn = np.ravel_multi_index((starting_voxel + components).T, grid.shape) \
        - starting_index
    # Run the meat of the algorithm.
    was_out_of_bounds, nodes, borders = calculate_components(
        starting_index, eqn, grid.flatten(), border_only
    )
    # Out of bounds is bad probably means we were outside of the bounding
    # object.
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
    # What is the maximum possible grid point so we can check for out of
    # bounds.
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
            return (was_out_of_bounds, np.array(list(seen)),
                    np.array([k for k, v in seen.items() if not v]))

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
        # Didn't find what we needed time to add current indices to check
        # further.
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
                return (was_out_of_bounds, np.array(list(seen)),
                        np.array([k for k, v in seen.items() if not v]))

            # Get adjacents that are False on the grid.
            ies = ies[~grid[ies]]
            # Most cases border only is better but sometimes marking non-border
            # components as well can be useful.
            if border_only:
                # Ignore any points that are obviously not near or on the
                # border. Reduces this algorithms run time by ten times in most
                # cases.
                if ies.shape[0] == 6 and seen[index]:
                    continue
            # Add index to key queue if it hasn't been seen or already in the
            # queue. Also add the adjacent indices of the index to the value.
            if i not in seen and i not in queue:
                queue[i] = ies
    # Return all seen indices including ones not on the border.
    return (was_out_of_bounds, np.array(list(seen)),
            np.array([k for k, v in seen.items() if not v]))


@njit(parallel=True, nogil=True, cache=True)
def eqn_grid(nodes, eqn, grid):
    """
    This function takes a 1D boolean grid and applies an eqn to
    every node in parallel (haven't found a way to do this without numba that
    is as fast.)
    """
    for i in prange(nodes.shape[0]):
        grid[nodes[i] + eqn] = True
    return grid


class SAS:
    """
    Grid version of the SAS with the capability to translate back and forth
    between coordinate systems.
    """
    _volume = None
    _vertices = None

    def __init__(self, coords, radii, start, stop, direction, grid_size,
                 fill_inside=False):
        """
        Build the gridified version of the SAS with the capability to translate
        the surface coordinates back to the original coordinate system.
        """
        # Generate the SAS grid from spherical points and radii of the
        # entrance.
        self.grid = Grid.from_cartesian_spheres(
            coords, radii, grid_size=grid_size, fill_inside=fill_inside)

        self.grid_size = grid_size
        # Find an empty starting voxel near the tip of the outer residue
        # hemisphere. This is the best location since it is the actual
        # beginning point of the entrance. The algorithm loops the voxel while
        # incrementing the voxel in the opposite direction of the outer
        # hemisphere normal if and until it reaches the atom outer residue
        # centroid at which point it raises a voxel not found exception.
        self.starting_voxel = self.grid.find_empty_voxel(
            start, stop, direction)

        # Calculate the SAS volume nodes and SAS border nodes using
        # connected components.
        self.nodes, self.sas_nodes = connected_components(
            self.grid.grid, self.starting_voxel)

    @property
    def vertices(self):
        """
        The array of vertices given by the nodes on the surface of the SES.
        """
        if self._vertices is None:
            # Calculate center of voxels then scale and shift them back to the
            # original atom coordinates system.
            self._vertices = (
                (np.array(np.unravel_index(self.sas_nodes,
                 self.grid.shape)).T + 0.5) * self.grid_size +
                self.grid.zero_shift)
        return self._vertices

    @property
    def volume(self):
        """
        Returns the volume given by the number of true values in the grid
        multiplied by the grid size.
        """
        if self._volume is None:
            # Calculate the volume by counting true values in the above grid
            # and multiplying by grid size cubed.
            self._volume = (np.count_nonzero(self.grid.grid)
                            * (self.grid_size ** 3))
        return self._volume


class SES:
    """
    Calculates the SES from the SAS described above.
    """
    _volume = None
    _vertices = None

    def __init__(self, sas, extension):
        """
        Generate the SES by taking all of the SAS nodes and expanding them
        spherically by the extension then also adding in the original SAS
        nodes.
        """
        self.grid_size = sas.grid_size
        self.starting_voxel = sas.starting_voxel
        # A beginning first voxel for calculating the probe extended grid using
        # the first sas border node.
        voxel = np.array(np.unravel_index(sas.sas_nodes[0], sas.grid.shape))

        # Build the initial grid using the first sas node voxel coordinate
        # extended by the probe radius divided by the grid size.
        volume_grid = rasterize.sphere(
            voxel, extension / self.grid_size,
            np.zeros(sas.grid.shape, dtype=bool), fill_inside=False
        )

        # flatten array for the subsequent calculations.
        volume_grid = volume_grid.flatten()

        # Calculate the spherical 1D equation for the probe radius grid sphere
        eqn = np.argwhere(volume_grid).flatten() - sas.sas_nodes[0]

        # Apply all of the spheres at the remaining SAS border nodes using the
        # sphere equation above calculated using the first sas node voxel.
        volume_grid = eqn_grid(sas.sas_nodes[1:], eqn, volume_grid)

        # Set the SES nodes from the initial connected components run to true
        # this fills any remaining holes which is faster then any binary fill
        # method from other libraries.
        volume_grid[sas.nodes] = True

        # Build the SAS volume grid object which includes the center
        # (non-border) voxels as well
        # Note: it uses the previous grid zero shift attribute from the first
        # atom coordinate based grid.
        self.grid = Grid(
            volume_grid.reshape(sas.grid.shape),
            zero_shift=sas.grid.zero_shift, grid_size=self.grid_size
        )

    @property
    def vertices(self):
        """
        The array of vertices given by the nodes on the surface of the SES.
        """
        if self._vertices is None:
            # Need to use connected components to find the actual surface nodes
            # ignoring the inner nodes with border_only.
            _, ses_nodes = connected_components(
                np.invert(self.grid.grid), self.starting_voxel,
                border_only=True
            )
            # Calculate center of voxels then scale and shift them back to the
            # original atom coordinates system.
            self._vertices = ((
                np.array(np.unravel_index(ses_nodes,
                         self.grid.shape)).T + 0.5) * self.grid_size +
                self.grid.zero_shift)
        return self._vertices

    @property
    def volume(self):
        """
        Returns the volume given by the number of true values in the grid
        multiplied by the grid size.
        """
        if self._volume is None:
            # Calculate the volume by counting true values in the above grid
            # and multiplying by grid size cubed.
            self._volume = (np.count_nonzero(self.grid.grid)
                            * (self.grid_size ** 3))
        return self._volume


class Grid:
    """
    Class holds a three dimensional boolean grid with convenience functions.
    """
    _coords = None
    _flatten = None

    def __init__(self, grid, zero_shift=0, grid_size=1):
        """
        Store some required properties. Note that the class itself isn't the
        boolean grid but it is self.grid
        """
        self._grid = grid
        self._grid_size = grid_size
        self._shape = grid.shape
        self._zero_shift = zero_shift

    def __str__(self):
        """
        String function print the boolean grid
        """
        return self.grid

    @property
    def coords(self):
        """
        Convenience property to return cartesians where true values are at.
        """
        if self._coords is None:
            self._coords = np.argwhere(self.grid)
        return self._coords

    @property
    def grid(self):
        """
        Convenience property to reference grid this should never be changed
        """
        return self._grid

    @property
    def grid_size(self):
        """
        Convenience property to reference grid size
        """
        return self._grid_size

    @property
    def zero_shift(self):
        """
        Convenience property to reference how far the original coordinate
        system was shifted
        """
        return self._zero_shift

    @property
    def shape(self):
        """
        Convenience property to reference total grid dimension size.
        """
        return self._shape

    @property
    def flatten(self):
        """
        Convenience property to convert 3D grid to 1D grid
        c-ordered array.
        """
        if self._flatten is None:
            self._flatten = self._grid.flatten()
        return self._flatten

    def gridify_point(self, point):
        """
        Take a cartesian point and project it onto the grid. Then return the
        voxel coordinate.
        """
        return ((point - self.zero_shift) / self.grid_size).astype(np.int64)

    def find_empty_voxel(self, start, stop, direction):
        """
        Given a directional vector starting with an initial cartesian point
        iterate in the direction of the vector until an empty voxel is found.
        Stop is supplied in the event an empty voxel is not found. The
        iteration must stop before going too far so giving a stopping point is
        required.
        """
        # Gridify and convert to float for the addition and comparisons
        location = self.gridify_point(start).astype(np.float64)
        stop_voxel = self.gridify_point(stop).astype(np.float64)
        # Convert the above to int which basically rounds down to integer.
        voxel = location.astype(np.int64)
        while self.grid[voxel[0], voxel[1], voxel[2]]:
            # Iteration in the direction of direction
            location += direction

            # We are now too close to the stop voxel so we need to stop before
            # ending up outside of the grid itself with an index error.
            if np.linalg.norm(location - stop_voxel) < 2:
                raise exception.StartingVoxelNotFound
            voxel = location.astype(np.int64)
        return voxel

    @classmethod
    def from_cartesian_spheres(
        cls, coords, radii, grid_size=1, fill_inside=False
    ):
        """
        Generate the 3D boolean grid from a set of cartesian (non-integer)
        spheres.
        """
        max_radius = radii.max()

        # Shift coords to positive integer space required for grid.
        zero_shift = coords.min(axis=0) - max_radius * 2

        grid_coords = coords - zero_shift
        # What dimension should the grid be in order to contain all spheres
        # plus radii.
        limits = np.ceil(
            (grid_coords.max(axis=0) + max_radius * 2) / grid_size
        ).astype(np.int64)

        # Scale the radii and coords to grid_size
        grid_coords /= grid_size
        grid_radii = radii / grid_size

        # Rasterize spheres on the grid marking spherical surface points as
        # true.
        grid = rasterize.spheres(
            grid_coords.astype(np.int64), grid_radii,
            np.zeros(limits, dtype=bool), fill_inside=fill_inside
        )

        return cls(grid, zero_shift, grid_size)
