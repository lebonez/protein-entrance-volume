"""
Created by: Mitchell Walls
Email: miwalls@siue.edu
"""
import numpy as np
from protein_entrance_volume import rasterize
from protein_entrance_volume import exception


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
