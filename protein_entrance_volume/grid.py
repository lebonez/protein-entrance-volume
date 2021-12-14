"""
Created by: Mitchell Walls
Email: miwalls@siue.edu
"""
import numpy as np
from protein_entrance_volume import rasterize


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

    @classmethod
    def from_spheres(cls, coords, radii, grid_size=1, fill_inside=False):
        """
        Generate the 3D boolean grid from set of spheres.
        """
        max_radius = radii.max()

        # Shift coords to positive integer space required for grid.
        zero_shift = coords.min(axis=0) - max_radius * 2
        coords -= zero_shift

        # What dimension should the grid be in order to contain all spheres plus radii.
        limits = np.ceil((coords.max(axis=0) + max_radius * 2) / grid_size).astype(np.int64)

        # Scale the radii and coords to grid_size
        coords /= grid_size
        radii /= grid_size

        # Rasterize spheres on the grid marking spherical surface points as true.
        grid = rasterize.spheres(coords.astype(np.int64), radii, np.zeros(limits, dtype=bool), fill_inside=fill_inside)

        return cls(grid, zero_shift, grid_size)
