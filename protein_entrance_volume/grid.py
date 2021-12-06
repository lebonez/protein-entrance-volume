"""
Created by: Mitchell Walls
Email: miwalls@siue.edu
"""
import numpy as np
from scipy import ndimage
from protein_entrance_volume import rasterize


class Grid:
    """
    Class holds a three dimensional boolean grid with convenience functions.
    TODO: Add more from_{shape} classmethod using rasterize.{shape} files.
    """
    _coords = None
    _flatten = None

    def __init__(self, grid, zero_shift=0):
        self._grid = grid
        self._shape = grid.shape
        self._zero_shift = zero_shift

    def __str__(self):
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

    @classmethod
    def from_spheres(cls, coords, radii, grid_size=1, fill=False):
        """
        Generate the 3D boolean grid from set of generic spheres.
        """
        max_radius = radii.max()

        # Shift coords to positive integer space required for grid.
        zero_shift = coords.min(axis=0) - max_radius * 2
        coords -= zero_shift

        # What dimension should the grid be in order to contain all spheres.
        limits = np.ceil((coords.max(axis=0) + max_radius * 2) / grid_size).astype(np.int64)

        # Scale the radii and coords to grid_size
        coords /= grid_size
        radii /= grid_size

        grid = rasterize.spheres(coords.astype(np.int64), radii, np.zeros(limits, dtype=bool))

        if fill:
            # FIXME: Probably a faster way to do this but cleanest is a library
            # like scipy.
            ndimage.binary_fill_holes(grid, output=grid)
        return cls(grid, zero_shift)
