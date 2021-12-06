"""
Created by: Mitchell Walls
Email: miwalls@siue.edu
"""
from numba.experimental import jitclass
from numba import njit, prange
from numba import types
import numpy as np


@jitclass([('svoxel', types.int64[:]), ('radius', types.float64), ('grid', types.b1[:,:,:])])
class Sphere:
    """
    Rasterized sphere on a boolean grid.
    """
    def __init__(self, svoxel, radius, grid):
        """
        Uses 2D mid-point circle algorithm modified for 3D spheres. This is the
        fastest way to gridify alot of spheres.
        """
        self.grid = grid
        self.svoxel = svoxel
        self.radius = radius
        R2 = np.floor(self.radius ** 2)
        zx = np.int64(np.floor(self.radius))
        x = 0
        while True:
            while x ** 2 + zx ** 2 > R2 and zx >= x:
                zx -= 1
            if zx < x:
                break
            z = zx
            y = 0
            while True:
                while x ** 2 + y ** 2 + z ** 2 > R2 and z >= x and z >= y:
                    z -= 1
                if z < x or z < y:
                    break
                self.fill_all(x, y, z)
                # Fill the inside as well.
                # FIXME: this should hopefully be not required in the future
                for nz in range(z):
                    self.fill_all(x, y, nz)
                y += 1
            x += 1

    def fill_signs(self, x, y, z):
        """
        Fill negatives for reflections the if statements ensures there are no
        duplicates.
        """
        self.grid[x + self.svoxel[0], y + self.svoxel[1], z + self.svoxel[2]] = True
        while True:
            z = -z
            if z >= 0:
                y = -y
                if y >= 0:
                    x = -x
                    if x >= 0:
                        break
            self.grid[x + self.svoxel[0], y + self.svoxel[1], z + self.svoxel[2]] = True

    def fill_all(self, x, y, z):
        """
        Fill all reflections.
        """
        self.fill_signs(x, y, z)
        if z > y:
            self.fill_signs(x, z, y)
        if z > x and z > y:
            self.fill_signs(z, y, x)


@njit(parallel=True, nogil=True, cache=True)
def spheres(coords, radii, grid):
    """
    Rasterize spheres in parallel wraps the jitclass above so that it can be
    cached and doesn't require recompilation everytime. Shared variable grid
    is changed without direct assignment in the class or this method.
    """
    for i in prange(coords.shape[0]):
        Sphere(coords[i], radii[i], grid)
    return grid
