"""
Created by: Mitchell Walls
Email: miwalls@siue.edu
"""
from numba.experimental import jitclass
from numba import njit, prange
from numba import types
import numpy as np


@jitclass([('svoxel', types.int64[:]), ('radius', types.float64),
           ('grid', types.b1[:, :, :])])
class Sphere:
    """
    Rasterized sphere on a boolean grid.
    """
    def __init__(self, svoxel, radius, grid, fill_inside=False):
        """
        Uses 2D mid-point circle algorithm modified for 3D spheres. This is the
        fastest way to gridify alot of spheres.
        """
        self.grid = grid
        self.svoxel = svoxel
        self.radius = radius
        radius2 = np.floor(self.radius ** 2)
        maxz_x = np.int64(np.floor(self.radius))
        x_coord = 0
        while True:
            while x_coord ** 2 + maxz_x ** 2 > radius2 and maxz_x >= x_coord:
                maxz_x -= 1
            if maxz_x < x_coord:
                break
            z_coord = maxz_x
            y_coord = 0
            while True:
                while x_coord ** 2 + y_coord ** 2 + z_coord ** 2 > radius2 \
                        and z_coord >= x_coord and z_coord >= y_coord:
                    z_coord -= 1
                if z_coord < x_coord or z_coord < y_coord:
                    break
                self.fill_all(x_coord, y_coord, z_coord)
                if fill_inside:
                    for nz_coord in range(z_coord):
                        self.fill_all(x_coord, y_coord, nz_coord)
                y_coord += 1
            x_coord += 1

    def fill_signs(self, x_coord, y_coord, z_coord):
        """
        Fill negatives for reflections the if statements ensures there are no
        duplicates.
        """
        self.grid[x_coord + self.svoxel[0], y_coord + self.svoxel[1],
                  z_coord + self.svoxel[2]] = True
        while True:
            z_coord = -z_coord
            if z_coord >= 0:
                y_coord = -y_coord
                if y_coord >= 0:
                    x_coord = -x_coord
                    if x_coord >= 0:
                        break
            self.grid[x_coord + self.svoxel[0], y_coord + self.svoxel[1],
                      z_coord + self.svoxel[2]] = True

    def fill_all(self, x_coord, y_coord, z_coord):
        """
        Fill all reflections.
        """
        self.fill_signs(x_coord, y_coord, z_coord)
        if z_coord > y_coord:
            self.fill_signs(x_coord, z_coord, y_coord)
        if z_coord > x_coord and z_coord > y_coord:
            self.fill_signs(z_coord, y_coord, x_coord)


@njit(nogil=True, cache=True)
def sphere(coord, radius, grid, fill_inside=False):
    """
    Rasterize a sphere on the grid same as spheres below but just for one
    sphere.
    """
    Sphere(coord, radius, grid, fill_inside)
    return grid


@njit(parallel=True, nogil=True, cache=True)
def spheres(coords, radii, grid, fill_inside=False):
    """
    Rasterize spheres in parallel wraps the jitclass above so that it can be
    cached and doesn't require recompilation everytime. Shared variable grid
    is changed without direct assignment in the class or this method.
    """
    for i in prange(coords.shape[0]):
        Sphere(coords[i], radii[i], grid, fill_inside)
    return grid
