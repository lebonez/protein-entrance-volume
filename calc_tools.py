"""
Just a bunch of numba compiled functions to speed up everything.
Created by: Mitchell Walls
Email: miwalls@siue.edu
"""
from numba import njit, prange
from numba.experimental import jitclass
from numba.typed import Dict, List
from numba import types
import numpy as np
from time import time_ns


@njit(parallel=True, nogil=True, cache=True)
def calculate_normals(borders, coords, eqn, components, grid):
    """
    Use SVD with cluster of borders to get best fit plane normal.
    This is too slow for large number of borders.
    """
    normals = np.empty((borders.shape[0],3))
    for i in prange(borders.shape[0]):
        index = borders[i]
        coord = coords[i]
        indices = index + eqn
        grid_indices = indices[grid[indices]]
        contained = np.where(grid_indices.reshape(grid_indices.size, 1) == borders)[1]
        near_coords = coords[contained]
        points_centered = (near_coords - coord)
        u = np.linalg.svd(points_centered.T.astype(np.float64))[0]
        normals[i] = u[:, 2]
        normals[i] /= np.linalg.norm(normals[i])
        point = (normals[i] * 2).astype(np.int64)
        check = coord + point
        for j in range(components.shape[0]):
            if (point == components[j]).all():
                break
        if grid[indices[j]]:
            normals[i] *= -1
    return normals


@njit(nogil=True, cache=True)
def generate_borders(index, eqn, secondary_eqn, grid):
    """
    Find border voxels along with a very rough approximated voxelized
    normal (not accurate at all). Helps with orienting estimated normals.
    """
    seen = Dict.empty(types.int64, types.b1)
    queue = Dict.empty(types.int64, types.b1)
    borders = Dict.empty(types.int64, types.int64)
    # Starting by finding a new border
    while True:
        check_indices = eqn + index
        check_indices = check_indices[grid[check_indices]]
        if check_indices.shape[0] < 6 and grid[index]:
            queue[index] = False
            seen[index] = True
            break
        else:
            index = check_indices[0]
    while queue:
        index = queue.popitem()[0]
        check_indices = secondary_eqn + index
        for i in check_indices:
            if i not in seen:
                adjacent_indices = eqn + i
                seen[i] = False
                border_indices = adjacent_indices[grid[adjacent_indices]]
                if border_indices.shape[0] < 6 and grid[i]:
                    for j in range(adjacent_indices.shape[0]):
                        if adjacent_indices[j] not in border_indices:
                            break
                    borders[i] = adjacent_indices[j]
                    queue[i] = False
    return np.array(list(borders.items()))


@njit(parallel=True, nogil=True, cache=True)
def probe_extend(borders, potentials, grid):
    """
    Extend grid borders by probe radius this is much faster than any set
    based while or for loop.
    """
    for i in prange(borders.shape[0]):
        grid[borders[i] + potentials] = True
    return grid


@njit(nogil=True, cache=True)
def process_components(indices, eqn, grid):
    """
    Process connected components marking if voxels are on border where
    number of empty voxels near it is not equal to 6.
    """
    # Can't use python sets even though they are really fast because they bug out
    # and typed sets are not implemented yet in Numba. For now I'm relying on
    # typed Dict it provides decent hash lookup speeds.
    seen = Dict.empty(types.int64, types.b1)
    queue = Dict.empty(types.int64, types.b1)
    limit = len(grid)
    was_out_of_bounds = False
    for n in indices:
        queue[n] = False
    while queue:
        index = queue.popitem()[0]
        indices = eqn + index
        if (indices > limit).any() or (indices < 0).any():
            was_out_of_bounds = True
            continue
        indices = indices[grid[indices]]
        seen[index] = indices.shape[0] == 6
        for i in indices:
            if i not in seen and i not in queue:
                queue[i] = False
    return was_out_of_bounds, np.array(list(seen)), np.array([k for k, v in seen.items() if not v])


@njit(nogil=True, cache=True)
def search_indices(new_indices, indices, seen, found, eqn, have_queue, out_of_bounds, grid):
    """
    Work through all possible components.
    """
    new_indices.pop()
    limit = len(grid)
    while indices:
        if have_queue:
            new_indices.update(indices)
            break
        index = indices.pop()
        seen.add(index)
        check = eqn + index
        for i in check:
            if i > limit or i < 0:
                out_of_bounds = True
                continue
            if i not in seen and i not in indices:
                new_indices.add(i)
        queue = np.array([q for q in check[grid[check]] if q not in found and q < limit and q > 0])
        if len(queue):
            have_queue = True
    return new_indices, queue, seen, found, have_queue, out_of_bounds


def connected_components(index, eqn, grid):
    """
    Finds the starting index then continues to the component search algorithm.
    """
    indices = set([index])
    seen = set([np.int64(-1)])
    found = set([np.int64(-1)])
    limit = len(grid) * 0.0005
    have_queue = False
    out_of_bounds = False
    max_len = 0
    while not out_of_bounds:
        new_indices = set([np.int64(0)])
        new_indices, queue, seen, found, have_queue, out_of_bounds = search_indices(new_indices, indices, seen, found, eqn, have_queue, out_of_bounds, grid)
        if have_queue and not out_of_bounds:
            was_out_of_bounds, nodes, borders = process_components(queue, eqn, grid)
            clen = len(nodes)
            if not was_out_of_bounds and clen > 100:
                cnodes = np.copy(nodes)
                cborders = np.copy(borders)
                break
            elif not was_out_of_bounds and clen > max_len:
                cnodes = np.copy(nodes)
                cborders = np.copy(borders)
                max_len = clen
            found.update(nodes)
            have_queue = False
        indices = new_indices
    return cnodes, cborders

@jitclass([('svoxel', types.int64[:]), ('radius', types.float64), ('grid', types.b1[:,:,:])])
class RasterizeSphere:
    """
    Voxelize the atom.
    """
    def __init__(self, svoxel, radius, grid):
        self.grid = grid
        self.svoxel = svoxel
        self.radius = radius
        R2 = np.floor(self.radius**2)
        zx = np.int64(np.floor(self.radius))
        x = 0
        while True:
            while (x)**2 + zx**2 > R2 and zx >= x:
                zx -= 1
            if zx < x:
                break
            z = zx
            y = 0
            while True:
                while x**2 + y**2 + z**2 > R2 and z >= x and z >= y:
                    z -= 1
                if z < x or z < y:
                    break
                self.fill_all(x, y, z)
                # Fill the inside as well.
                for nz in range(z):
                    self.fill_all(x, y, nz)
                y += 1
            x += 1

    def fill_signs(self, x, y, z):
        """
        Fill negatives for reflections the if statements ensures there are no
        duplicates.
        """
        self.grid[x + self.svoxel[0], y + self.svoxel[1], z + self.svoxel[2]] = False
        while True:
            z = -z
            if z >= 0:
                y = -y
                if y >= 0:
                    x = -x
                    if x >= 0:
                        break
            self.grid[x + self.svoxel[0], y + self.svoxel[1], z + self.svoxel[2]] = False

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
def parallel_spheres(svoxels, radii, grid):
    """
    Develope a 3-D grid of values determining whether it is inside an atom or not.
    Flatten it for parsing 1-D data for speed.
    """
    for i in prange(svoxels.shape[0]):
        RasterizeSphere(svoxels[i], radii[i], grid)
    return grid.flatten()


@njit(parallel=True, nogil=True, cache=True)
def mesh_volume(vertices, triangles):
    """
    NOTE: this is much faster then any other Open3D implementation.
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
