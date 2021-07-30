"""
Created by: Mitchell Walls
Email: miwalls@siue.edu
"""
from operator import itemgetter
from time import time_ns
from os import getcwd
import numpy as np
import matplotlib.cm
import calc_tools
import matplotlib.pyplot as plt


class Cavity:
    """
    Class is used mostly as a convenience rather than a necessity.
    """

    def __init__(self, spheres, probe_radius=1.4, grid_size=0.5, centroids=None, visualize=False):
        """
        Do all of the shifting of the spheres to make voxelizing possible
        """
        self.spheres = spheres.copy(deep=True)
        self.probe_radius = np.float64(probe_radius)
        self.radius_shift = self.spheres.radius.max() + 2 * self.probe_radius

        # Recent change to find spheres only within larger atoms border.
        self.boundary_maxes = ((self.spheres[self.spheres['id'] == 'boundary'][['x', 'y', 'z']].max().to_numpy()) + self.radius_shift).astype(np.int64)
        self.boundary_mins = (self.spheres[self.spheres['id'] == 'boundary'][['x', 'y', 'z']].min().to_numpy() - self.radius_shift).astype(np.int64)
        self.inbounds_spheres = self.spheres[(self.boundary_mins < self.spheres[['x','y','z']]).all(axis=1) & (self.boundary_maxes > self.spheres[['x','y','z']]).all(axis=1)].copy(deep=True)
        self.inbounds_spheres['probe_radius'] = self.inbounds_spheres.radius + self.probe_radius

        self.visualize = visualize
        self.grid_size = grid_size
        if probe_radius == np.float64(0):
            self.radius_shift = 2 * self.inbounds_spheres.radius.max()
        self.shift = abs(self.inbounds_spheres[['x', 'y', 'z']].min()) + self.radius_shift
        self.inbounds_spheres[['x', 'y', 'z']] += self.shift
        self.centroids = centroids + self.shift.to_numpy()
        self.generate_grid()

    @staticmethod
    def cartesian_product(*arrays):
        """
        Generates all combinations from n arrays to get cartesians.
        This is way faster than itertools.
        """
        la = len(arrays)
        dtype = np.result_type(*arrays)
        arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
        for i, a in enumerate(np.ix_(*arrays)):
            arr[...,i] = a
        return arr.reshape(-1, la)

    def generate_grid(self):
        """
        The heart of the class that generates all the requirements for
        voxelizing and returns coordinates in the original shifted coordinate
        system.
        """
        mins = np.array([0, 0, 0], dtype=int)
        maxes = (self.inbounds_spheres[['x', 'y', 'z']].max() + self.radius_shift).to_numpy()

        assert (mins == 0).all(), "Minimum coords should be [0 0 0] got {}.".format(mins)

        self.x_y_z = np.ceil(maxes / self.grid_size).astype(int)
        self.scale = self.x_y_z / maxes

        # Can't use np.prod because of 64bit overflow in case of small grid size
        self.n_grid = 1
        for i in self.x_y_z:
            self.n_grid *= int(i)
        self.voxel_size = np.prod(self.grid_size ** 3)

        self.cavity_center = np.sum(self.centroids, axis=0) / 2
        self.cavity_center_n = np.ravel_multi_index((self.cavity_center * self.scale).astype(int), self.x_y_z)

        self.centroid_voxels = (self.centroids * self.scale).astype(int)
        self.centroid_voxels_n = np.ravel_multi_index(self.centroid_voxels.T, self.x_y_z)

        components = list()
        for i in [-1, 1]:
            components.append(np.array([i, 0, 0]))
            components.append(np.array([0, i, 0]))
            components.append(np.array([0, 0, i]))

        self.eqn = np.ravel_multi_index(((self.cavity_center * self.scale).astype(int) + np.array(components)).T, self.x_y_z) - self.cavity_center_n

        # start = time_ns()
        self.calc_radii()
        scaled_coords = self.inbounds_spheres[['x', 'y', 'z', 'probe_radius']].to_numpy() / self.grid_size
        self.grid = calc_tools.parallel_spheres(scaled_coords[:,:3].astype(np.int64), scaled_coords[:,3], np.ones(self.x_y_z, dtype=bool))
        # print("made_the_grid:", (time_ns() - start) * 10 ** (-9))

        #start = time_ns()
        nodes, borders = calc_tools.connected_components(self.cavity_center_n, self.eqn, self.grid)
        #print("connected_components initial grid:", (time_ns() - start) * 10 ** (-9))

        self.grid_minimized = np.zeros(self.n_grid, dtype=bool)
        self.grid_minimized[nodes] = True
        self.grid_minimized = calc_tools.probe_extend(borders, self.probe_potentials, self.grid_minimized)
        # start = time_ns()
        self.volume = np.count_nonzero(self.grid_minimized) * self.voxel_size
        # print("probe_extend:", (time_ns() - start) * 10 ** (-9))

        if self.visualize:
            components = self.cartesian_product(*np.linspace(np.ones(3, dtype=int) * -1, np.ones(3, dtype=int), num=3).T).astype(int)
            secondary_eqn = np.ravel_multi_index(((self.cavity_center * self.scale).astype(int) + np.array(components)).T, self.x_y_z) - self.cavity_center_n
            #start = time_ns()
            index = borders[0]
            borders = calc_tools.generate_borders(index, self.eqn, secondary_eqn, self.grid_minimized)
            #borders = np.array([k for k in borders.items()])
            self.cavity_normals = np.array(np.unravel_index(borders[:,1], self.x_y_z)).T
            borders = borders[:,0]
            components = self.cartesian_product(*np.linspace(np.ones(3, dtype=int) * -3, np.ones(3, dtype=int) * 3, num=7).T).astype(int)
            secondary_eqn = np.ravel_multi_index(((self.cavity_center * self.scale).astype(int) + np.array(components)).T, self.x_y_z) - self.cavity_center_n

            #self.cavity_normals = calc_tools.calculate_normals(borders, np.array(np.unravel_index(borders, self.x_y_z)).T, secondary_eqn, np.array(components), self.grid_minimized)
            #print("connected_components probe extension:", (time_ns() - start) * 10 ** (-9))
            cs = np.array(np.unravel_index(borders, self.x_y_z)).T
            self.cavity_normals -= cs
            # cs = np.array([np.array(np.unravel_index(starting_index, self.x_y_z)).T, np.array([0,0,0])])
            self.cavity_voxels = (cs * self.grid_size) + self.grid_size / 2 - self.shift.to_numpy()
            self.cavity_center -= self.shift.to_numpy()

    def calc_radii(self):
        """
        Calculate raveled indices of probe radius for when the initial volume is expanded.
        """
        x_y_z = np.ceil(np.ceil(2 * self.probe_radius) * np.ones(3, dtype=int) / self.grid_size).astype(int)
        voxels = self.cartesian_product(*np.linspace(np.zeros(3, dtype=int), x_y_z - 1, num=x_y_z[0]).T).astype(int)
        centroid = np.int64(x_y_z / 2)
        centroid_index = np.ravel_multi_index(centroid, self.x_y_z)
        voxel_indices = np.ravel_multi_index(voxels.T, self.x_y_z)
        centered = voxels - centroid + 0.5
        self.probe_potentials = voxel_indices[np.linalg.norm(centered, axis=1) <= self.probe_radius / self.grid_size] - centroid_index
