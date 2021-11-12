#!/usr/bin/env python
"""
Created by: Mitchell Walls
Email: miwalls@siue.edu
"""
import argparse
from time import time_ns
from itertools import combinations
import warnings
import glob
from Bio.PDB import PDBParser
from Bio.PDB.ResidueDepth import _get_atom_radius
from Bio import BiopythonWarning
import pandas as pd
import numpy as np
from cavity import Cavity
import tempfile
import open3d as o3d
import visualization
import calc_tools
from numba.core.errors import NumbaPendingDeprecationWarning


warnings.simplefilter('ignore', category=BiopythonWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

def parse_args():
    "Parse args"
    parser = argparse.ArgumentParser(description="Parse pdb file to get tunnel volume")
    parser.add_argument('-o', '--outer-residues', required=True, type=int, nargs='+', help="A list of three or more outer residues to define the entrance of the tunnel.")
    parser.add_argument('-i', '--inner-residues', required=True, type=int, nargs='+', help="A list of three or more inner residues to define the end of the tunnel.")
    parser.add_argument('--no-outer', default=False, action='store_true', help="Don't use outer residues boundary this is helpful if the outer residues shift positions alot.")
    parser.add_argument('--no-inner', default=False, action='store_true', help="Don't use inner residues boundary this is helpful if the inner residues shift positions alot.")
    parser.add_argument('-r', '--probe-radius', default=1.4, type=float, help="Radius of the algorithm probe to define the inner surface of the cavity (default: %(default)s).")
    parser.add_argument('-g', '--grid-size', default=0.5, type=float, help="The size of the grid to use to calculate the cavity inner surface (default: %(default)s).")
    parser.add_argument('-f', '--pdb-file', required=True, type=str, help="Path to the PDB file.")
    parser.add_argument('-V', '--visualize', const='html', nargs='?', choices=('html', 'ply', 'xyz'), help="If specified, creates a visualization (default: html).")
    return parser.parse_args()

class Atoms:
    "Build a dataframe containing real atoms and faux atoms to define borders."
    def __init__(self, atoms, no_outer=False, no_inner=False):
        self.atoms = pd.DataFrame(atoms)
        self.no_outer = no_outer
        self.no_inner = no_inner

    @classmethod
    def parse_atoms(cls, pdb_file, structure_id="prot", PERMISSIVE=False, outer_residues=[], inner_residues=[], no_outer=False, no_inner=False):
        """
        Make a list of dicts describing details about every atom noting the
        coordinates, radius, residue id, and outer/inner status creating a
        dataframe.
        """
        pdb = PDBParser(PERMISSIVE=PERMISSIVE)
        structure = pdb.get_structure(structure_id, pdb_file)
        atoms = []
        for residue in structure[0].get_residues():
            residue_id = residue.get_full_id()[3][1]
            for atom in residue.get_atoms():
                x, y, z = [*atom.get_coord()]
                atoms.append(
                    dict(
                        id=atom.fullname,
                        x = x, y = y, z = z,
                        # Some atoms don't get a radius so it seems fine to set them to 1
                        radius=_get_atom_radius(atom) if _get_atom_radius(atom) > 1 else 1.0,
                        residue_id=residue_id,
                        outer_residue=residue_id in outer_residues,
                        inner_residue=residue_id in inner_residues,
                    )
                )
        return cls(atoms, no_outer=no_outer, no_inner=no_inner)

    @staticmethod
    def best_fit_plane(points):
        """
        Find the best fit plane given a set of points.
        """
        centroid = np.mean(points, axis=0)
        points_centered = (points - centroid)
        u = np.linalg.svd(points_centered.T)[0]
        normal = u[:, 2]
        return centroid, normal

    @staticmethod
    def side_point(plane, point):
        """
        Finds which side a point lies on a plane. Returns a -1 or 1.
        Plane should be [center, normal].
        """
        side_vector = point - plane[0]
        magnitude = np.linalg.norm(side_vector)
        unit_vector = side_vector / magnitude
        return int(np.sign(np.dot(plane[1], unit_vector)))

    @staticmethod
    def generate_sphere_points(num_pts=200):
        """
        Generate some unit cartesian points on the surface of a sphere
        """
        indices = np.arange(0, num_pts, dtype=float) + 0.5
        phi = np.arccos(1 - 2*indices/num_pts)
        theta = np.pi * (1 + 5**0.5) * indices
        x = (np.cos(theta) * np.sin(phi))
        y = (np.sin(theta) * np.sin(phi))
        z = (np.cos(phi))
        return np.transpose([x, y, z])

    def generate_faux_atoms(self, radius=1):
        """
        Create a dict of faux atoms to define the boundaries of the cavity
        """
        self.calc_faux_atoms()
        faux_atoms = []
        outer_points = []
        if not self.no_outer:
            for point in self.outer_spheres + self.outer_plane[0]:
                if self.side_point(self.outer_plane, point) != self.outer_side:
                    outer_points.append(point)
                    faux_atoms.append(
                        dict(
                            id='outer',
                            x=point[0], y=point[1], z=point[2],
                            radius=radius,
                            residue_id=-1,
                            outer_residue=False,
                            inner_residue=False,
                        )
                    )
        inner_points = []
        if not self.no_inner:
            for point in self.inner_spheres + self.inner_plane[0]:
                if self.side_point(self.inner_plane, point) != self.inner_side:
                    inner_points.append(point)
                    faux_atoms.append(
                        dict(
                            id='inner',
                            x=point[0], y=point[1], z=point[2],
                            radius=radius,
                            residue_id=-1,
                            outer_residue=False,
                            inner_residue=False,
                        )
                    )
        for point in self.boundary_spheres:
            faux_atoms.append(
                dict(
                    id='boundary',
                    x=point[0], y=point[1], z=point[2],
                    radius=radius,
                    residue_id=-1,
                    outer_residue=False,
                    inner_residue=False,
                )
            )
        return faux_atoms, [self.outer_plane[0], self.inner_plane[0]]

    def calc_faux_atoms(self):
        """
        Calculate the faux atoms properties by calculating the best fit plane
        of the boundary faux atoms and generating spherical points on the
        opposite side of the planes.
        """
        self.outer_plane = self.best_fit_plane(self.outer_atoms[['x','y','z']].to_numpy())
        self.inner_plane = self.best_fit_plane(self.inner_atoms[['x','y','z']].to_numpy())

        self.outer_side = self.side_point(self.outer_plane, self.inner_plane[0])
        self.inner_side = self.side_point(self.inner_plane, self.outer_plane[0])
        points = self.generate_sphere_points(1000)

        outer_mean = []
        self.outer_spheres = np.copy(points)
        for residue in self.outer_ids:
            outer_mean.append(self.outer_atoms[self.outer_atoms.residue_id==residue][['x', 'y', 'z']].mean().to_numpy())
        outer_distance = 0
        for i, mean in enumerate(combinations(outer_mean, 2)):
            outer_distance += np.linalg.norm(mean[0] - mean[1])
        outer_distance = outer_distance / (i + 1)
        self.outer_spheres *= outer_distance / 2

        inner_mean = []
        self.inner_spheres = np.copy(points)
        for residue in self.inner_ids:
            inner_mean.append(self.inner_atoms[self.inner_atoms.residue_id==residue][['x', 'y', 'z']].mean().to_numpy())
        inner_distance = 0
        for i, mean in enumerate(combinations(inner_mean, 2)):
            inner_distance += np.linalg.norm(mean[0] - mean[1])
        inner_distance = inner_distance / (i + 1)
        self.inner_spheres *= inner_distance / 2

        self.boundary_spheres = np.copy(points)
        midpoint = (self.inner_plane[0] + self.outer_plane[0]) / 2
        distance = inner_distance
        if outer_distance < inner_distance:
            distance = outer_distance
        distance = (np.linalg.norm(self.inner_plane[0] - self.outer_plane[0]) + distance) / 2
        self.boundary_spheres *= distance
        self.boundary_spheres += midpoint

    def generate_boundaries(self):
        """
        Generates the boundaries near the inner and outer residue atoms.
        """
        self.outer_atoms = self.atoms[self.atoms['outer_residue']]
        self.outer_ids = self.outer_atoms.residue_id.unique()
        self.inner_atoms = self.atoms[self.atoms['inner_residue']]
        self.inner_ids =  self.inner_atoms.residue_id.unique()

        return self.generate_faux_atoms()

    def cavity(self, probe_radius, grid_size, visualize=False):
        """
        Generates the structure of the cavity enclosed by boundaries generated
        using the provided residues.
        """
        faux_atoms, centroids = self.generate_boundaries()
        all_atoms = self.atoms.append(pd.DataFrame(faux_atoms))
        cavity = Cavity(all_atoms[['id', 'x', 'y', 'z', 'radius']], probe_radius, grid_size, centroids, visualize=visualize)
        return cavity


def main():
    "Main function to find cavity"
    args= parse_args()
    atoms = Atoms.parse_atoms(
        args.pdb_file,
        outer_residues=args.outer_residues,
        inner_residues=args.inner_residues,
        no_outer=args.no_outer, no_inner=args.no_inner
    )
    cavity = atoms.cavity(args.probe_radius, args.grid_size, visualize=args.visualize)

    if args.visualize:
        if args.visualize == 'html':
            visualization.generate_html(atoms.atoms, cavity.cavity_voxels, cavity.cavity_center, cavity.shift, cavity.grid_size)
        elif args.visualize == 'ply':
            visualization.generate_mesh(cavity.cavity_voxels, cavity.cavity_normals, cavity.cavity_center)
        elif args.visualize == 'xyz':
            visualization.generate_xyz(cavity.cavity_voxels)


    print(cavity.volume, 'Å³')


if __name__ == '__main__':
    main()
