#!/usr/bin/env python
"""
Created by: Mitchell Walls
Email: miwalls@siue.edu
"""
import argparse
import numpy as np
from skimage import measure
from time import time_ns
from protein_entrance_volume.atoms import Atoms
from protein_entrance_volume.grid import Grid
from protein_entrance_volume import boundary
from protein_entrance_volume import surface
from protein_entrance_volume import visualization
from protein_entrance_volume import utils


def parse_args():
    """
    Parse args
    """
    parser = argparse.ArgumentParser(description="Parse pdb file to get protein entrance volume determined by inner and outer residues.")
    parser.add_argument('-o', '--outer-residues', required=True, type=int, nargs='+', help="A list of three or more outer residues to define the initial entrance of the tunnel.")
    parser.add_argument('-i', '--inner-residues', required=True, type=int, nargs='+', help="A list of three or more inner residues to define the desired ending location in tunnel.")
    parser.add_argument('--no-outer', default=False, action='store_true', help="Don't use outer residues boundary this is helpful if the outer residues shift positions alot.")
    parser.add_argument('--no-inner', default=False, action='store_true', help="Don't use inner residues boundary this is helpful if the inner residues shift positions alot.")
    parser.add_argument('-r', '--probe-radius', default=1.4, type=float, help="Radius of the algorithm probe to define the inner surface of the cavity (default: %(default)s).")
    parser.add_argument('-g', '--grid-size', default=0.5, type=float, help="The size of the grid to use to calculate the cavity inner surface (default: %(default)s).")
    parser.add_argument('-f', '--pdb-file', required=True, type=str, help="Path to the PDB file.")
    parser.add_argument('-V', '--visualize', const='html', nargs='?', choices=('html', 'ply', 'xyz'), help="If specified, creates a visualization (default: html).")
    return parser.parse_args()


def main():
    """
    Main function to use the cli
    """
    start = time_ns()
    args = parse_args()
    if len(args.outer_residues) < 3 or len(args.inner_residues) < 3:
        msg = 'Options "--outer-residues" and "--inner-residues" must be >= 3'
        raise argparse.ArgumentTypeError(msg)

    atoms = Atoms.parse_atoms(
        args.pdb_file,
        outer_residues=args.outer_residues,
        inner_residues=args.inner_residues
    )

    # Reduce computation complexity for the grid by only using spheres within
    # the mbr calculated by the inner and outer residue atoms.
    # Manipulate the coords and radii to include some boundaries as well as
    # generate the actual object we are using for the grid. This includes
    # extending by the probe radius and creating faux boundary spheres.

    # TODO: This is results in incredibly slow later code mostly due to the
    # addition of roughly a thousand new spheres.
    # This could probably be optimized more.
    atoms_mbr = atoms.residues_mbr(extension=args.probe_radius)
    mbr_coords = atoms_mbr[0]
    # Increase radii by probe size for SAS calculation.
    mbr_radii = atoms_mbr[1] + args.probe_radius
    mbr_distance = atoms_mbr[2]

    # TODO: Don't append inside the class but only returns the new sphere then
    # append here. Also needs better documenting and commenting.
    boundary_points = boundary.Sphere(mbr_distance, atoms.arc, extension=args.probe_radius).coords

    # if not args.no_outer:
    #     closest_coord = atoms.or_coords[utils.closest_node(atoms.orc, atoms.or_coords)]
    #     distance = np.linalg.norm(closest_coord - atoms.orc)
    #     boundary_points.extend(boundary.HalfSphere(atoms.or_coords, distance, atoms.orc, atoms.irc, extension=args.probe_radius).coords)
    #
    # if not args.no_inner:
    #     closest_coord = atoms.ir_coords[utils.closest_node(atoms.irc, atoms.ir_coords)]
    #     distance = np.linalg.norm(closest_coord - atoms.irc)
    #     boundary_points.extend(boundary.HalfSphere(atoms.ir_coords, distance, atoms.irc, atoms.orc, extension=args.probe_radius).coords)
    # ENDTODO

    # Calculate the entrance volume features.
    sas = surface.Surface(mbr_coords, mbr_radii, centroid=atoms.orc, boundary_points=boundary_points, num_points=1000)

    # Expand sas calculated above by the probe sphere radius and fill the volume if there are any openings.
    grid = Grid.from_spheres(sas.surface_points, np.full(sas.surface_points.shape[0], args.probe_radius), grid_size=args.grid_size)
    # outside_point = np.abs(sas.surface_points.max(axis=0)) * 2
    # ses = surface.Surface(sas.surface_points, np.full(sas.surface_points.shape[0], args.probe_radius), outside_point, num_points=100)

    # Flatten array and get number of true values thezn multiple by grid volume.
    # print("Volume:", np.count_nonzero(grid.flatten) * (args.grid_size ** 3), "Å³")

    verts, faces, _, _ = measure.marching_cubes(grid.grid)
    # print(measure.mesh_surface_area(verts, faces) * (args.grid_size ** 2))
    # print("Mesh Area:", utils.mesh_area(verts, faces) * (args.grid_size ** 2), "Å²")
    print("Mesh Volume:", utils.mesh_volume(verts, faces) * (args.grid_size ** 3), "Å³")


    if args.visualize:
        if args.visualize == 'html':
            print(verts)

        elif args.visualize == 'ply':
            print("Verts:", verts, "Faces:", faces, sep="\n")
            # visualization.matplotlib_mesh(verts, faces, grid.shape)

        elif args.visualize == 'xyz':
            print(verts)

    end = time_ns() - start
    print("Took:", end*10**(-9), "s")

if __name__ == '__main__':
    main()
