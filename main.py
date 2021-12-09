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
from protein_entrance_volume import volume
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

    # What radius we need for the outer maximum boundary sphere
    mbr_distance = atoms_mbr[2] - args.probe_radius
    # Generate the boundary sphere coords
    b_num_points = utils.grid_num_points(mbr_distance, args.grid_size)
    b_max = boundary.sphere(mbr_distance, atoms.arc, num_points=b_num_points)
    if not args.no_outer:
        # Furthest coord for outer residue atoms
        o_furthest_coord = atoms.or_coords[utils.furthest_node(atoms.orc, atoms.or_coords)]
        # Distance of furthest coord to the orc mins probe_radius so that it is SAS.
        o_distance = np.linalg.norm(o_furthest_coord - atoms.orc) - args.probe_radius

        o_num_points = utils.grid_num_points(o_distance, args.grid_size)
        # Outer residue boundary half spheres.
        b_outer = boundary.half_sphere(atoms.or_coords, o_distance, atoms.orc, atoms.irc, num_points=o_num_points)
    if not args.no_inner:
        # Furthest coord for inner residue atoms
        i_furthest_coord = atoms.ir_coords[utils.furthest_node(atoms.irc, atoms.ir_coords)]
        # Distance of furthest coord to the irc mins probe_radius so that it is SAS.
        i_distance = np.linalg.norm(i_furthest_coord - atoms.irc) - args.probe_radius
        i_num_points = utils.grid_num_points(i_distance, args.grid_size)
        # Inner residue boundary half spheres.
        b_inner = boundary.half_sphere(atoms.ir_coords, i_distance, atoms.irc, atoms.orc, num_points=i_num_points)

    # Combine all of the boundary coords.
    b_points = np.append(np.append(b_max, b_outer, axis=0), b_inner, axis=0)
    # Calculate the entrance volume features.
    num_points = utils.grid_num_points(mbr_radii.max(), args.grid_size)
    sas = surface.SAS(mbr_coords, mbr_radii, boundary_points=b_points, num_points=num_points)

    # Using the boundary coordinates to find a good starting index
    plane = utils.best_fit_plane(atoms.or_coords)
    opposing_side = utils.side_point(plane, atoms.irc)
    if utils.side_point(plane, plane[1] + atoms.orc) == opposing_side:
        normal = plane[1] * -1
    starting_point = o_distance * normal + atoms.orc
    closest_index = utils.closest_node(starting_point, b_outer)

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
