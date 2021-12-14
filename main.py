#!/usr/bin/env python
"""
Created by: Mitchell Walls
Email: miwalls@siue.edu
"""
import argparse
from time import time_ns
import numpy as np
from protein_entrance_volume.atoms import Atoms
from protein_entrance_volume.grid import Grid
from protein_entrance_volume import boundary
from protein_entrance_volume import rasterize
from protein_entrance_volume import utils
from protein_entrance_volume import exception
from protein_entrance_volume import mesh


def parse_args():
    """
    Parse args
    """
    parser = argparse.ArgumentParser(description="Parse pdb file to get protein entrance volume determined by inner and outer residues.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-o', '--outer-residues', required=True, type=int, nargs='+', help="A list of three or more outer residues to define the initial entrance of the tunnel.")
    parser.add_argument('-i', '--inner-residues', required=True, type=int, nargs='+', help="A list of three or more inner residues to define the desired ending location in tunnel.")
    parser.add_argument('--no-outer', default=False, action='store_true', help="Don't use outer residues boundary this is helpful if the outer residues shift positions alot.")
    parser.add_argument('--no-inner', default=False, action='store_true', help="Don't use inner residues boundary this is helpful if the inner residues shift positions alot.")
    parser.add_argument('-r', '--probe-radius', default=1.4, type=float, help="Radius of the algorithm probe to define the inner surface of the cavity (default: %(default)s).")
    parser.add_argument('-g', '--grid-size', default=0.5, type=float, help="The size of the grid to use to calculate the cavity inner surface (default: %(default)s).")
    parser.add_argument('-R', '--resolution', default=4, type=float, help="Lower values decreases runtime and higher values for accuracy default: %(default)s).")
    parser.add_argument('-f', '--pdb-file', required=True, type=str, help="Path to the PDB file.")
    parser.add_argument('-v', '--vertices-file', default="", type=str, help="""
Output the vertices to file which file types depends on the file extension provided in this argument.
    xyz: Outputs the vertices as a molecular xyz file with each vertices marked as an "x" atom and has volume as the comment line after number of atoms.
    csv: Vertices array is dumped to a file with "x,y,z" as header and each line containing a comma separated x,y,z coordinate.
    txt: Vertices array is dumped to a txt file with first line containing volume of vertices and x y z coordinates space separated.
    npz: Recommended if loading the vertices array back into numpy uses much less space and is faster.
    """)
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

    # Build the atoms dataframe object.
    atoms = Atoms.parse_atoms(
        args.pdb_file,
        outer_residues=args.outer_residues,
        inner_residues=args.inner_residues
    )

    # This value is essentially used to scale the number of points for boundary
    # spheres so that they are evenly space by the probe radius divided by
    # resolution.
    resolution = args.resolution
    # Reduce computation complexity for the grid by only using spheres within
    # the mbr calculated by the inner and outer residue atoms.
    atoms_mbr = atoms.residues_mbr(extension=args.probe_radius)
    mbr_coords = atoms_mbr[0]

    # Increase radii by probe size for SAS calculation.
    mbr_radii = atoms_mbr[1] + args.probe_radius

    b_distance = utils.average_distance(atoms.arc, mbr_coords)
    b_num_points = utils.sphere_num_points(b_distance, args.probe_radius / resolution)

    # Generate the boundary sphere coords
    b_max = boundary.sphere(b_distance, atoms.arc, num_points=b_num_points)
    b_points = np.copy(b_max)
    o_distance = utils.average_distance(atoms.orc, atoms.or_coords)
    i_distance = utils.average_distance(atoms.irc, atoms.ir_coords)
    if not args.no_outer:
        o_num_points = utils.sphere_num_points(o_distance, args.probe_radius / resolution)
        # Outer residue boundary half spheres.
        b_outer = boundary.half_sphere(atoms.or_coords, o_distance, atoms.orc, atoms.irc, num_points=o_num_points)
        b_points = np.append(b_points, b_outer, axis=0)
    if not args.no_inner:
        i_num_points = utils.sphere_num_points(i_distance, args.probe_radius / resolution)
        # Inner residue boundary half spheres.
        b_inner = boundary.half_sphere(atoms.ir_coords, i_distance, atoms.irc, atoms.orc, num_points=i_num_points)
        b_points = np.append(b_points, b_inner, axis=0)

    # Build a radii array that has same length as boundary points.
    b_radii = np.full(b_points.shape[0], args.probe_radius)
    # Combine all of the boundary coords with the mbr.
    coords = np.append(mbr_coords, b_points, axis=0)
    radii = np.append(mbr_radii, b_radii, axis=0)

    # Find a starting point for the surface search that is on the center of the
    # surface of the outer residue half sphere..
    plane = utils.best_fit_plane(atoms.or_coords)
    opposing_side = utils.side_point(plane, atoms.irc)
    if utils.side_point(plane, plane[1] + atoms.orc) == opposing_side:
        # Swap normal to point away from the orc if it wasn't already.
        normal = plane[1] * -1
    distance = o_distance
    starting_point = (distance - args.probe_radius) * normal + atoms.orc

    # Generate the SAS grid from spherical points and radii within the MBR.
    grid = Grid.from_spheres(coords, radii, grid_size=args.grid_size)

    # Find a good starting voxel to do volume search by starting at the starting
    # point that was calculated above and moving towards the orc.
    starting_voxel = grid.gridify_point(starting_point)
    i = 1
    # Swap the normal back to point towards the orc opposite of starting point.
    normal *= -1
    # Loop through and add ith iteration times the normal until we get an empty
    # (outside of spheres) voxel.
    while grid.grid[starting_voxel[0], starting_voxel[1], starting_voxel[2]]:
        point = starting_point + i * normal
        if np.linalg.norm(point - atoms.orc) < 1:
            raise exception.StartingVoxelNotFound
        starting_voxel = grid.gridify_point(point)
        i += 1

    # Store this starting voxel for vertices file generation
    stored_starting_voxel = starting_voxel.copy()

    # Calculate the SAS using connected_components
    sas_nodes = utils.connected_components(grid.grid, starting_voxel, border_only=True)
    # Convert SAS raveled indices to coordinates
    border_points = np.array(np.unravel_index(sas_nodes, grid.shape)).T
    # Generate a new spherical grid for volume calculation using SAS coordinates.
    volume = rasterize.spheres(border_points, np.full(border_points.shape[0], args.probe_radius) / args.grid_size, np.zeros(grid.shape, dtype=bool), fill_inside=False)

    # Find a good starting voxel to do the filling of the hole inside of the volume calculated above.
    starting_voxel = grid.gridify_point(atoms.orc)
    i = 1
    # swap the normal from above back to point away from the orc.
    normal *= -1
    # Loop through and add ith iteration times the normal until we get an empty
    # voxel.
    while volume[starting_voxel[0], starting_voxel[1], starting_voxel[2]]:
        point = starting_point + i * normal
        if np.linalg.norm(point - atoms.orc) < 1:
            starting_voxel = None
            break
        starting_voxel = grid.gridify_point(point)
        i += 1

    if starting_voxel is not None:
        fill_nodes = utils.connected_components(volume, starting_voxel)
        fill_coords = np.unravel_index(fill_nodes, grid.shape)
        volume[fill_coords[0], fill_coords[1], fill_coords[2]] = True
    volume_amount = np.count_nonzero(volume) * (args.grid_size ** 3)
    # Get number of True values in the flattened array and multiply by grid size.
    print("Volume: {} Å³".format(volume_amount))

    if args.vertices_file:
        # Run connected components on the volume grid to get SES nodes
        # ses_nodes = utils.connected_components(np.invert(volume), stored_starting_voxel)
        # verts = np.array(np.unravel_index(ses_nodes, grid.shape)).T * args.grid_size + grid.zero_shift
        mesh = mesh.Triangle(volume)
        # Convert SES nodes to coordinates in the original atom coordinate system.
        # Generate an xyz file with atoms called X.
        if args.vertices_file.endswith(".xyz"):
            with open(args.vertices_file, 'w+') as vf:
                vf.write("{}\nVolume: {} Å³\n".format(str(verts.shape[0]), volume_amount))
                for l in verts:
                    vf.write("X {} {} {}\n".format(*list(l)))
        elif args.vertices_file.endswith(".csv"):
            np.savetxt(args.vertices_file, verts, header="x,y,z", comments="", delimiter=",")
        elif args.vertices_file.endswith(".txt"):
            np.savetxt(args.vertices_file, verts, header="Volume: {} Å³".format(volume_amount))
        elif args.vertices_file.endswith(".npz"):
            np.savez_compressed(args.vertices_file, verts)
        else:
            raise exception.InvalidFileExtension([".xyz", ".csv", ".txt", ".npz"])

    end = time_ns() - start
    print("Took:", end*10**(-9), "s")


if __name__ == '__main__':
    main()
