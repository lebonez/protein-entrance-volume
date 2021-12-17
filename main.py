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
    parser.add_argument('-g', '--grid-size', default=0.2, type=float, help="The size of the grid to use to calculate the cavity inner surface (default: %(default)s).")
    parser.add_argument('-R', '--resolution', default=4, type=float, help="Lower values decreases runtime and higher values for accuracy default: %(default)s).")
    parser.add_argument('-f', '--pdb-file', required=True, type=str, help="Path to the PDB file.")
    parser.add_argument('-v', '--vertices-file', default="", type=str, help="""
Output the vertices to file which file types depends on the file extension provided in this argument.
    xyz: Outputs the vertices as a molecular xyz file with each vertices marked as an "X" atom and has volume as the comment line after number of atoms.
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
        msg = 'Number of integer values for "--outer-residues" and "--inner-residues" must be >= 3'
        raise argparse.ArgumentError(None, msg)

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

    # The coordinates of atoms inside the MBR
    mbr_coords = atoms_mbr[0]

    # Increase the radii of the MBR atoms by probe size for SAS calculation.
    mbr_radii = atoms_mbr[1] + args.probe_radius

    # Create a larger outer spherical boundary set of faux spheres with radius
    # being the average distance from the centroid of all residues to the coords
    # of all residues.
    b_distance = utils.average_distance(atoms.arc, atoms.ar_coords)

    # Calculate the optimal number of faux spheres so that the faux spheres are
    # within the probe_radius divided by the resolution to each other.
    b_num_points = utils.sphere_num_points(b_distance, args.probe_radius / resolution)

    # Generate the larger outer boundary sphere coords
    b_max = boundary.sphere(b_distance, atoms.arc, num_points=b_num_points)

    # Copy points for creating a complete array of boundary coords
    b_points = np.copy(b_max)

    # Same as above but for the outer and inner residues separately.
    o_distance = utils.average_distance(atoms.orc, atoms.or_coords)
    i_distance = utils.average_distance(atoms.irc, atoms.ir_coords)
    # No outer means to not create the outer residue faux spheres boundary
    if not args.no_outer:
        o_num_points = utils.sphere_num_points(o_distance, args.probe_radius / resolution)
        # Outer residue boundary half spheres coords.
        b_outer = boundary.half_sphere(atoms.or_coords, o_distance, atoms.orc, atoms.irc, num_points=o_num_points)
        # Append to previous copied array
        b_points = np.append(b_points, b_outer, axis=0)
    # No inner means to not create the inner residue faux spheres boundary
    if not args.no_inner:
        i_num_points = utils.sphere_num_points(i_distance, args.probe_radius / resolution)
        # Inner residue boundary half spheres coords.
        b_inner = boundary.half_sphere(atoms.ir_coords, i_distance, atoms.irc, atoms.orc, num_points=i_num_points)
        # Append to previous copied array
        b_points = np.append(b_points, b_inner, axis=0)

    # Build a radii array that has same length as boundary points and set the radii to the probe_radius
    # This allows the SAS to directly translate to an SES along the spherical and half spherical boundaries coordinates.
    b_radii = np.full(b_points.shape[0], args.probe_radius)
    # Combine all of the boundary coords and radii with the mbr coords and radii.
    coords = np.append(mbr_coords, b_points, axis=0)
    radii = np.append(mbr_radii, b_radii, axis=0)

    # Find a starting point for the surface search that is on the center of the
    # surface of the outer residue half sphere..
    # This is the most ideal location to start the search since it
    # would be the actual beginning of the entrance.
    # Best fit plane of the outer residue coords
    plane = utils.best_fit_plane(atoms.or_coords)
    # Find our orientation of the normal by determining the side the inner residue
    # centroid lies on and make sure the normal from the plane above points in the
    # opposite direction.
    opposing_side = utils.side_point(plane, atoms.irc)
    # Make sure the normal is not pointing at the opposing side if it is swap it.
    if utils.side_point(plane, plane[1] + atoms.orc) == opposing_side:
        # Swap normal to point away from the irc if it wasn't already.
        normal = plane[1] * -1
    # Calculate the point using the distance to the outer residue half spheres
    # subtracted by the probe radius multiplied by the normal from above then
    # shifted to the outer residue centroid.
    starting_point = (o_distance - args.probe_radius) * normal + atoms.orc
    # Generate the SAS grid from spherical points and radii within the MBR.
    grid = Grid.from_cartesian_spheres(coords, radii, grid_size=args.grid_size, fill_inside=True)
    # Find a good starting voxel to do SAS search by starting at the starting
    # point that was calculated above and moving towards the orc.
    starting_voxel = grid.gridify_point(starting_point)
    i = 1
    # Swap the normal back to point towards towards the orc from the starting point
    normal *= -1
    # Loop through and add ith iteration times the normal until we get an empty
    # (outside of spheres) voxel.
    # test_points.append(starting_point)
    while grid.grid[starting_voxel[0], starting_voxel[1], starting_voxel[2]]:
        # Calculate the point using the ith magnitude normal
        point = starting_point + (i * normal)
        # We are now too close to the outer residue centroid to consider any
        # starting voxel valid so we raise custom exception here and user needs
        # to tune command line parameters.
        if np.linalg.norm(point - atoms.orc) < args.grid_size * 2:
            raise exception.StartingVoxelNotFound
        # Convert point to the integer grid voxel coordinate.
        starting_voxel = grid.gridify_point(point)
        # Increment the ith magnitude normal factor.
        i += 1

    # Calculate the SAS volume nodes and SAS border nodes using connected_components
    nodes, sas_nodes = utils.connected_components(grid.grid, starting_voxel)

    # Rasterize an example sphere on the same grid as above at the center of the grid
    single_sphere = rasterize.sphere(np.int64(np.array(grid.shape) / 2), args.probe_radius / args.grid_size, np.zeros(grid.shape, dtype=bool), fill_inside=False)
    # Calculate the equation for the example rasterized sphere
    eqn = np.where(single_sphere.flatten())[0] - np.ravel_multi_index(np.int64(np.array(grid.shape) / 2), grid.shape)
    # Apply all of the spheres at every SAS border node.
    volume_grid = utils.eqn_grid(sas_nodes, eqn, np.zeros(np.prod(grid.shape), dtype=bool))
    # Set the nodes from the previous connected components to true.
    volume_grid[nodes] = True
    # Build the volume grid object
    grid = Grid(volume_grid.reshape(grid.shape), zero_shift=grid.zero_shift, grid_size=args.grid_size)
    # Calculate the volume.
    volume_amount = np.count_nonzero(grid.grid) * (args.grid_size ** 3)
    print("Volume: {} Å³".format(volume_amount))

    if args.vertices_file:
        # Run connected components on the volume grid to get SES border nodes for file generation
        _, ses_nodes = utils.connected_components(np.invert(grid.grid), starting_voxel, border_only=True)
        verts = np.array(np.unravel_index(ses_nodes, grid.shape)).T * args.grid_size + grid.zero_shift
        # Convert SES nodes to coordinates in the original atom coordinate system.
        # Generate an xyz file with atoms called X.
        if args.vertices_file.endswith(".xyz"):
            with open(args.vertices_file, 'w+', encoding='utf-8') as vertices_file:
                vertices_file.write("{}\nVolume: {} Å³\n".format(str(verts.shape[0]), volume_amount))
                for vert in verts:
                    vertices_file.write("X {} {} {}\n".format(*list(vert)))
        elif args.vertices_file.endswith(".csv"):
            np.savetxt(args.vertices_file, verts, header="x,y,z", comments="", delimiter=",")
        elif args.vertices_file.endswith(".txt"):
            np.savetxt(args.vertices_file, verts, header="Volume: {} Å³".format(volume_amount))
        elif args.vertices_file.endswith(".npz"):
            np.savez_compressed(args.vertices_file, verts)
        else:
            raise exception.InvalidFileExtension([".xyz", ".csv", ".txt", ".npz"])
    end = time_ns() - start
    print("Took:", end * 10 ** (-9), "s")


if __name__ == '__main__':
    main()
