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
from protein_entrance_volume import io
from protein_entrance_volume import visualization


def parse_args():
    """
    Parse args
    """
    parser = argparse.ArgumentParser(
        description="Parse pdb file to get "
        "protein entrance volume determined by inner and outer residues.",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-o', '--outer-residues', required=True, type=int,
        nargs='+', help="A list of three or more outer residues to define the "
        "initial entrance of the tunnel.")
    parser.add_argument(
        '-i', '--inner-residues', required=True, type=int,
        nargs='+', help="A list of three or more inner residues to define the "
        "desired ending location in the tunnel.")
    parser.add_argument(
        '--no-outer', default=False, action='store_true',
        help="Don't use outer residues boundary hemisphere this is helpful if "
        "the outer residues shift positions alot and intersect.")
    parser.add_argument(
        '--no-inner', default=False, action='store_true',
        help="Don't use inner residues boundary hemisphere this is helpful if "
        "the inner residues shift positions alot and intersect.")
    parser.add_argument(
        '-r', '--probe-radius', default=1.4, type=float,
        help="Radius of the algorithm probe to define the inner surface of "
        "the cavity (default: %(default)s).")
    parser.add_argument(
        '-g', '--grid-size', default=0.2, type=float,
        help="The size of the grid to use to calculate the cavity inner "
        "surface (default: %(default)s).")
    parser.add_argument(
        '-R', '--resolution', default=4, type=float,
        help="Lower values decreases runtime and higher values for accuracy "
        "(default: %(default)s).")
    parser.add_argument(
        '-f', '--pdb-file', required=True, type=str,
        help="Path to the PDB file.")
    parser.add_argument(
        '-V', '--visualize', const='scatter', nargs='?',
        choices=('scatter',),
        help="If specified, creates a visualization plot (default: scatter).")
    parser.add_argument(
        '-v', '--vertices-file', default="", type=str,
        help="""
Output the vertices to file which file types depends on the file extension
provided in this argument.
    xyz: Outputs the vertices as a molecular xyz file with each vertices
         marked as an "X" atom and has volume as the comment line after number
         of atoms by far slowest file output.
    csv: Vertices array is dumped to a file with "x,y,z" as header and each
         line containing a comma separated x,y,z coordinate.
    txt: Vertices array is dumped to a txt file with first line containing
         volume of vertices and x y z coordinates space separated.
    npz: Recommended if loading the vertices array back into numpy for post
         processing uses much less space and is faster.
    """)
    return parser.parse_args()


def main():
    """
    Main function to use the cli
    """
    start = time_ns()
    args = parse_args()
    if len(args.outer_residues) < 3 or len(args.inner_residues) < 3:
        msg = 'Number of integer values for "--outer-residues" and ' \
              '--inner-residues" must be >= 3'
        raise argparse.ArgumentError(None, msg)

    # Build the atoms dataframe object.
    atoms = Atoms.parse_atoms(
        args.pdb_file,
        outer_residues=args.outer_residues,
        inner_residues=args.inner_residues
    )

    # Reduce computation complexity for the grid by only using spheres within
    # the mbr calculated by the inner and outer residue atoms.
    atoms_mbr = atoms.residues_mbr(extension=args.probe_radius)

    # how far should each faux sphere be on the boundary hemispheres and
    # sphere. Smaller distance make a slight difference in accuracy but increases
    # computation time significantly.
    points_distance = args.probe_radius / args.resolution

    # Generate the larger outer boundary sphere coords
    b_sphere = boundary.Sphere(atoms.arc, atoms.ar_coords, points_distance)
    coords = b_sphere.coords
    # No outer means to not create the outer residue faux hemisphere boundary
    if not args.no_outer:
        # Outer residue boundary hemisphere coords same explanations as
        # above except for only generates on the outer side of the best fit
        # plane of the outer residue atoms.
        o_hsphere = boundary.Hemisphere(atoms.orc, atoms.or_coords, atoms.irc,
                                        points_distance)
        coords = np.vstack((coords, o_hsphere.coords))
    # No inner means to not create the inner residue faux spheres boundary
    if not args.no_inner:
        # Inner residue boundary hemisphere coords same explanations as
        # above except for only generates on the outer side of the best fit
        # plane of the inner residue atoms.
        i_hsphere = boundary.Hemisphere(atoms.irc, atoms.ir_coords,
                                        atoms.orc, points_distance)
        coords = np.vstack((coords, i_hsphere.coords))
    # Append probe extended atom radii array with an array of length equal to
    # the total number of boundary spheres filled with values of probe radius.
    radii = np.append(atoms_mbr[1] + args.probe_radius,
                      np.full(coords.shape[0], args.probe_radius))
    coords = np.vstack((atoms_mbr[0], coords))

    # Generate the SAS grid from spherical points and radii within the MBR.
    grid = Grid.from_cartesian_spheres(coords, radii, grid_size=args.grid_size,
                                       fill_inside=True)

    # Find an empty starting voxel near the tip of the outer residue
    # hemisphere. This is the best location since it is the actual beginning
    # point of the entrance. The algorithm loops the voxel while incrementing
    # the voxel in the opposite direction of the outer hemisphere normal if and
    # until it reaches the atom outer residue centroid at which point it raises
    # a voxel not found exception.
    starting_voxel = grid.find_empty_voxel(o_hsphere.tip, atoms.orc,
                                           -o_hsphere.normal)

    # Calculate the SAS volume nodes and SAS border nodes using
    # connected components.
    nodes, sas_nodes = utils.connected_components(grid.grid, starting_voxel)

    # A beginning first voxel for calculating the probe extended grid using
    # the first sas border node.
    voxel = np.array(np.unravel_index(sas_nodes[0], grid.shape))

    # Build the initial grid using the first sas node voxel coordinate
    # extended by the probe radius divided by the grid size.
    volume_grid = rasterize.sphere(
        voxel, args.probe_radius / args.grid_size,
        np.zeros(grid.shape, dtype=bool), fill_inside=False
    )

    # flatten array for the subsequent calculations.
    volume_grid = volume_grid.flatten()

    # Calculate the spherical 1D equation for the probe radius grid sphere
    eqn = np.argwhere(volume_grid).flatten() - sas_nodes[0]

    # Apply all of the spheres at the remaining SAS border nodes using the
    # sphere equation above calculated using the first sas node voxel.
    volume_grid = utils.eqn_grid(sas_nodes[1:], eqn, volume_grid)

    # Set the SES nodes from the initial connected components run to true this
    # fills any remaining holes which is faster then any binary fill method
    # from other libraries.
    volume_grid[nodes] = True

    # Build the SAS volume grid object which includes the center (non-border)
    # voxels as well
    # Note: it uses the previous grid zero shift attribute from the first atom
    # coordinate based grid.
    grid = Grid(
        volume_grid.reshape(grid.shape), zero_shift=grid.zero_shift,
        grid_size=args.grid_size
    )

    # Calculate the volume by counting true values in the above grid and
    # multiplying by grid size cubed.
    volume_amount = np.count_nonzero(grid.grid) * (args.grid_size ** 3)

    # Print out the volume.
    print(f"Volume: {volume_amount} Å³")
    print(f"Took: {(time_ns() - start) * 10 ** (-9)}s")

    if args.vertices_file or args.visualize:
        # Run connected components on the volume grid to get SES border nodes
        # for file generation also invert the grid since connected components
        # searches for false values.
        _, ses_nodes = utils.connected_components(
            np.invert(grid.grid), starting_voxel, border_only=True
        )
        # Calculate center of voxels then scale and shift them back to the
        # original atom coordinates system.
        verts = ((np.array(np.unravel_index(ses_nodes, grid.shape)).T + 0.5)
                 * args.grid_size + grid.zero_shift)
        if args.vertices_file:
            io.vertices_file(args.vertices_file, verts)
        if args.visualize:
            visualization.coordinates(verts, plot_type=args.visualize)


if __name__ == '__main__':
    main()
