#!/usr/bin/env python
"""
Created by: Mitchell Walls
Email: miwalls@siue.edu
"""
import argparse
from time import time_ns
import numpy as np
from protein_entrance_volume import atoms
from protein_entrance_volume import rasterize
from protein_entrance_volume import grid
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
    protein = atoms.Protein.parse_atoms(
        args.pdb_file, outer_residues=args.outer_residues,
        inner_residues=args.inner_residues)

    # Generate the entrance of the protein which includes making all of the
    # boundaries and minimizing the problem scope to a minimum bounding
    # rectangle.
    protein.generate_entrance(
        args.no_outer, args.no_inner, args.probe_radius,
        args.resolution)

    # Calculate the SAS from the entrance coords and radii. The hemisphere tip
    # gives us a starting location since we need to know where the volume
    # itself begins and the protein orc is where we should stop looking. Also
    # the algorithm itself tries to find the starting voxel of the SAS by
    # iterating the direction of the opposite of the outer hemisphere normal.
    sas = grid.SAS(protein.entrance.coords, protein.entrance.radii,
        protein.entrance.outer_hemisphere.tip, protein.orc,
        -protein.entrance.outer_hemisphere.normal, args.grid_size,
        fill_inside=True)

    ses = grid.SES(sas, args.probe_radius)

    # Print out the volume.
    print(f"Volume: {ses.volume} Å³")
    print(f"Took: {(time_ns() - start) * 10 ** (-9)}s")

    if args.vertices_file or args.visualize:
        # Run connected components on the volume grid to get SES border nodes
        # for file generation also invert the grid since connected components
        # searches for false values.
        _, ses_nodes = utils.connected_components(
            np.invert(ses.grid.grid), sas.starting_voxel, border_only=True
        )
        # Calculate center of voxels then scale and shift them back to the
        # original atom coordinates system.
        verts = ((np.array(np.unravel_index(ses_nodes, sas.grid.shape)).T
                 + 0.5) * args.grid_size + sas.grid.zero_shift)
        if args.vertices_file:
            io.vertices_file(args.vertices_file, verts)
        if args.visualize:
            visualization.coordinates(verts, plot_type=args.visualize)


if __name__ == '__main__':
    main()
