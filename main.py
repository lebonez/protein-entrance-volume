#!/usr/bin/env python
"""
Created by: Mitchell Walls
Email: miwalls@siue.edu
"""
import argparse
from time import time_ns
from protein_entrance_volume import parser
from protein_entrance_volume import grid
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
        '-R', '--resolution', default=4, type=int,
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
Output the vertices of the entrance volume SES to file where the file type
generated is deteremined by the file extension type provided in this argument.
    xyz: Outputs the vertices as a molecular xyz file with each vertices
         marked as a "DOT" atom. This is by far the slowest file output.
    csv: Vertices array is dumped to a file with "x,y,z" as header and each
         line containing a comma separated x,y,z coordinate.
    txt: Vertices array is dumped to a txt file with first line containing
         volume of vertices and x y z coordinates space separated.
    npz: Numpy array compressed binary file which is recommended if loading the
         vertices array back into numpy for post processing uses much less
         space and is faster.
    """)
    return parser.parse_args()


def main():
    """
    Main function to use the cli
    """
    args = parse_args()

    if args.probe_radius > 2 or args.probe_radius < 1:
        msg = 'Probe radius must be a value between 1 and 2.'
        raise argparse.ArgumentError(None, msg)

    if len(args.outer_residues) < 3 or len(args.inner_residues) < 3:
        msg = 'Number of integer values for "--outer-residues" and ' \
              '--inner-residues" must be >= 3'
        raise argparse.ArgumentError(None, msg)

    # Build the protein objects processes multiple frames
    proteins = parser.parse_pdb(
        args.pdb_file, outer_residues=args.outer_residues,
        inner_residues=args.inner_residues)

    results_file = f"{time_ns()}.results"

    for i, protein in enumerate(proteins):
        print(f"Frame: {i}")
        start = time_ns()
        # Generate the entrance of the protein which includes making all of the
        # boundaries and minimizing the problem scope to a minimum bounding
        # rectangle.
        protein.generate_entrance(
            args.no_outer, args.no_inner, args.probe_radius,
            args.resolution)

        try:
            # Calculate the SAS from the entrance coords and radii. The hemisphere
            # gives us a starting location since we need to know where the volume
            # itself begins and the protein orc is where we should stop looking.
            # Also the algorithm itself tries to find the starting voxel of the SAS
            # by iterating the direction of the opposite of the outer hemisphere
            # normal.
            sas = grid.SAS(
                protein.entrance.coords, protein.entrance.radii,
                protein.entrance.outer_hemisphere.tip, protein.orc,
                -protein.entrance.outer_hemisphere.normal, args.grid_size,
                fill_inside=True)

            # Using the SAS calculate the SES by expanding all the SAS border nodes
            # by the probe radius spherically.
            ses = grid.SES(sas, args.probe_radius)

            if args.vertices_file or args.visualize:
                # Grab the outer border vertices of the SES this takes a while
                # because it runs connected components again searching for borders
                # only so it is slightly faster than a full node search.
                verts = ses.vertices
                if args.vertices_file:
                    name_list = args.vertices_file.split('.')
                    name_list[-2] = f"{name_list[-2]}_{i+1}"
                    io.vertices_file(".".join(name_list), verts)
                if args.visualize:
                    visualization.coordinates(verts, plot_type=args.visualize)

            # Print out the volume.
            with open(results_file, "a+") as results:
                results.write(f"{i+1},{ses.volume}\n")
            print(f"Volume: {ses.volume} Å³")
        except Exception as e:
            with open(results_file, "a+") as results:
                results.write(f"{i+1},{e}\n")
            print(f"Failed: {e} Å³")

        print(f"Took: {(time_ns() - start) * 10 ** (-9)}s")


if __name__ == '__main__':
    main()
