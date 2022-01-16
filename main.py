#!/usr/bin/env python
"""
Created by: Mitchell Walls
Email: miwalls@siue.edu
"""
import argparse
import time
from protein_entrance_volume import parser
from protein_entrance_volume import grid
from protein_entrance_volume import io
from protein_entrance_volume import utils
from protein_entrance_volume import visualization


def parse_args():
    """
    Parse args
    """
    arg_parser = argparse.ArgumentParser(
        description="Parse pdb file to get "
        "protein entrance volume determined by inner and outer residues.",
        formatter_class=argparse.RawTextHelpFormatter)
    arg_parser.add_argument(
        '-o', '--outer-residues', required=True, type=int,
        nargs='+', help="A list of three or more outer residues to define the "
        "initial entrance of the tunnel.")
    arg_parser.add_argument(
        '-i', '--inner-residues', required=True, type=int,
        nargs='+', help="A list of three or more inner residues to define the "
        "desired ending location in the tunnel.")
    arg_parser.add_argument(
        '--no-outer', default=False, action='store_true',
        help="Don't use outer residues boundary hemisphere this is helpful if "
        "the outer residues shift positions alot and intersect.")
    arg_parser.add_argument(
        '--no-inner', default=False, action='store_true',
        help="Don't use inner residues boundary hemisphere this is helpful if "
        "the inner residues shift positions alot and intersect.")
    arg_parser.add_argument(
        '-r', '--probe-radius', default=1.4, type=float,
        help="Radius of the algorithm probe to define the inner surface of "
        "the cavity (default: %(default)s).")
    arg_parser.add_argument(
        '-g', '--grid-size', default=0.2, type=float,
        help="The size of the grid to use to calculate the cavity inner "
        "surface (default: %(default)s).")
    arg_parser.add_argument(
        '-R', '--resolution', default=4, type=int,
        help="Lower values decreases runtime and higher values for accuracy "
        "(default: %(default)s).")
    arg_parser.add_argument(
        '-f', '--pdb-file', required=True, type=str,
        help="Path to the PDB file.")
    arg_parser.add_argument(
        '-F', '--frames', required=False, type=str,
        help="Specify specific frames to run for a multiframe PDB file. Can "
        "be a range (i.e. 253-1014), comma separated (i.e. 1,61,76,205), or "
        "a single frame (i.e. 25). Can also be a combination "
        "(i.e. 1-25,205,1062-2052).")
    arg_parser.add_argument(
        '-d', '--dump-results', default=False, action='store_true',
        help="Dumps results to a ((datetime).results) file where each line is "
        "({frame_index},{volume}).")
    arg_parser.add_argument(
        '-V', '--visualize', const='scatter', nargs='?',
        choices=('scatter',),
        help="If specified, creates a visualization plot (default: scatter).")
    arg_parser.add_argument(
        '-v', '--vertices-file', default="", type=str,
        help="""
Output the vertices of the entrance volume SES to file where the file type
generated is deteremined by the file extension type provided in this argument.
If there are multiple frames the current frame index is prepended to the file
extension (i.e. example_{index}.xyz)
    xyz: Outputs the vertices as a molecular xyz file with each vertices
         marked as a "DOT" atom. This is by far the slowest file output.
    csv: Vertices array is dumped to a file with "x,y,z" as header and each
         line containing a comma separated x,y,z coordinate.
    txt: Vertices array is dumped to a txt file with x y z coordinates on each
         line space separated.
    npz: Numpy array compressed binary file which is recommended if loading the
         vertices array back into numpy for post processing uses much less
         space and is faster.
    """)
    return arg_parser.parse_args()


def main():
    """
    Main function to use the cli
    """
    args = parse_args()

    if args.frames:
        frames = utils.parse_frames(args.frames)

    if args.probe_radius > 2 or args.probe_radius < 1:
        msg = 'Probe radius must be a value between 1 and 2.'
        raise argparse.ArgumentError(None, msg)

    if len(args.outer_residues) < 3 or len(args.inner_residues) < 3:
        msg = 'Number of integer values for "--outer-residues" and ' \
              '--inner-residues" must be >= 3'
        raise argparse.ArgumentError(None, msg)

    # Build the protein objects processes multiple frames
    protein_frames = parser.parse_pdb(
        args.pdb_file, outer_residues=args.outer_residues,
        inner_residues=args.inner_residues)

    results_file = f"{time.strftime('%Y%m%d-%H%M%S')}.results"

    for i, protein in enumerate(protein_frames):
        if args.frames and (i + 1) not in frames:
            continue

        print(f"Frame: {i + 1}")
        start = time.time_ns()
        # Generate the entrance of the protein which includes making all of the
        # boundaries and minimizing the problem scope to a minimum bounding
        # rectangle.
        protein.generate_entrance(
            args.no_outer, args.no_inner, args.probe_radius,
            args.resolution)

        try:
            # Calculate the SAS from the entrance coords and radii. The
            # hemisphere gives us a starting location since we need to know
            # where the volume itself begins and the protein orc is where we
            # should stop looking. Also the algorithm itself tries to find the
            # starting voxel of the SAS by iterating the direction of the
            # opposite of the outer hemisphere normal.
            sas = grid.SAS(
                protein.entrance.coords, protein.entrance.radii,
                protein.entrance.outer_hemisphere.tip, protein.orc,
                -protein.entrance.outer_hemisphere.normal, args.grid_size,
                fill_inside=True)

            # Using the SAS calculate the SES by expanding all the SAS border
            # nodes by the probe radius spherically.
            ses = grid.SES(sas, args.probe_radius)

            if args.vertices_file or args.visualize:
                # Grab the outer border vertices of the SES this takes a while
                # because it runs connected components again searching for
                # borders only so it is slightly faster than a full node
                # search.
                verts = ses.vertices
                if args.vertices_file:
                    name_list = args.vertices_file.split('.')
                    name_list[-2] = f"{name_list[-2]}_{i + 1}"
                    io.vertices_file(".".join(name_list), verts)
                if args.visualize:
                    visualization.coordinates(verts, plot_type=args.visualize)

            # Write out the results to a timestamped file.
            if args.dump_results:
                with open(results_file, "a+", encoding='utf-8') as results:
                    results.write(f"{i + 1},{ses.volume}\n")
            # Print out the volume.
            print(f"Volume: {ses.volume} Å³")
        except Exception as exception:
            # Write out the failures to a timestamped file.
            if args.dump_results:
                with open(results_file, "a+", encoding='utf-8') as results:
                    results.write(f"{i + 1},{exception}\n")
            print(f"Failed: {exception} Å³")

        print(f"Took: {(time.time_ns() - start) * 10 ** (-9)}s")


if __name__ == '__main__':
    main()
