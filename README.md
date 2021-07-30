# Protein Entrance Volume

This is an algorithm ran by command line that calculates the volume of an entrance cavity using user provided boundaries determined by residue ID.

### Command line basics.

usage: main.py [-h] -o OUTER_RESIDUES [OUTER_RESIDUES ...] -i INNER_RESIDUES [INNER_RESIDUES ...]
               [--no-outer] [--no-inner] [-r PROBE_RADIUS] [-g GRID_SIZE] -f PDB_FILE [-V]

Parse pdb file to get tunnel volume

optional arguments:
  -h, --help            show this help message and exit
  -o OUTER_RESIDUES [OUTER_RESIDUES ...], --outer-residues OUTER_RESIDUES [OUTER_RESIDUES ...]
                        A list of three or more outer residues to define the entrance of the tunnel.
  -i INNER_RESIDUES [INNER_RESIDUES ...], --inner-residues INNER_RESIDUES [INNER_RESIDUES ...]
                        A list of three or more inner residues to define the end of the tunnel.
  --no-outer            Don't use outer residues boundary this is helpful if the outer residues shift
                        positions alot.
  --no-inner            Don't use inner residues boundary this is helpful if the inner residues shift
                        positions alot.
  -r PROBE_RADIUS, --probe-radius PROBE_RADIUS
                        Radius of the algorithm probe to define the inner surface of the cavity.
  -g GRID_SIZE, --grid-size GRID_SIZE
                        The size of the grid to use to calculate the cavity inner surface.
  -f PDB_FILE, --pdb-file PDB_FILE
                        Path to the PDB file.
  -V, --visualize       Creates and HTML file of the plot

For help using contact: miwalls@siue.edu
