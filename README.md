# Protein Entrance Volume

This is an algorithm ran by command line that calculates the volume of an entrance cavity using user provided boundaries determined by residue ID.

### Installation
Should work with all python versions 3.6 and above. Recommend Python 3.9 as of 12/17/2021 (Numba does not work with higher versions).

Download or clone the repo:

```
wget https://github.com/lebonez/protein-entrance-volume/archive/refs/heads/main.zip
unzip main.zip
```

or

```
git clone https://github.com/lebonez/protein-entrance-volume.git
```

Install python requirements:

```
cd protein-entrance-volume
pip install -f requirements.txt
```

### Example
```
python main.py -o 107 78 79 230 -i 224 106 108 -r 1.5 -g 0.2 -R 4 -f example/pdbs/example.pdb -v example.xyz
```

Dumping to an xyz file allows it to be imported easily to VMD for viewing just load in the example.pdb
then load in the example.xyz file to view the results.

### Command line basics.

```
usage: main.py [-h] -o OUTER_RESIDUES [OUTER_RESIDUES ...] -i INNER_RESIDUES [INNER_RESIDUES ...] [--no-outer] [--no-inner] [-r PROBE_RADIUS] [-g GRID_SIZE] [-R RESOLUTION] -f PDB_FILE [-F FRAMES] [-d]
               [-V [{scatter}]] [-v VERTICES_FILE]

Parse pdb file to get protein entrance volume determined by inner and outer residues.

optional arguments:
  -h, --help            show this help message and exit
  -o OUTER_RESIDUES [OUTER_RESIDUES ...], --outer-residues OUTER_RESIDUES [OUTER_RESIDUES ...]
                        A list of three or more outer residues to define the initial entrance of the tunnel.
  -i INNER_RESIDUES [INNER_RESIDUES ...], --inner-residues INNER_RESIDUES [INNER_RESIDUES ...]
                        A list of three or more inner residues to define the desired ending location in the tunnel.
  --no-outer            Don't use outer residues boundary hemisphere this is helpful if the outer residues shift positions alot and intersect.
  --no-inner            Don't use inner residues boundary hemisphere this is helpful if the inner residues shift positions alot and intersect.
  -r PROBE_RADIUS, --probe-radius PROBE_RADIUS
                        Radius of the algorithm probe to define the inner surface of the cavity (default: 1.4).
  -g GRID_SIZE, --grid-size GRID_SIZE
                        The size of the grid to use to calculate the cavity inner surface (default: 0.2).
  -R RESOLUTION, --resolution RESOLUTION
                        Lower values decreases runtime and higher values for accuracy (default: 4).
  -f PDB_FILE, --pdb-file PDB_FILE
                        Path to the PDB file.
  -F FRAMES, --frames FRAMES
                        Specify specific frames to run for a multiframe PDB file. Can be a range (i.e. 253-1014), comma separated (i.e. 1,61,76,205), or a single frame (i.e. 25). Can also be a combination (i.e. 1-25,205,1062-2052).
  -d, --dump-results    Dumps results to a ((datetime).results) file where each line is ({frame_index},{volume}).
  -V [{scatter}], --visualize [{scatter}]
                        If specified, creates a visualization plot (default: scatter).
  -v VERTICES_FILE, --vertices-file VERTICES_FILE

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
```

For help using contact: miwalls@siue.edu
