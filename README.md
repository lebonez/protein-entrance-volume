# Protein Entrance Volume

This is an algorithm ran by command line that calculates the volume of an entrance cavity using user provided boundaries determined by residue ID.

### Installation
Should work with most version 3 Pythons. Recommend Python 3.9

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
python main.py -o 107 78 79 230 -i 224 106 108 -r 1.5 -g 0.2 -R 4 -f example/pdbs/example.pdb
```

### Command line basics.

```
usage: main.py [-h] -o OUTER_RESIDUES [OUTER_RESIDUES ...] -i INNER_RESIDUES [INNER_RESIDUES ...] [--no-outer] [--no-inner] [-r PROBE_RADIUS] [-g GRID_SIZE] [-R RESOLUTION] -f PDB_FILE [-v VERTICES_FILE]

Parse pdb file to get protein entrance volume determined by inner and outer residues.

optional arguments:
  -h, --help            show this help message and exit
  -o OUTER_RESIDUES [OUTER_RESIDUES ...], --outer-residues OUTER_RESIDUES [OUTER_RESIDUES ...]
                        A list of three or more outer residues to define the initial entrance of the tunnel.
  -i INNER_RESIDUES [INNER_RESIDUES ...], --inner-residues INNER_RESIDUES [INNER_RESIDUES ...]
                        A list of three or more inner residues to define the desired ending location in tunnel.
  --no-outer            Don't use outer residues boundary this is helpful if the outer residues shift positions alot.
  --no-inner            Don't use inner residues boundary this is helpful if the inner residues shift positions alot.
  -r PROBE_RADIUS, --probe-radius PROBE_RADIUS
                        Radius of the algorithm probe to define the inner surface of the cavity (default: 1.4).
  -g GRID_SIZE, --grid-size GRID_SIZE
                        The size of the grid to use to calculate the cavity inner surface (default: 0.2).
  -R RESOLUTION, --resolution RESOLUTION
                        Lower values decreases runtime and higher values for accuracy (default: 4).
  -f PDB_FILE, --pdb-file PDB_FILE
                        Path to the PDB file.
  -v VERTICES_FILE, --vertices-file VERTICES_FILE

                        Output the vertices to file which file types depends on the file extension provided in this argument.
                            xyz: Outputs the vertices as a molecular xyz file with each vertices marked as an "X" atom and has volume as the comment line after number of atoms.
                            csv: Vertices array is dumped to a file with "x,y,z" as header and each line containing a comma separated x,y,z coordinate.
                            txt: Vertices array is dumped to a txt file with first line containing volume of vertices and x y z coordinates space separated.
                            npz: Recommended if loading the vertices array back into numpy for post processing uses much less space and is faster.
```

For help using contact: miwalls@siue.edu
