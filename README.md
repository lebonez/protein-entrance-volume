# Protein Entrance Volume

This is an algorithm ran by command line that calculates the volume of an entrance cavity using user provided boundaries determined by residue ID.

### Known issues

* The Numba compiled connected components algorithm doesn't handle errors well and segmentation faults. Now handles unreasonable volumes by returning largest.
* Finding an initial free voxel to start the connected components algorithm can be inaccurate but it starts with the centroid of the cavity calculated using the centroid between the inner and outer user provided residues. It does its best to filter out invalid starting indexes by filtering out very small connected component results less than 100 empty voxels.
* Very small grid sizes less than 0.05 angstroms can cause the code to never finish. Some of the code is log(n) time I used hash lookup tables as much as possible (Python sets/Numba typed dicts) but spherical grid calculations are not that optimized it does as much as possible in parallel but performance drops off past 8 physical CPU cores. One simple speed up would be to not iterate through all border voxels to extend by border radius but to do this analytically (haven't found a method yet maybe could use Poisson reconstruction to do this?). There can be trillions of border voxels for small grid sizes.
* Visualize is not great needs command arguments it'll likely be confusing and might not work yet. Read the bottom of main.py can give an idea of how it works.
* There is an issue where certain atoms from the PDB file have no radius. Typically this is the heme so for now it is just set to 1 angstrom by default.
* Some frames might have a situation where the entrance doesn't actually include any inner part of the protein. This is caused by the "hole" of the entrance is smaller than the probe radius supplied by the command line.
* I think that is all....

### Command line basics.

```
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
  -V, --visualize       Creates an ply/html file of the volume
```

For help using contact: miwalls@siue.edu
