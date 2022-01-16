### Cleanup or Optimizations

Grid SAS and SES classes are messy and not clear/generic enough to be
considered pythonic.

Add in some abstract inheritance classes to clean up repetitive code in classes.

### New Features

As a result of below make pip install from git location.

Would be nice to have an installation using setup with cli described below having
the program called be something like "pev -f example.pdb..." rather than "python main.py -f example.pdb..."

Create a nice cli so that we can do more things like calculate the SES/SAS of
the whole protein and not just the volume and much more. https://docs.python.org/3/library/argparse.html#sub-commands

Handle other files types like DCD or PSF files rather than just PDBS. Problem is this adds command line complexity.
PDBs get really large at greater than hundreds of thousands of frames

### Known Issues

There is an issue where certain atoms from the PDB file have no radius.
Typically this is the heme and other hydrogen atoms
so for now it is just set to 1.2 angstrom by default.
