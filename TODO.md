### Cleanup or Optimizations

Grid SAS and SES classes are messy and not clear/generic enough to be
considered pythonic.

Add in some abstract inheritance classes to clean up repetitive code in classes.

### New Features

Would be nice to have an installation using setup with cli described below having
the program called be something like "pev -f example.pdb..." rather than "python main.py -f example.pdb..."

Create a nice cli so that we can do more things like calculate the SES/SAS of
the whole protein and not just the volume and much more. https://docs.python.org/3/library/argparse.html#sub-commands

Handle multiple frames and other file types in atoms class as well as any file
output (dcd and others).

### Known Issues

There is an issue where certain atoms from the PDB file have no radius.
Typically this is the heme and other hydrogen atoms
so for now it is just set to 1.2 angstrom by default.
