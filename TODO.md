### Cleanup or Optimizations

Grid SAS and SES classes are messy are not clear/generic enough to be
considered pythonic.

Add in some abstract inheritance classes to clean up repetitive code in classes.

### New Features

Handle multiple frames and other file types in atoms class as well as any file
output (dcd and others).

### Known Issues

There is an issue where certain atoms from the PDB file have no radius.
Typically this is the heme and other hydrogen atoms
so for now it is just set to 1.2 angstrom by default.
