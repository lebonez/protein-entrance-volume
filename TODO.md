### New Features

Handle multiple frames and other file types in atoms class as well as any file
output (dcd and others).

Maybe move away from biopython and pandas that way we only have two dependencies
being numba and numpy. Should only need a table of radii to replace
biopython. Pandas would just require some neat python trickery.

### Known Issues

There is an issue where certain atoms from the PDB file have no radius.
Typically this is the heme so for now it is just set to 1 angstrom by default.
