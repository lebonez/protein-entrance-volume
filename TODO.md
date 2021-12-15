### Critical
Possible that the fill holes part of main after the
volume grid is generated could miss multiple holes. This is possible for larger
volumes with lots of weird angles. Only way I can think of fixing is to
calculate the volume using a triangle mesh calculated by the SES vertices or
using the much slower binary fill holes algorithm.

### Optimizations and Cleanups
Sphere number points could be a single equation and not a while loop.

The outer boundary outside of the residues half spheres should be re-engineered.

Boundary half spheres can be optimized.

Boundary spheres calculation in main is kind of messy could be cleared up.

### New Features

Handle multiple frames and other file types in atoms class as well as any file
output (dcd and others).

Grid needs more from_{shape} classmethods using rasterize.{shape} files.

Maybe move away from biopython and pandas that way we only have two dependencies
being numba and numpy.
