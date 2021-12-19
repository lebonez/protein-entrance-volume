"""
Created by: Mitchell Walls
Email: miwalls@siue.edu
"""
import warnings
import pandas as pd
import numpy as np
from Bio import BiopythonWarning
from Bio.PDB import PDBParser
from Bio.PDB.ResidueDepth import _get_atom_radius
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from protein_entrance_volume import utils
from protein_entrance_volume import boundary


warnings.filterwarnings("ignore", category=PDBConstructionWarning)
warnings.filterwarnings("ignore", category=BiopythonWarning)


class Protein:
    """
    Build a class with dataframe containing atoms of protein including
    convenience functions and properties.
    """
    _coords = None
    _radii = None
    _or_coords = None
    _ir_coords = None
    _ar_coords = None
    _orc = None
    _irc = None
    _arc = None
    _entrance = None

    def __init__(self, atoms):
        """
        Set dataframe to class. Note to never reference _df outside of this
        class just add needed functions and properties to this class.
        """
        self._df = pd.DataFrame(atoms)

    def __str__(self):
        """
        String function just print the dataframe.
        """
        return str(self._df)

    @property
    def coords(self):
        """
        Convenience property to return a 2D array of atom coordinates
        """
        if self._coords is None:
            self._coords = self._df[['x', 'y', 'z']].to_numpy()
        return self._coords

    @property
    def radii(self):
        """
        Convenience property to return a 1D array of atom radii
        """
        if self._radii is None:
            self._radii = self._df.radius.to_numpy()
        return self._radii

    @property
    def or_coords(self):
        """
        Convenience property to return outer residues coords (or_coords).
        """
        if self._or_coords is None:
            self._or_coords = self._df[
                self._df['outer_residue']][['x', 'y', 'z']].to_numpy()
        return self._or_coords

    @property
    def ir_coords(self):
        """
        Convenience property to return inner residues coords (ir_coords).
        """
        if self._ir_coords is None:
            self._ir_coords = self._df[
                self._df['inner_residue']][['x', 'y', 'z']].to_numpy()
        return self._ir_coords

    @property
    def ar_coords(self):
        """
        Convenience property to return all (inner and outer) residues coords
        (ar_coords).
        """
        if self._ar_coords is None:
            self._ar_coords = np.append(self.or_coords, self.ir_coords, axis=0)
        return self._ar_coords

    @property
    def orc(self):
        """
        Convenience property to return outer residues centroid (orc).
        """
        if self._orc is None:
            self._orc = self.or_coords.mean(axis=0)
        return self._orc

    @property
    def irc(self):
        """
        Convenience property to return inner residues centroid (irc).
        """
        if self._irc is None:
            self._irc = self.ir_coords.mean(axis=0)
        return self._irc

    @property
    def arc(self):
        """
        Convenience property to return all (inner and outer) residues centroid
        (arc).
        """
        if self._arc is None:
            self._arc = (self.orc + self.irc) / 2
        return self._arc

    @property
    def entrance(self):
        """
        Convenience property to return the entrance and also generate the
        entrance if it hasn't been yet. Generate entrance below has more
        parameters that can be tuned.
        """
        if self._entrance is None:
            self.generate_entrance()
        return self._entrance

    @classmethod
    def parse_atoms(cls, pdb_file, structure_id="prot", outer_residues=None,
                    inner_residues=None):
        """
        Make a list of dicts describing details about every atom noting the
        coordinates, radius, residue id, and outer/inner status creating a
        dataframe.
        """
        pdb = PDBParser(PERMISSIVE=False)
        structure = pdb.get_structure(structure_id, pdb_file)
        atoms = []
        for residue in structure[0].get_residues():
            residue_id = residue.get_full_id()[3][1]
            for atom in residue.get_atoms():
                x_coord, y_coord, z_coord = [*atom.get_coord()]
                radius = max(_get_atom_radius(atom), 1.0)
                atoms.append(
                    dict(
                        id=atom.fullname, x=x_coord, y=y_coord, z=z_coord,
                        radius=radius, residue_id=residue_id,
                        outer_residue=residue_id in outer_residues,
                        inner_residue=residue_id in inner_residues,
                    )
                )
        return cls(atoms)

    def residues_mbr(self, extension=0):
        """
        Calculates all of the atoms within a minimum bounding rectangle (mbr)
        determined by the arc and irc. Note: that using the orc rather than the
        irc would give same results. Extension is used to give a value to
        extend past the max radius (i.e. probe_radius).
        """
        # Get the distance between the irc and arc plus the max radius
        distance = np.linalg.norm(self.irc - self.arc)
        distance += (self.radii.max() + extension) * 2

        # Calculate minimum and maximum x, y, z coordinates both as an array
        atoms_min = self.arc - distance
        atoms_max = self.arc + distance

        # Calculate the indices within the mbr.
        indices = utils.inside_mbr(self.coords, atoms_min, atoms_max)
        # Return filtered coordinates, radii, and mbr radius (distance)
        return self.coords[indices], self.radii[indices]

    def generate_entrance(self, no_outer=False, no_inner=False,
                          extension=0, resolution=4):
        """
        Given the outer and inner residues we minimize the problem to a
        bouding rectangle then add in the boundary coordinates. The boundary
        coordinates are made up from an outer larger spherical boundary and the
        outer and inner residue hemispheres.
        """
        # Reduce computation complexity for the grid by only using spheres
        # within the mbr calculated by the inner and outer residue atoms.
        atoms_mbr = self.residues_mbr(extension=extension)

        # how far should each faux sphere be on the boundary hemispheres and
        # sphere. Smaller distance make a slight difference in accuracy but
        # increases computation time significantly.
        points_distance = extension / resolution

        # Generate the larger outer boundary sphere coords
        boundary_sphere = boundary.Sphere(
            self.arc, self.ar_coords, points_distance)
        coords = boundary_sphere.coords
        # No outer means to not create the outer residue faux hemisphere
        # boundary
        if not no_outer:
            # Outer residue boundary hemisphere coords same explanations as
            # above except for only generates on the outer side of the best fit
            # plane of the outer residue atoms.
            outer_hemisphere = boundary.Hemisphere(
                self.orc, self.or_coords, self.irc, points_distance)
            coords = np.vstack((coords, outer_hemisphere.coords))
        # No inner means to not create the inner residue faux spheres boundary
        if not no_inner:
            # Inner residue boundary hemisphere coords same explanations as
            # above except for only generates on the outer side of the best fit
            # plane of the inner residue atoms.
            inner_hemisphere = boundary.Hemisphere(
                self.irc, self.ir_coords, self.orc, points_distance)
            coords = np.vstack((coords, inner_hemisphere.coords))
        # Append probe extended atom radii array with an array of length equal
        # to the total number of boundary spheres filled with values of probe
        # radius.
        radii = np.append(
            atoms_mbr[1] + extension, np.full(coords.shape[0], extension)
        )
        coords = np.vstack((atoms_mbr[0], coords))
        self._entrance = Entrance(
            coords, radii, boundary_sphere, outer_hemisphere, inner_hemisphere)


class Entrance:
    """
    Holds all of the information about the entrance system that is minimized to
    a minimum bounding rectangle and a boundary sphere and outer and
    inner hemispheres.
    """
    _volume = None

    def __init__(self, coords, radii, boundary_sphere, outer_hemisphere,
                 inner_hemisphere):
        """
        Simply assign all the values of the entrance that are useful.
        """
        self._coords = coords
        self._radii = radii
        self._boundary_sphere = boundary_sphere
        self._outer_hemisphere = outer_hemisphere
        self._inner_hemisphere = inner_hemisphere

    @property
    def coords(self):
        """
        Convenience property to return a 2D array of coordinates
        """
        return self._coords

    @property
    def radii(self):
        """
        Convenience property to return a 1D array of radii
        """
        return self._radii

    @property
    def boundary_sphere(self):
        """
        Convenience property to return the outer boundary sphere
        """
        return self._boundary_sphere

    @property
    def outer_hemisphere(self):
        """
        Convenience property to return the outer residue boundary hemisphere
        """
        return self._outer_hemisphere

    @property
    def inner_hemisphere(self):
        """
        Convenience property to return the inner residue boundary hemisphere
        """
        return self._inner_hemisphere
