"""
Created by: Mitchell Walls
Email: miwalls@siue.edu
"""
import pandas as pd
import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.ResidueDepth import _get_atom_radius
from protein_entrance_volume import utils


class Atoms:
    """
    Build a class with dataframe containing atoms including convenience
    functions and properties.
    """
    _coords = None
    _radii = None
    _or_coords = None
    _ir_coords = None
    _orc = None
    _irc = None
    _arc = None

    def __init__(self, atoms):
        """
        Set dataframe to class. Note to never reference _df outside of this
        class just add needed functions and properties to this class.
        """
        self._df = pd.DataFrame(atoms)

    def __str__(self):
        return self._df

    @property
    def coords(self):
        """
        Convenience property to return an 2D array of coordinate pairs
        """
        if self._coords is None:
            self._coords = self._df[['x', 'y', 'z']].to_numpy()
        return self._coords

    @property
    def radii(self):
        """
        Convenience property to return an 1D array of atom radii
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
            self._or_coords = self._df[self._df['outer_residue']][['x', 'y', 'z']].to_numpy()
        return self._or_coords

    @property
    def ir_coords(self):
        """
        Convenience property to return inner residues coords (ir_coords).
        """
        if self._ir_coords is None:
            self._ir_coords = self._df[self._df['inner_residue']][['x', 'y', 'z']].to_numpy()
        return self._ir_coords

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

    @classmethod
    def parse_atoms(cls, pdb_file, structure_id="prot", PERMISSIVE=False, outer_residues=[], inner_residues=[]):
        """
        Make a list of dicts describing details about every atom noting the
        coordinates, radius, residue id, and outer/inner status creating a
        dataframe.
        TODO: This needs to handle multiple frames and other file types
        (dcd and others).
        """
        pdb = PDBParser(PERMISSIVE=PERMISSIVE)
        structure = pdb.get_structure(structure_id, pdb_file)
        atoms = []
        for residue in structure[0].get_residues():
            residue_id = residue.get_full_id()[3][1]
            for atom in residue.get_atoms():
                x, y, z = [*atom.get_coord()]
                atoms.append(
                    dict(
                        id=atom.fullname,
                        x = x, y = y, z = z,
                        # FIXME: Some atoms don't get a radius for now set them to 1
                        radius=_get_atom_radius(atom) if _get_atom_radius(atom) > 1 else 1.0,
                        residue_id=residue_id,
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
        extend past the max radius (i.e. probe_radius)
        """
        # Get the distance between the irc and arc plus the max radius
        distance = np.linalg.norm(self.irc - self.arc)
        distance += self.radii.max() + extension

        # Calculate minimum and maximum x, y, z coordinates both as an array
        min = self.arc - distance
        max = self.arc + distance

        # Calculate the indices within the mbr.
        indices = utils.inside_mbr(self.coords, min, max)
        # Return filtered coordinates, radii, and mbr radius (distance)
        return self.coords[indices], self.radii[indices], distance