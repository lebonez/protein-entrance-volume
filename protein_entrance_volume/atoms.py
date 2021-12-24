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


# warnings.filterwarnings("ignore", category=PDBConstructionWarning)
# warnings.filterwarnings("ignore", category=BiopythonWarning)


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
                radius = max(get_atom_radius(atom), 1.0)
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

_atomic_radii = {
    #   atom num dist  Rexplicit Runited-atom
    1: (0.57, 1.40, 1.40),
    2: (0.66, 1.40, 1.60),
    3: (0.57, 1.40, 1.40),
    4: (0.70, 1.54, 1.70),
    5: (0.70, 1.54, 1.80),
    6: (0.70, 1.54, 2.00),
    7: (0.77, 1.74, 2.00),
    8: (0.77, 1.74, 2.00),
    9: (0.77, 1.74, 2.00),
    10: (0.67, 1.74, 1.74),
    11: (0.70, 1.74, 1.86),
    12: (1.04, 1.80, 1.85),
    13: (1.04, 1.80, 1.80),  # P, S, and LonePairs
    14: (0.70, 1.54, 1.54),  # non-protonated nitrogens
    15: (0.37, 1.20, 1.20),  # H, D  hydrogen and deuterium
    16: (0.70, 0.00, 1.50),  # obsolete entry, purpose unknown
    17: (3.50, 5.00, 5.00),  # pseudoatom - big ball
    18: (1.74, 1.97, 1.97),  # Ca calcium
    19: (1.25, 1.40, 1.40),  # Zn zinc    (traditional radius)
    20: (1.17, 1.40, 1.40),  # Cu copper  (traditional radius)
    21: (1.45, 1.30, 1.30),  # Fe heme iron
    22: (1.41, 1.49, 1.49),  # Cd cadmium
    23: (0.01, 0.01, 0.01),  # pseudoatom - tiny dot
    24: (0.37, 1.20, 0.00),  # hydrogen vanishing if united-atoms
    25: (1.16, 1.24, 1.24),  # Fe not in heme
    26: (1.36, 1.60, 1.60),  # Mg magnesium
    27: (1.17, 1.24, 1.24),  # Mn manganese
    28: (1.16, 1.25, 1.25),  # Co cobalt
    29: (1.17, 2.15, 2.15),  # Se selenium
    30: (3.00, 3.00, 3.00),  # obsolete entry, original purpose unknown
    31: (1.15, 1.15, 1.15),  # Yb ytterbium +3 ion --- wild guess only
    38: (0.95, 1.80, 1.80),  # obsolete entry, original purpose unknown
}


def get_atom_radius(atom, rtype="united"):
    """Translate an atom object to an atomic radius defined in MSMS (PRIVATE).
    Uses information from the parent residue and the atom object to define
    the atom type.
    Returns the radius (float) according to the selected type:
     - explicit (reads hydrogens)
     - united (default)
    """
    if rtype == "explicit":
        typekey = 1
    elif rtype == "united":
        typekey = 2
    else:
        raise ValueError(
            f"Radius type ({rtype!r}) not understood. Must be 'explicit' or 'united'"
        )

    resname = atom.parent.resname
    het_atm = atom.parent.id[0]

    at_name = atom.name
    at_elem = atom.element

    # Hydrogens
    if at_elem == "H" or at_elem == "D":
        return _atomic_radii[15][typekey]
    # HETATMs
    elif het_atm == "W" and at_elem == "O":
        return _atomic_radii[2][typekey]
    elif het_atm != " " and at_elem == "CA":
        return _atomic_radii[18][typekey]
    elif het_atm != " " and at_elem == "CD":
        return _atomic_radii[22][typekey]
    elif resname == "ACE" and at_name == "CA":
        return _atomic_radii[9][typekey]
    # Main chain atoms
    elif at_name == "N":
        return _atomic_radii[4][typekey]
    elif at_name == "CA":
        return _atomic_radii[7][typekey]
    elif at_name == "C":
        return _atomic_radii[10][typekey]
    elif at_name == "O":
        return _atomic_radii[1][typekey]
    elif at_name == "P":
        return _atomic_radii[13][typekey]
    # CB atoms
    elif at_name == "CB" and resname == "ALA":
        return _atomic_radii[9][typekey]
    elif at_name == "CB" and resname in {"ILE", "THR", "VAL"}:
        return _atomic_radii[7][typekey]
    elif at_name == "CB":
        return _atomic_radii[8][typekey]
    # CG atoms
    elif at_name == "CG" and resname in {
        "ASN",
        "ASP",
        "ASX",
        "HIS",
        "HIP",
        "HIE",
        "HID",
        "HISN",
        "HISL",
        "LEU",
        "PHE",
        "TRP",
        "TYR",
    }:
        return _atomic_radii[10][typekey]
    elif at_name == "CG" and resname == "LEU":
        return _atomic_radii[7][typekey]
    elif at_name == "CG":
        return _atomic_radii[8][typekey]
    # General amino acids in alphabetical order
    elif resname == "GLN" and at_elem == "O":
        return _atomic_radii[3][typekey]
    elif resname == "ACE" and at_name == "CH3":
        return _atomic_radii[9][typekey]
    elif resname == "ARG" and at_name == "CD":
        return _atomic_radii[8][typekey]
    elif resname == "ARG" and at_name in {"NE", "RE"}:
        return _atomic_radii[4][typekey]
    elif resname == "ARG" and at_name == "CZ":
        return _atomic_radii[10][typekey]
    elif resname == "ARG" and at_name.startswith(("NH", "RH")):
        return _atomic_radii[5][typekey]
    elif resname == "ASN" and at_name == "OD1":
        return _atomic_radii[1][typekey]
    elif resname == "ASN" and at_name == "ND2":
        return _atomic_radii[5][typekey]
    elif resname == "ASN" and at_name.startswith("AD"):
        return _atomic_radii[3][typekey]
    elif resname == "ASP" and at_name.startswith(("OD", "ED")):
        return _atomic_radii[3][typekey]
    elif resname == "ASX" and at_name.startswith("OD1"):
        return _atomic_radii[1][typekey]
    elif resname == "ASX" and at_name == "ND2":
        return _atomic_radii[3][typekey]
    elif resname == "ASX" and at_name.startswith(("OD", "AD")):
        return _atomic_radii[3][typekey]
    elif resname in {"CYS", "CYX", "CYM"} and at_name == "SG":
        return _atomic_radii[13][typekey]
    elif resname in {"CYS", "MET"} and at_name.startswith("LP"):
        return _atomic_radii[13][typekey]
    elif resname == "CUH" and at_name == "SG":
        return _atomic_radii[12][typekey]
    elif resname == "GLU" and at_name.startswith(("OE", "EE")):
        return _atomic_radii[3][typekey]
    elif resname in {"GLU", "GLN", "GLX"} and at_name == "CD":
        return _atomic_radii[10][typekey]
    elif resname == "GLN" and at_name == "OE1":
        return _atomic_radii[1][typekey]
    elif resname == "GLN" and at_name == "NE2":
        return _atomic_radii[5][typekey]
    elif resname in {"GLN", "GLX"} and at_name.startswith("AE"):
        return _atomic_radii[3][typekey]
    # Histdines and friends
    # There are 4 kinds of HIS rings: HIS (no protons), HID (proton on Delta),
    #   HIE (proton on epsilon), and HIP (protons on both)
    # Protonated nitrogens are numbered 4, else 14
    # HIS is treated here as the same as HIE
    #
    # HISL is a deprotonated HIS (the L means liganded)
    elif resname in {"HIS", "HID", "HIE", "HIP", "HISL"} and at_name in {"CE1", "CD2"}:
        return _atomic_radii[11][typekey]
    elif resname in {"HIS", "HID", "HIE", "HISL"} and at_name == "ND1":
        return _atomic_radii[14][typekey]
    elif resname in {"HID", "HIP"} and at_name in {"ND1", "RD1"}:
        return _atomic_radii[4][typekey]
    elif resname in {"HIS", "HIE", "HIP"} and at_name in {"NE2", "RE2"}:
        return _atomic_radii[4][typekey]
    elif resname in {"HID", "HISL"} and at_name in {"NE2", "RE2"}:
        return _atomic_radii[14][typekey]
    elif resname in {"HIS", "HID", "HIP", "HISL"} and at_name.startswith(("AD", "AE")):
        return _atomic_radii[4][typekey]
    # More amino acids
    elif resname == "ILE" and at_name == "CG1":
        return _atomic_radii[8][typekey]
    elif resname == "ILE" and at_name == "CG2":
        return _atomic_radii[9][typekey]
    elif resname == "ILE" and at_name in {"CD", "CD1"}:
        return _atomic_radii[9][typekey]
    elif resname == "LEU" and at_name.startswith("CD"):
        return _atomic_radii[9][typekey]
    elif resname == "LYS" and at_name in {"CG", "CD", "CE"}:
        return _atomic_radii[8][typekey]
    elif resname == "LYS" and at_name in {"NZ", "KZ"}:
        return _atomic_radii[6][typekey]
    elif resname == "MET" and at_name == "SD":
        return _atomic_radii[13][typekey]
    elif resname == "MET" and at_name == "CE":
        return _atomic_radii[9][typekey]
    elif resname == "PHE" and at_name.startswith(("CD", "CE", "CZ")):
        return _atomic_radii[11][typekey]
    elif resname == "PRO" and at_name in {"CG", "CD"}:
        return _atomic_radii[8][typekey]
    elif resname == "CSO" and at_name in {"SE", "SEG"}:
        return _atomic_radii[9][typekey]
    elif resname == "CSO" and at_name.startswith("OD"):
        return _atomic_radii[3][typekey]
    elif resname == "SER" and at_name == "OG":
        return _atomic_radii[2][typekey]
    elif resname == "THR" and at_name == "OG1":
        return _atomic_radii[2][typekey]
    elif resname == "THR" and at_name == "CG2":
        return _atomic_radii[9][typekey]
    elif resname == "TRP" and at_name == "CD1":
        return _atomic_radii[11][typekey]
    elif resname == "TRP" and at_name in {"CD2", "CE2"}:
        return _atomic_radii[10][typekey]
    elif resname == "TRP" and at_name == "NE1":
        return _atomic_radii[4][typekey]
    elif resname == "TRP" and at_name in {"CE3", "CZ2", "CZ3", "CH2"}:
        return _atomic_radii[11][typekey]
    elif resname == "TYR" and at_name in {"CD1", "CD2", "CE1", "CE2"}:
        return _atomic_radii[11][typekey]
    elif resname == "TYR" and at_name == "CZ":
        return _atomic_radii[10][typekey]
    elif resname == "TYR" and at_name == "OH":
        return _atomic_radii[2][typekey]
    elif resname == "VAL" and at_name in {"CG1", "CG2"}:
        return _atomic_radii[9][typekey]
    elif at_name in {"CD", "CD"}:
        return _atomic_radii[8][typekey]
    # Co-factors, and other weirdos
    elif (
        resname in {"FS3", "FS4"}
        and at_name.startswith("FE")
        and at_name.endswith(("1", "2", "3", "4", "5", "6", "7"))
    ):
        return _atomic_radii[21][typekey]
    elif (
        resname in {"FS3", "FS4"}
        and at_name.startswith("S")
        and at_name.endswith(("1", "2", "3", "4", "5", "6", "7"))
    ):
        return _atomic_radii[13][typekey]
    elif resname == "FS3" and at_name == "OXO":
        return _atomic_radii[1][typekey]
    elif resname == "FEO" and at_name in {"FE1", "FE2"}:
        return _atomic_radii[21][typekey]
    elif resname == "HEM" and at_name in {"O1", "O2"}:
        return _atomic_radii[1][typekey]
    elif resname == "HEM" and at_name == "FE":
        return _atomic_radii[21][typekey]
    elif resname == "HEM" and at_name in {
        "CHA",
        "CHB",
        "CHC",
        "CHD",
        "CAB",
        "CAC",
        "CBB",
        "CBC",
    }:
        return _atomic_radii[11][typekey]
    elif resname == "HEM" and at_name in {
        "NA",
        "NB",
        "NC",
        "ND",
        "N A",
        "N B",
        "N C",
        "N D",
    }:
        return _atomic_radii[14][typekey]
    elif resname == "HEM" and at_name in {
        "C1A",
        "C1B",
        "C1C",
        "C1D",
        "C2A",
        "C2B",
        "C2C",
        "C2D",
        "C3A",
        "C3B",
        "C3C",
        "C3D",
        "C4A",
        "C4B",
        "C4C",
        "C4D",
        "CGA",
        "CGD",
    }:
        return _atomic_radii[10][typekey]
    elif resname == "HEM" and at_name in {"CMA", "CMB", "CMC", "CMD"}:
        return _atomic_radii[9][typekey]
    elif resname == "HEM" and at_name == "OH2":
        return _atomic_radii[2][typekey]
    elif resname == "AZI" and at_name in {"N1", "N2", "N3"}:
        return _atomic_radii[14][typekey]
    elif resname == "MPD" and at_name in {"C1", "C5", "C6"}:
        return _atomic_radii[9][typekey]
    elif resname == "MPD" and at_name == "C2":
        return _atomic_radii[10][typekey]
    elif resname == "MPD" and at_name == "C3":
        return _atomic_radii[8][typekey]
    elif resname == "MPD" and at_name == "C4":
        return _atomic_radii[7][typekey]
    elif resname == "MPD" and at_name in {"O7", "O8"}:
        return _atomic_radii[2][typekey]
    elif resname in {"SO4", "SUL"} and at_name == "S":
        return _atomic_radii[13][typekey]
    elif resname in {"SO4", "SUL", "PO4", "PHO"} and at_name in {
        "O1",
        "O2",
        "O3",
        "O4",
    }:
        return _atomic_radii[3][typekey]
    elif resname == "PC " and at_name in {"O1", "O2", "O3", "O4"}:
        return _atomic_radii[3][typekey]
    elif resname == "PC " and at_name == "P1":
        return _atomic_radii[13][typekey]
    elif resname == "PC " and at_name in {"C1", "C2"}:
        return _atomic_radii[8][typekey]
    elif resname == "PC " and at_name in {"C3", "C4", "C5"}:
        return _atomic_radii[9][typekey]
    elif resname == "PC " and at_name == "N1":
        return _atomic_radii[14][typekey]
    elif resname == "BIG" and at_name == "BAL":
        return _atomic_radii[17][typekey]
    elif resname in {"POI", "DOT"} and at_name in {"POI", "DOT"}:
        return _atomic_radii[23][typekey]
    elif resname == "FMN" and at_name in {"N1", "N5", "N10"}:
        return _atomic_radii[4][typekey]
    elif resname == "FMN" and at_name in {
        "C2",
        "C4",
        "C7",
        "C8",
        "C10",
        "C4A",
        "C5A",
        "C9A",
    }:
        return _atomic_radii[10][typekey]
    elif resname == "FMN" and at_name in {"O2", "O4"}:
        return _atomic_radii[1][typekey]
    elif resname == "FMN" and at_name == "N3":
        return _atomic_radii[14][typekey]
    elif resname == "FMN" and at_name in {"C6", "C9"}:
        return _atomic_radii[11][typekey]
    elif resname == "FMN" and at_name in {"C7M", "C8M"}:
        return _atomic_radii[9][typekey]
    elif resname == "FMN" and at_name.startswith(("C1", "C2", "C3", "C4", "C5")):
        return _atomic_radii[8][typekey]
    elif resname == "FMN" and at_name.startswith(("O2", "O3", "O4")):
        return _atomic_radii[2][typekey]
    elif resname == "FMN" and at_name.startswith("O5"):
        return _atomic_radii[3][typekey]
    elif resname == "FMN" and at_name in {"OP1", "OP2", "OP3"}:
        return _atomic_radii[3][typekey]
    elif resname in {"ALK", "MYR"} and at_name == "OT1":
        return _atomic_radii[3][typekey]
    elif resname in {"ALK", "MYR"} and at_name == "C01":
        return _atomic_radii[10][typekey]
    elif resname == "ALK" and at_name == "C16":
        return _atomic_radii[9][typekey]
    elif resname == "MYR" and at_name == "C14":
        return _atomic_radii[9][typekey]
    elif resname in {"ALK", "MYR"} and at_name.startswith("C"):
        return _atomic_radii[8][typekey]
    # Metals
    elif at_elem == "CU":
        return _atomic_radii[20][typekey]
    elif at_elem == "ZN":
        return _atomic_radii[19][typekey]
    elif at_elem == "MN":
        return _atomic_radii[27][typekey]
    elif at_elem == "FE":
        return _atomic_radii[25][typekey]
    elif at_elem == "MG":
        return _atomic_radii[26][typekey]
    elif at_elem == "CO":
        return _atomic_radii[28][typekey]
    elif at_elem == "SE":
        return _atomic_radii[29][typekey]
    elif at_elem == "YB":
        return _atomic_radii[31][typekey]
    # Others
    elif at_name == "SEG":
        return _atomic_radii[9][typekey]
    elif at_name == "OXT":
        return _atomic_radii[3][typekey]
    # Catch-alls
    elif at_name.startswith(("OT", "E")):
        return _atomic_radii[3][typekey]
    elif at_name.startswith("S"):
        return _atomic_radii[13][typekey]
    elif at_name.startswith("C"):
        return _atomic_radii[7][typekey]
    elif at_name.startswith("A"):
        return _atomic_radii[11][typekey]
    elif at_name.startswith("O"):
        return _atomic_radii[1][typekey]
    elif at_name.startswith(("N", "R")):
        return _atomic_radii[4][typekey]
    elif at_name.startswith("K"):
        return _atomic_radii[6][typekey]
    elif at_name in {"PA", "PB", "PC", "PD"}:
        return _atomic_radii[13][typekey]
    elif at_name.startswith("P"):
        return _atomic_radii[13][typekey]
    elif resname in {"FAD", "NAD", "AMX", "APU"} and at_name.startswith("O"):
        return _atomic_radii[1][typekey]
    elif resname in {"FAD", "NAD", "AMX", "APU"} and at_name.startswith("N"):
        return _atomic_radii[4][typekey]
    elif resname in {"FAD", "NAD", "AMX", "APU"} and at_name.startswith("C"):
        return _atomic_radii[7][typekey]
    elif resname in {"FAD", "NAD", "AMX", "APU"} and at_name.startswith("P"):
        return _atomic_radii[13][typekey]
    elif resname in {"FAD", "NAD", "AMX", "APU"} and at_name.startswith("H"):
        return _atomic_radii[15][typekey]
    else:
        warnings.warn(f"{at_name}:{resname} not in radii library.", BiopythonWarning)
        return 0.01
