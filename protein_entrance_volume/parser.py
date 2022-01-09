"""
Created by: Mitchell Walls
Email: miwalls@siue.edu
"""
import numpy as np
from protein_entrance_volume import atoms


def parse_pdb(pdb_file, outer_residues=None, inner_residues=None):
    """
    Make an array of describing details about every atom noting the
    coordinates, radii, and inner/outer residue boolean.
    """
    allowed_records = {
        "ATOM",
        "HETATM",
    }
    resseqs = []
    coords = []
    radii = []
    with open(pdb_file, 'r', encoding='utf-8') as handle:
        lines = handle.readlines()
        if not lines:
            raise ValueError(f"PDB file '{pdb_file}' is empty.")
    for i, line in enumerate(lines):

        record_type = line[0:6].strip()
        if record_type not in allowed_records:
            continue

        fullname = line[12:16]
        # get rid of whitespace in atom names
        split_list = fullname.split()
        if len(split_list) != 1:
            # atom name has internal spaces, e.g. " N B ", so
            # we do not strip spaces
            name = fullname
        else:
            # atom name is like " CA ", so we can strip spaces
            name = split_list[0]
        resname = line[17:20].strip()
        resseq = int(line[22:26].split()[0])  # sequence identifier

        if record_type == "HETATM":  # hetero atom flag
            if resname in ["WAT", "HOH"]:
                hetero_flag = "W"
            else:
                hetero_flag = "H"
        else:
            hetero_flag = " "

        try:
            x_coord = float(line[30:38])
            y_coord = float(line[38:46])
            z_coord = float(line[46:54])
        except ValueError as value_error:
            raise ValueError(
                f"Invalid or missing coordinate(s) at line {i}."
            ) from value_error
        element = line[76:78].strip().upper()

        coords.append(np.array((x_coord, y_coord, z_coord)))
        resseqs.append(resseq)
        radii.append(
            atoms.get_atom_radius(name, element, resname, hetero_flag)
        )

    coords = np.array(coords)
    resseqs = np.array(resseqs)
    radii = np.array(radii)

    # Generate a boolean array of definining where the outer, inner, and
    # all (outer and inner) residues are located.
    outer_residues_bool = np.in1d(resseqs, outer_residues)
    inner_residues_bool = np.in1d(resseqs, inner_residues)
    all_residues_bool = np.logical_or(outer_residues_bool,
                                      inner_residues_bool)

    return atoms.Protein(coords, radii, outer_residues_bool,
                         inner_residues_bool, all_residues_bool)
