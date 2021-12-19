"""
Created by: Mitchell Walls
Email: miwalls@siue.edu
"""
import numpy as np
from protein_entrance_volume import exception


def vertices_file(file, verts):
    """
    Takes a filename string and a verts array and dumps the verts to file depending
    on the file extension.
    """
    if file.endswith(".xyz"):
        # Convert SES nodes to coordinates in the original atom coordinate
        # system. Generate an xyz file with atoms called X by far the
        # slowest.
        with open(file, 'w+', encoding='utf-8') \
                as xyz_file:
            xyz_file.write(
                f"{str(verts.shape[0])}\n\n"
            )
            for vert in verts:
                xyz_file.write(f"X {vert[0]} {vert[1]} {vert[2]}\n")
    elif file.endswith(".csv"):
        # CSV file with one vertices x,y,z per row and x,y,z header
        np.savetxt(
            file, verts, header="x,y,z", comments="",
            delimiter=","
        )
    elif file.endswith(".txt"):
        # Dump vertices
        np.savetxt(
            file, verts
        )
    elif file.endswith(".npz"):
        # Dump verts to a npz array file fastest and smallest way ideal
        # for doing post processing of the verts.
        np.savez_compressed(file, verts)
    else:
        raise exception.InvalidFileExtension((".xyz", ".csv", ".txt", ".npz"))
