"""
Created by: Mitchell Walls
Email: miwalls@siue.edu
"""
from protein_entrance_volume import utils


class SAS:
    def __init__(self, atoms, probe_radius):
        # Find an empty starting voxel near the tip of the outer residue
        # hemisphere. This is the best location since it is the actual
        # beginning point of the entrance.
        starting_voxel = grid.find_empty_voxel(start, stop, normal)

        # Calculate the SAS volume nodes and SAS border nodes using
        # connected components.
        self._nodes, self._borders = utils.connected_components(
            grid.grid, starting_voxel)


class SES:
    def __init__():
        pass
