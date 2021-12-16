"""
Created by: Mitchell Walls
Email: miwalls@siue.edu
"""
class OutOfBounds(Exception):

    def __init__(self):
        self.message = \
        """
        Error: connected components algorithm ended up out of bounds. Typically
        changing grid or probe size can help this. Double check your residue
        IDs as well.
        """
        super().__init__(self.message)


class StartingVoxelNotFound(Exception):

    def __init__(self):
        self.message = \
        """
        Error: Starting voxel was not found for the connected component search.
        Typically changing grid or probe size can help this. Double check your
        residue IDs as well.
        """
        super().__init__(self.message)


class InvalidFileExtension(Exception):
    def __init__(self, extensions):
        self.message = \
        """
        Error: the file extension must be one of ({}).
        """.format((', ').join(extensions))
        super().__init__(self.message)
