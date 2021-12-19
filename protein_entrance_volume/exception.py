"""
Created by: Mitchell Walls
Email: miwalls@siue.edu
"""
class OutOfBounds(Exception):
    """
    Error for if the connected components search ends up out of c-ordered 1D
    boolean array bounds.
    """
    def __init__(self):
        self.message = \
        """
        Error: connected components algorithm ended up out of bounds. Typically
        changing grid or probe size can help this. Double check your residue
        IDs as well.
        """
        super().__init__(self.message)


class StartingVoxelNotFound(Exception):
    """
    Algorithm must find a good starting voxel to begin the connected components
    search this is the error if it can't.
    """
    def __init__(self):
        self.message = \
        """
        Error: Starting voxel was not found for the connected component search.
        Typically changing grid or probe size can help this. Double check your
        residue IDs as well.
        """
        super().__init__(self.message)


class InvalidFileExtension(Exception):
    """
    Error for when an incorrect vertices out file extension is used.
    """
    def __init__(self, extensions):
        self.message = \
        f"""
        Error: the file extension must be one of ({(', ').join(extensions)}).
        """
        super().__init__(self.message)


class InvalidPlotType(Exception):
    """
    Error for when an incorrect vertices out file extension is used.
    """
    def __init__(self, plot_types):
        self.message = \
        f"""
        Error: the plot type must be one of ({(', ').join(extensions)}).
        """
        super().__init__(self.message)
