"""
Created by: Mitchell Walls
Email: miwalls@siue.edu
"""
import numpy as np


class Triangle:
    _faces = None
    _vertices = None

    def __init__(self, grid):
        self._grid = grid

    @property
    def grid(self):
        return self._grid

    @property
    def faces(self):
        if self._faces is None:
            self._faces = calculate_faces()
        return self._faces

    @property
    def vertices(self):
        if self._vertices is None:
            self._vertices = calculate_vertices()
        return self._vertices
