#!/usr/bin/env python

import open3d as o3d
import sys


def main():
    mesh = o3d.io.read_triangle_mesh(sys.argv[1])
    mesh.paint_uniform_color([1, 0.706, 0])
    o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True, point_show_normal=True)


if __name__ == '__main__':
    main()
