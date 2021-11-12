#!/usr/bin/env python
import open3d as o3d
import numpy as np
import sys


def main():
    with open('prot_heme.xyz', 'w+') as xyz:
        for i in range(int(sys.argv[2]) + 1):
            print('{}/prot_heme_{}.pdb'.format(sys.argv[1], i))
            mesh = o3d.io.read_triangle_mesh('{}/prot_heme_{}.pdb'.format(sys.argv[1], i))
            points = np.asarray(mesh.vertices)
            xyz.write('{}\n'.format(points.shape[0]))
            xyz.write('\n')
            for point in points:
                xyz.write('X {} {} {}\n'.format(*point))


if __name__ == '__main__':
    main()
