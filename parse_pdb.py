#!/usr/bin/env python
import itertools as it


with open("pdbs/prot_heme.pdb", 'r') as pdb:
    i = 0
    for key,group in it.groupby(pdb, lambda line: line.strip().endswith('END')):
        if not key:
            with open("pdbs/prot_heme_{}.pdb".format(i), 'w+') as p:
                p.write(''.join(list(group)))
            print(i)
            i += 1
