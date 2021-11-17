#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=8

cd /home/miwalls/MEV
/home/miwalls/python/MEV/bin/python main.py -o 107 78 79 230 -i 224 106 108 -r 1.5 -g 0.1 -f "pdbs/prot_heme_$1.pdb" -V
