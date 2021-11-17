#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=8

cd /home/miwalls/MEV
for ((i=$1; i<=$2; i++)); do /home/miwalls/python/MEV/bin/python main.py -o 107 78 79 230 -i 224 106 108 -r 1.5 -g 0.1 -f "pdbs/prot_heme_$i.pdb" -V; done
