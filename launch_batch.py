#!/usr/bin/env python
import numpy as np
import subprocess as sp
import sys

prev=-1
lenge = 1
last_frame_number = int(sys.argv[1])
for i in np.arange(0,last_frame_number,500):
    sp.run(['sbatch', 'slurm.sh', str(prev + 1), str(i)])
    prev=i
sp.run(['sbatch', 'slurm.sh', str(prev + 1), str(last_frame_number)])
