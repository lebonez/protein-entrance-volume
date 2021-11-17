#!/usr/bin/env python


import glob
import argparse
import re
import pandas as pd


get_frame = re.compile('.*_([0-9]*)\.results')
parser = argparse.ArgumentParser(description='What files to parse.')
parser.add_argument('fileglob', type=str, help="path to file glob as an example 'results/*'.")

args = parser.parse_args()
files = glob.glob(args.fileglob)

files_output = []
for i, file in enumerate(files):
    with open(file, 'r') as fh:
        output = fh.read().strip().split(',')
        files_output.append({'frame': int(get_frame.search(file.split('/')[1]).group(1)), 'mesh_volume': float(output[0]), 'cavity_volume': float(output[1])})

df = pd.DataFrame(files_output)
df.set_index('frame', inplace=True)
df.sort_index(inplace=True)
df.to_csv('results_prot_heme.csv')
