from prody import parseDCD, parsePSF
import numpy as np
import sys


def main():
    dcd = parseDCD(sys.argv[1])
    psf = parsePSF(sys.argv[2])
    for i, f in enumerate(dcd):
        print(i)
        np.savez_compressed(f"files/prot_heme_{i}.npz", f.getCoords())


if __name__ == '__main__':
    main()
