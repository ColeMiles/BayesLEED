import os
import csv
import argparse

parser = argparse.ArgumentParser(
    description="Collates beam CSV files into a single file"
)
parser.add_argument("outfile", help="Name of collated file")
parser.add_argument("sourcedir", nargs="?", default=".", help="Directory containing csvs")
parser.add_argument("--smooth", type=int, default=None, help="Smooth the data with a Gaussian kernel of size N")
parser.add_argument("--interpolate", action="store_true", help="If set, interpolates data at 1eV steps")
args = parser.parse_args()

def find(lst, val):
    for i, lval in enumerate(lst):
        if lval == val:
            return i
    return None

def isclose(a, b, eps=1e-6):
    return abs(a - b) < eps

def interpolate_holes(E_data, I_data, E_step):
    """ Finds holes in the E_data greater than E_step,
         and fills them by linearly interpolating between
         values we have
    """
    i = 1
    while i < len(E_data):
        diff = E_data[i] - E_data[i-1]
        if not isclose(diff, E_step):
            E_data.insert(i, E_data[i-1] + E_step)
            I_data.insert(
                i, 
                I_data[i-1] + (I_data[i] - I_data[i-1]) * E_step / diff
            )
        i += 1
    return E_data, I_data


with open(args.outfile, "w") as ofile:
    datafiles = list(sorted(filter(lambda x: x.endswith("csv"), os.listdir(args.sourcedir))))
    num_beams = len(datafiles)

    ofile.write("FeSe (20 u.c.) {} beams\n".format(num_beams))
    beamnums = ""
    for i in range(num_beams):
        beamnums += "{:>3d}".format(i)
    beamnums += "\n"
    ofile.write(beamnums)
    ofile.write("(F7.2,F10.6)\n")

    for ifilename in sorted(filter(lambda x: x.endswith("csv"), os.listdir(args.sourcedir))):
        filename = os.path.splitext(ifilename)[0]
        beamx, beamy = int(filename[-2]), int(filename[-1])
        with open(os.path.join(args.sourcedir, ifilename), "r") as ifile:
            reader = csv.reader(ifile)

            # Look at header, find correct columns
            header = next(reader)
            E_idx = find(header, "Energy")
            I_idx = find(header, "RawInt")
            if E_idx is None or I_idx is None:
                raise ValueError("Wrong Header!")

            # Read all of the data from the CSV file
            E_data = []
            I_data = []
            for row in reader:
                E_data.append(float(row[E_idx]))
                I_data.append(float(row[I_idx]))

            E_data, I_data = interpolate_holes(E_data, I_data, 2.0)

            # Write back to combined output file
            ofile.write("({:3.1f},{:3.1f})\n".format(float(beamx), float(beamy)))
            ofile.write("{:4d}{:13.4E}\n".format(len(E_data), 1.0))
            for E, I in zip(E_data, I_data):
                ofile.write("{:7.2f}{:10.6f}\n".format(E, I))
