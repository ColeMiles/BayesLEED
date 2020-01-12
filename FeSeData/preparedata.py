import os
import csv
import argparse

parser = argparse.ArgumentParser(
    description="Collates beam CSV files into a single file"
)
parser.add_argument("outfile", help="Name of collated file")
parser.add_argument("sourcedir", nargs="?", default=".", help="Directory containing csvs")
args = parser.parse_args()

def find(lst, val):
    for i, lval in enumerate(lst):
        if lval == val:
            return i
    return None

with open(args.outfile, "w") as ofile:
    ofile.write("FeSe (20 u.c.) 7 beams\n")
    ofile.write("  1  2  3  4  5  6  7\n")
    ofile.write("(F7.2,F10.6)\n")

    for ifilename in sorted(filter(lambda x: x.endswith("csv"), os.listdir(args.sourcedir))):
        filename = os.path.splitext(ifilename)[0]
        beamx, beamy = int(filename[-2]), int(filename[-1])
        with open(ifilename, "r") as ifile:
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

            # Write back to combined output file
            ofile.write("({:3.1f},{:3.1f})\n".format(float(beamx), float(beamy)))
            ofile.write("{:4d}{:13.4E}\n".format(len(E_data), 1.0))
            for E, I in zip(E_data, I_data):
                ofile.write("{:7.2f}{:10.6f}\n".format(E, I))
