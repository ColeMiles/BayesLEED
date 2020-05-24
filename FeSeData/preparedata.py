import os
import csv
import itertools
import argparse

parser = argparse.ArgumentParser(
    description="Collates beam CSV files into a single file"
)
parser.add_argument("outfile", help="Name of collated file")
parser.add_argument("sourcedir", nargs="?", default=".", help="Directory containing csvs")
parser.add_argument("--smooth", type=int, default=None, help="Smooth the data with a Gaussian kernel of size N")
parser.add_argument("--interpolate", action="store_true", help="If set, interpolates data at 1eV steps")
parser.add_argument("--plottable", action="store_true", help="If set, creates the output file in the same format as run.NormIt for easy plotting")
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

def write_TLEED_format(filename, beamnames, beamdata):
    num_beams = len(beamnames)
    with open(filename, "w") as ofile:
        ofile.write("FeSe (20 u.c.) {} beams\n".format(num_beams))
        beamnums = ""
        for i in range(num_beams):
            beamnums += "{:>3d}".format(i)
        beamnums += "\n"
        ofile.write(beamnums)
        ofile.write("(F7.2,F10.6)\n")

        # Write out the beams! 
        for (beamx, beamy), (E_data, I_data) in zip(beamnames, beamdata):
            ofile.write("({:3.1f},{:3.1f})\n".format(float(beamx), float(beamy)))
            ofile.write("{:4d}{:13.4E}\n".format(len(E_data), 1.0))
            for E, I in zip(E_data, I_data):
                ofile.write("{:7.2f}{:10.6f}\n".format(E, I))


def write_plot_format(filename, beamneams, beamdata):
    max_length = max(len(Es) for (Es, Is) in beamdata) 
    
    # Extend the rest of the I(E) curves with zeros except the max length
    ext_beamdata = []
    for (Es, Is) in beamdata:
        if len(Es) < max_length:
            Es = itertools.chain(Es, itertools.repeat(0))
            Is = itertools.chain(Is, itertools.repeat(0))
        ext_beamdata.append(Es)
        ext_beamdata.append(Is)

    with open(filename, "w") as ofile:
        for dataline in zip(*ext_beamdata):
            for datum in dataline:
                ofile.write("{:>7.2f}".format(datum))
            ofile.write("\n")


datafiles = list(sorted(filter(lambda x: x.endswith("csv"), os.listdir(args.sourcedir))))

beamnames = []
beamdata = []

# Read in / clean data
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

        beamnames.append((float(beamx), float(beamy)))
        beamdata.append((E_data, I_data))

if not args.plottable:
    write_TLEED_format(args.outfile, beamnames, beamdata)
else:
    write_plot_format(args.outfile, beamnames, beamdata)

