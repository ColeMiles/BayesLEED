""" Python code for interfacing with the TensErLEED code.
"""
import os
import subprocess
import numpy as np

""" The section which will be inserted into FIN with each perturbative change,
      which is ready to be .format()'d with the final coordinates in x (z?)
"""
COORD_SECT = """
  1                       LAY = 1: layer type no. 1 has overlayer lateral periodicity
 10                       number of Bravais sublayers, 1st layer
  1 {:.4f} 1.8955 1.8955  sublayer no. 1 is of site type 1 (La)
  7 {:.4f} 0.0000 0.0000  sublayer no. 2 is of site type 7 (apO)
  4 {:.4f} 0.0000 0.0000  sublayer no. 3 is of site type 4 (Ni)
 10 {:.4f} 1.8955 0.0000  sublayer no. 4 is of site type 10 (eqO)
 10 {:.4f} 0.0000 1.8955  sublayer no. 5 is of site type 10 (eqO)
  2 {:.4f} 1.8955 1.8955  sublayer no. 6 is of site type 2 (La)
  8 {:.4f} 0.0000 0.0000  sublayer no. 7 is of site type 8 (apO)
  5 {:.4f} 0.0000 0.0000  sublayer no. 8 is of site type 5 (Ni)
 11 {:.4f} 1.8955 0.0000  sublayer no. 9 is of site type 11 (eqO)
 11 {:.4f} 0.0000 1.8955  sublayer no.10 is of site type 11 (eqO)
"""[1:] # Remove that initial newline

""" The unperturbed x (z?) coordinates
"""
UNPERTURBED = np.array(
    [0.0000, 0.0000, 1.9500, 1.9500, 1.9500, 3.9000, 3.9000, 5.8500, 5.8500, 5.8500]
)

def write_displacements(leedfile, displacements, idtag):
    """ Edits a TLEED input script to contain updated coordinates.
        displacements should be a np.array of length 8
    """
    # Repeat the displacements that Jordan does to make a length-10 array
    displacements = np.concatenate((
        displacements[0:4],
        [displacements[3]],
        displacements[4:],
        [displacements[-1]]
    ))

    new_coords = np.round(UNPERTURBED + displacements, 4)
    new_coord_sect = COORD_SECT.format(*new_coords)

    # Read in entirety of old script
    with open(leedfile, "r") as initfile:
        oldfile_contents = initfile.readlines()
    # Find line before where new coordinates need to be inserted, as well as the
    #  line which marks the following section
    indbefore, indafter = -1, -1
    for i, line in enumerate(oldfile_contents):
        if line == "-   layer type 1 ---\n":
            indbefore = i+1
        elif line == "-   layer type 2 ---\n":
            indafter = i
    # Check that both lines were found
    if indbefore == -1 or indafter == -1:
        raise ValueError("LEED input file does not contain section marker lines")

    # Write new script as 1st section, then new coords, then 2nd section
    with open(leedfile + str(idtag), "w") as newfile:
        newfile.writelines(oldfile_contents[:indbefore])
        newfile.write(new_coord_sect)
        newfile.writelines(oldfile_contents[indafter:])

def run_command(executable, inputfile):
    return subprocess.run([executable], stdin=open(inputfile, "r"), text=True)

def run_refcalc(directory, idtag):
    """ Given the path to a base directory, makes a new 
         calculation on it, and returns the resulting R-factor.
    """
    newdir = os.path.join(directory, "refcalc"+str(idtag))
    os.mkdir(newdir)
    subprocess.run(["cp", "ref-calc.LaNiO3", newdir])
    # TODO: Finish this
    refcalc_res = run_command()
    # Output is now in fd.out. Feed this to rfactor program, which puts output
    #  in ROUT.
    raise NotImplementedError()

def calc_rfactor(executable, inputfile, idtag):
    raise NotImplementedError()
