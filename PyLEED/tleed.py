""" Python code for interfacing with the TensErLEED code.
"""
import os
import shutil
import subprocess
import re
import logging
import numpy as np

""" The section which will be inserted into FIN with each perturbative change,
      which is ready to be .format()'d with the final coordinates in z
"""
COORD_SECT = """
  1                       LAY = 1: layer type no. 1 has overlayer lateral periodicity
 10                       number of Bravais sublayers, 1st layer
  1{:>7.4f} 1.8955 1.8955  sublayer no. 1 is of site type 1 (La)
  7{:>7.4f} 0.0000 0.0000  sublayer no. 2 is of site type 7 (apO)
  4{:>7.4f} 0.0000 0.0000  sublayer no. 3 is of site type 4 (Ni)
 10{:>7.4f} 1.8955 0.0000  sublayer no. 4 is of site type 10 (eqO)
 10{:>7.4f} 0.0000 1.8955  sublayer no. 5 is of site type 10 (eqO)
  2{:>7.4f} 1.8955 1.8955  sublayer no. 6 is of site type 2 (La)
  8{:>7.4f} 0.0000 0.0000  sublayer no. 7 is of site type 8 (apO)
  5{:>7.4f} 0.0000 0.0000  sublayer no. 8 is of site type 5 (Ni)
 11{:>7.4f} 1.8955 0.0000  sublayer no. 9 is of site type 11 (eqO)
 11{:>7.4f} 0.0000 1.8955  sublayer no.10 is of site type 11 (eqO)
"""[1:] # Remove that initial newline

""" The unperturbed z coordinates
"""
UNPERTURBED = np.array(
    [0.0000, 0.0000, 1.9500, 1.9500, 1.9500, 3.9000, 3.9000, 5.8500, 5.8500, 5.8500]
)

SOL_RFACTOR = 0.2794

class LEEDManager:
    def __init__(self, basedir, leed_executable, rfactor_executable, exp_datafile, templatefile):
        """ Create a LEEDManager to keep track of TensErLEED components.
                basedir: The base directory to do computation in
                leed_executable: Path to the LEED executable
                rfactor_executable: Path to the rfactor executable
                exp_datafile: Path to the experimental datafile
                templatefile: The base template for the LEED input file (FIN)
        """
        for path in [basedir, leed_executable, rfactor_executable, templatefile]:
            if not os.path.exists(path):
                raise ValueError("File not found: {}".format(path))
        self.basedir = os.path.abspath(basedir)
        self.leed_exe = os.path.abspath(leed_executable)
        self.rfactor_exe = os.path.abspath(rfactor_executable)
        self.exp_datafile = os.path.abspath(exp_datafile)
        # Copy the exp datafile to the working directory if not already there
        copy_exp_datafile = os.path.join(self.basedir, os.path.split(self.exp_datafile)[1])
        try:
            shutil.copyfile(self.exp_datafile, copy_exp_datafile)
        except shutil.SameFileError:
            pass
        with open(templatefile, "r") as f:
            self.input_template = f.readlines()
        self.calc_number = 0

    def _start_calc(self, displacements, calcid):
        """ Start a new process running TLEED, and return the subprocess
              handle and working directory without waiting for it to finish
        """
        newdir = os.path.join(self.basedir, "ref-calc" + str(calcid))
        os.makedirs(newdir, exist_ok=True)
        #shutil.copy(self.exp_datafile, os.path.join(newdir, "WEXPEL"))
        input_filename = os.path.join(newdir, "FIN")
        stdout_filename = os.path.join(newdir, "protocol")
        write_displacements(self.input_template, displacements, input_filename)
        process = subprocess.Popen(
            [self.leed_exe], 
            stdin=open(input_filename, "r"),
            stdout=open(stdout_filename, "w"),
            cwd=newdir,
            text=True
        )
        return process, newdir

    def _rfactor_calc(self, refcalc_outfile):
        """ Runs the r-factor calculation, and parses the output,
             returning the result
        """
        result = subprocess.run(
            [self.rfactor_exe],
            cwd=self.basedir,
            stdin=open(refcalc_outfile, "r"),
            capture_output=True,
            text=True
        )
        return extract_rfactor(result.stdout)

    def ref_calc(self, displacements):
        """ Do the full process of performing a reference calculation.
                displacements: A length 8 np.array of atomic displacements.
            NOTE: Do not call this function in parallel, as there is a race
                condition on self.calc_number. Instead, use batch_ref_calcs.
        """
        self.calc_number += 1
        calc_process, pdir = self._start_calc(displacements, self.calc_number)
        calc_process.wait()
        result_filename = os.path.join(pdir, "fd.out")
        return self._rfactor_calc(result_filename)
        #result = subprocess.run(
        #    [self.rfactor_exe],
        #    stdin=open(result_filename, "r"),
        #    capture_output=True,
        #    text=True
        #)
        #return extract_rfactor(result.stdout)
    
    def batch_ref_calcs(self, displacements):
        """ Run multiple reference calculations in parallel, one for each
             row of displacements.
        """
        if len(displacements.shape) != 2:
            raise ValueError("LEEDManager.batch_ref_calcs expects a 2-dimensional argument")
        num_evals = displacements.shape[0]

        # Start up all of the calculation processes
        logging.info("Starting {} reference calculations...".format(num_evals))
        processes = []
        for i in range(num_evals):
            self.calc_number += 1
            processes.append(self._start_calc(displacements[i], self.calc_number))

        # Wait for all of them to complete, calculate r-factors for each
        # The r-factor calculations are fast enough that we may as well run
        #   them serially
        rfactors = []
        for p, pdir in processes:
            p.wait()
            result_filename = os.path.join(pdir, "fd.out")
            rfactors.append(self._rfactor_calc(result_filename))
            #result = subprocess.run(
            #    [self.rfactor_exe],
            #    cwd=pdir,
            #    stdin=open(result_filename, "r"),
            #    capture_output=True,
            #    text=True
            #)
            #rfactors.append(extract_rfactor(result.stdout))
        logging.info("Reference calculations completed.")
        return np.array(rfactors)


def write_displacements(input_template, displacements, newfilename):
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

    # Find line before where new coordinates need to be inserted, as well as the
    #  line which marks the following section
    indbefore, indafter = -1, -1
    for i, line in enumerate(input_template):
        if line == "-   layer type 1 ---\n":
            indbefore = i+1
        elif line == "-   layer type 2 ---\n":
            indafter = i
    # Check that both lines were found
    if indbefore == -1 or indafter == -1:
        raise ValueError("LEED input file does not contain section marker lines")

    # Write new script as 1st section, then new coords, then 2nd section
    with open(newfilename, "w") as newfile:
        newfile.writelines(input_template[:indbefore])
        newfile.write(new_coord_sect)
        newfile.writelines(input_template[indafter:])

def extract_rfactor(output):
    p = re.compile(r"AVERAGE R-FACTOR =  (\d\.\d+)")
    m = re.search(p, output)
    if m is not None:
        return float(m.group(1))
    else:
        import ipdb
        ipdb.set_trace()
        raise ValueError("No average R-factor line found in input")
