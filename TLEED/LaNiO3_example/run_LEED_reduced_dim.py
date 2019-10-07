import os
import numpy as np

def Write_Crystal_Structure_reduced_dim(displacements, IDtag):
   DIR = "/home/jordan/bayesopt/TLEED/LaNiO3_example/"
   filepath = DIR + "run.ref-calc"

   f = open(filepath, "r")
   contents = f.readlines()
   f.close()

   for i in range(len(contents)):
      if contents[i].strip() == "# Insert coordinates here:":
         insertion_line_num = i
         break

   new_lines =  []
   new_lines.append("  1 " + f"{np.round(0.0000 + displacements[0],4):.4f}" + " 1.8955 1.8955  sublayer no. 1 is of site type 1 (La)\n")
   new_lines.append("  7 " + f"{np.round(0.0000 + displacements[1],4):.4f}" + " 0.0000 0.0000  sublayer no. 2 is of site type 7 (apO)\n")
   new_lines.append("  4 " + f"{np.round(1.9500 + displacements[2],4):.4f}" + " 0.0000 0.0000  sublayer no. 3 is of site type 4 (Ni)\n")
   new_lines.append(" 10 " + f"{np.round(1.9500 + displacements[3],4):.4f}" + " 1.8955 0.0000  sublayer no. 4 is of site type 10 (eqO)\n")
   new_lines.append(" 10 " + f"{np.round(1.9500 + displacements[3],4):.4f}" + " 0.0000 1.8955  sublayer no. 5 is of site type 10 (eqO)\n")
   new_lines.append("  2 " + f"{np.round(3.9000 + displacements[4],4):.4f}" + " 1.8955 1.8955  sublayer no. 6 is of site type 2 (La)\n")
   new_lines.append("  8 " + f"{np.round(3.9000 + displacements[5],4):.4f}" + " 0.0000 0.0000  sublayer no. 7 is of site type 8 (apO)\n")
   new_lines.append("  5 " + f"{np.round(5.8500 + displacements[6],4):.4f}" + " 0.0000 0.0000  sublayer no. 8 is of site type 5 (Ni)\n")
   new_lines.append(" 11 " + f"{np.round(5.8500 + displacements[7],4):.4f}" + " 1.8955 0.0000  sublayer no. 9 is of site type 11 (eqO)\n")
   new_lines.append(" 11 " + f"{np.round(5.8500 + displacements[7],4):.4f}" + " 0.0000 1.8955  sublayer no.10 is of site type 11 (eqO)\n")

   newfilepath = DIR + 'Ref-calc/run.ref-calc-' + IDtag
   f = open(newfilepath, "w")
   f.writelines(contents[:insertion_line_num] + new_lines + contents[insertion_line_num+1:])
   f.close()

def Run_LEED_reduced_dim(coord_perturbs, IDtag):

   DIR = "/home/jordan/bayesopt/TLEED/LaNiO3_example/"

   # Run reference calculation
   Write_Crystal_Structure_reduced_dim(coord_perturbs, IDtag)
   os.system('chmod +x ' + DIR + 'Ref-calc/run.ref-calc-' + IDtag)
   run_ref_calc_cmd = DIR + 'Ref-calc/run.ref-calc-' + IDtag + " " + IDtag
   os.system(run_ref_calc_cmd)
   
   # Compute r-factor
   run_r_factor_cmd = DIR + 'run.r-factor ' + IDtag
   os.system(run_r_factor_cmd)

   # Get r-factor from file
   filepath = DIR + 'R-factor_out/rf-out.LaNiO3-optStruct-' + IDtag
   with open(filepath) as f:
      lines = f.read().splitlines()
   
   rfactor = float(lines[-1].split('AVERAGE R-FACTOR = ')[1].strip() )
   return rfactor
