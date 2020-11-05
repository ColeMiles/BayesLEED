import copy
import itertools
import pickle

import numpy as np

import pyleed


def main():
    manager = pyleed.bayessearch.create_manager(
        '../../TLEED/work',
        '../../TLEED/',
        pyleed.problems.FESE_BEAMINFO_TRIMMED,
        '../../TLEED/beamlists/NBLIST.FeSe-1x1',
        '../../TLEED/phaseshifts/FeSeBulk.phase', 10,
        executable='ref-calc.FeSe'
    )

    atom_idxs = [0, 5]
    # Displacements (in Angstroms)
    displacements = np.linspace(0.0, 0.5, num=16)

    base_struct = pyleed.problems.FESE_20UC
    all_joint_disps = list(itertools.product(displacements, displacements))

    for i in range(0, len(all_joint_disps), 8):
        structs = []
        for j in range(8):
            struct = copy.deepcopy(base_struct)
            joint_disps = all_joint_disps[i+j]
            for idx, disp in zip(atom_idxs, joint_disps):
                struct.layers[0].zs[idx] += disp / struct.cell_params[2]
            structs.append(struct)
        produce_tensors = i == 0
        manager.batch_ref_calcs(structs, produce_tensors=produce_tensors)
        manager.wait_active_calcs()

    ref_rfactors = np.array([r for (calc, r) in manager.completed_refcalcs])
    np.save('ref_rfactors.npy', ref_rfactors)

    # Do the delta calculations for each one also
    base_calc = manager.completed_refcalcs[0]

    for i in range(0, len(all_joint_disps), 8):
        structs = []
        for j in range(8):
            struct = copy.deepcopy(base_struct)
            joint_disps = all_joint_disps[i+j]
            for idx, disp in zip(atom_idxs, joint_disps):
                struct.layers[0].zs[idx] += disp / struct.cell_params[2]
            structs.append(struct)
        structs = [(s, base_calc) for s in structs]
        manager.batch_delta_calcs(structs)
        manager.wait_active_calcs()

    delta_rfactors = np.array([r for (calc, r) in manager.completed_deltacalcs])
    np.save('delta_rfactors.npy', delta_rfactors)

    structs_rfactors = [
        (calc.struct, r) for (calc, r) in manager.completed_refcalcs
    ]
    delta_structs_rfactors = [
        (calc.struct, r) for (calc, r) in manager.completed_deltacalcs
    ]
    pickle.dump(structs_rfactors, open('ref_structs_rfactors.pkl', 'wb'))
    pickle.dump(delta_structs_rfactors, open('delta_structs_rfactors.pkl', 'wb'))


if __name__ == "__main__":
    main()
