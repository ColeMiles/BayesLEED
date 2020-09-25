import os

import pyleed

del_dir = '~/ProgScratch/BayesLEED/TLEED/work/Deltas'

delta_amps = []

for delfile in os.listdir(del_dir):
    fullpath = os.path.join(del_dir, delfile)

    amps = pyleed.tleed.parse_deltas(fullpath)

    delta_amps.append(amps)

