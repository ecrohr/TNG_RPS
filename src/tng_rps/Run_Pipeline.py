#!/usr/bin/env python3

# wrapper script to run all analysis related to tracking subhalos across time.

### import modules
import illustris_python as il
import multiprocessing as mp
import numpy as np
import h5py
import rohr_utils as ru

# import globals
import globals
globals.globals()

# create the indices
if globals.SubfindIndices:
    from Create_SubfindIndices import run_subfindindices
    run_subfindindices()

# create the snapshot flags
if globals.SubfindSnapshot:
    from Create_SubfindSnapshot_Flags import run_subfindsnapshot_flags
    run_subfindsnapshot_flags()

# run the gas radial profile calculation
if globals.SubfindGasRadProf:
    from Create_SubfindGasRadProf import run_subfindGRP
    run_subfindGRP()

