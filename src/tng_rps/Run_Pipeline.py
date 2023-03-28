#!/usr/bin/env python3

# wrapper script to run all analysis related to tracking subhalos across time.

### import modules
import illustris_python as il
import multiprocessing as mp
import numpy as np
import h5py
import rohr_utils as ru

global sim, basePath, Header, h
global max_snap, min_snap, SnapNums, Times, BoxSizes
global gas_ptn, dm_ptn, tracer_ptn, star_ptn, bh_ptn

sim = 'L680n8192TNG'
basePath = ru.loadbasePath(sim)

max_snap = 99
min_snap = 0

Header = il.groupcat.loadHeader(basePath, max_snap)
h = Header['HubbleParam']

SnapNums = range(max_snap, min_snap-1, -1)
Times = np.zeros(len(SnapNums), dtype=float)
BoxSizes = np.zeros(len(SnapNums), dtype=float)
for i, SnapNum in enumerate(SnapNums):
    header = il.groupcat.loadHeader(basePath, SnapNum)
    Times[i] = header['Time']
    BoxSizes[i] = header['BoxSize'] * Times[i] / h

gas_ptn = il.util.partTypeNum('gas')
dm_ptn = il.util.partTypeNum('dm')
tracer_ptn = il.util.partTypeNum('tracer')
star_ptn = il.util.partTypeNum('star')
bh_ptn = il.util.partTypeNum('bh')
