#!/usr/bin/env python3

# define all global variables used in analysis

import rohr_utils as ru
import illustris_python as il
import numpy as np
import os

def globals():
    """
    define all global variables here, and then import in other scripts
    for now, this is hand-coded, but eventually can be updated to use
    a command line parser or parameter file.
    """
    global sim, basePath, outdirec, outfname
    global TNGCluster_flag, mp_flag, zooniverse_flag, centrals_flag
    global max_snap, min_snap, Header, h, SnapNums, Times, BoxSizes
    global gas_ptn, dm_ptn, tracer_ptn, star_ptn, bh_ptn
    
    sim = 'L680n8192TNG'
    basePath = ru.loadbasePath(sim)

    TNGCluster_flag = True
    mp_flag = True
    zooniverse_flag = False
    centrals_flag = False
        
    outdirec, outfname = return_outdirec_outfname()
        
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
    
    return

def return_outdirec_outfname():
    """
    given the simulation and flags, determine the directory and filename
    """
    
    if (zooniverse_flag):
        ins_key = 'inspected'
        outfname = 'zooniverse_%s_%s_branches.hdf5'%(sim, ins_key)
        outdirec = '../Output/%s_subfindGRP/'%sim
        
        if (os.path.isdir(outdirec)):
            if os.path.isfile(outdirec + outfname):
                raise ValueError('Warning %s file exists.'%(outdirec+outfname))
            else:
                return outdirec, outfname
        else:
            print('Directory %s does not exist. Creating it now.'%outdirec)
            os.system('mkdir %s'%outdirec)
            return outdirec, outfname
            
    if centrals_flag:
        outfname = 'central_subfind_%s_branches.hdf5'%(sim)
    else:
        outfname = 'subfind_%s_branches.hdf5'%(sim)
    
    outdirec = '../Output/%s_subfindGRP/'%sim
    
    if (os.path.isdir(outdirec)):
        print('Directory %s already exists.'%outdirec)
        if os.path.isfile(outdirec + outfname):
            raise ValueError('Warning %s file exists.'%(outdirec+outfname))
        else:
            return outdirec, outfname
    else:
        print('Directory %s does not exist. Creating it now.'%outdirec)
        os.system('mkdir %s'%outdirec)
        return outdirec, outfname
            
