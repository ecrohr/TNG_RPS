#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 12:02:49 2021

@author: rohr
"""

### import modules
import illustris_python as il
import multiprocessing as mp
import numpy as np
import h5py
import rohr_utils as ru
from itertools import repeat
import glob

def run_subfindindices(Config):
    """
    run the Create_SubfindIndices module
    """
    
    SnapNum = Config.SnapNums_SubfindIDs
    SubfindID = Config.SubfindIDs

    print(SnapNum.size, SubfindID.size)

    # run return_subfindindices
    if Config.mp_flag:
        pool = mp.Pool(Config.Nmpcores) # should be 8 if running interactively
        result_list = pool.starmap(return_subfindindices, zip(SnapNum,
                                                              SubfindID,
                                                              repeat(Config)))
    else:
        result_list = []
        for index, subfindID in enumerate(SubfindID):
            result_list.append(return_subfindindices(SnapNum[index],
                                                     subfindID, Config))

    # reformat result and save
    result = {}
    for d in result_list:
        result.update(d)
                
    with h5py.File(Config.outdirec + Config.outfname, 'a') as outf:
        for group_key in result.keys():
            group = outf.require_group(group_key)
            for dset_key in result[group_key].keys():
                dset = result[group_key][dset_key]
                dataset = group.require_dataset(dset_key, shape=dset.shape, dtype=dset.dtype)
                dataset[:] = dset
            
        outf.close()
    
    return


def return_subfindindices(snap, subfindID, Config):
    """
    Given the snap and subfindID, load the MPB/MDB between min_snap and max_snap.
    Then record various properties at each of these snaps.
    Returns dictionary of these properties.
    """

    # initialize results
    SnapNums = Config.SnapNums
    sim = Config.sim
    basePath = Config.basePath
    Times = Config.Times
    BoxSizes = Config.BoxSizes
    h = Config.h
    max_snap = Config.max_snap
    min_snap = Config.min_snap
    treeName = Config.treeName
    star_ptn = Config.star_ptn
    dm_ptn = Config.dm_ptn
    
    return_key = '%03d_%08d'%(snap, subfindID)

    result_keys = ['SubfindID',
                   'Subhalo_Mstar_Rgal',
                   'SubhaloStellarPhotometrics',
                   'Subhalo_Rgal',
                   'SubhaloGrNr', 'SubGroupFirstSub',
                   'SubhaloParent',
                   'SubhaloSFR', 'SubhaloSFRinRad',
                   'SubhaloMass', 'SubhaloDMMass',
                   'SubhaloPos', 'SubhaloBHMdot',
                   'SubhaloBHMass', 'SubhaloSpin',
                   'SubhaloVel', 'HostSubhaloVel',
                   'HostSubhaloPos', 'HostSubfindID',
                   'HostSubhalo_Mstar_Rgal',
                   'HostSubalo_Rgal',
                   'HostGroup_M_Crit200',
                   'HostGroup_R_Crit200',
                   'HostSubhaloGrNr',
                   'HostCentricDistance_phys', 'HostCentricDistance_norm']
    
    result = {}
    result[return_key] = {}

    int_keys = ['SubfindID',
                'SubhaloGrNr', 'SubhaloGroupFirstSub',
                'SubhaloParent',
                'HostSubfindID', 'HostSubhaloGrNr']

    threed_keys = ['SubhaloPos', 'SubhaloVel', 'SubhaloSpin',
                   'HostSubhaloVel', 'HostSubhaloPos']
    
    eightd_keys = ['SubhaloStellarPhotometrics']

    for key in result_keys:
        if key in int_keys:
            result[return_key][key] = np.zeros(SnapNums.size, dtype=int) - 1
        elif key in threed_keys:
            result[return_key][key] = np.zeros((SnapNums.size, 3), dtype=float) - 1
        elif key in eightd_keys:
            result[return_key][key] = np.zeros((SnapNums.size, 8), dtype=float) - 1
        else:
            result[return_key][key] = np.zeros(SnapNums.size, dtype=float) - 1

    result[return_key]['SnapNum'] = SnapNums


    print('Working on %s snap %03d subfindID %08d'%(sim, snap, subfindID))
    
    # now load the whole main progenitor branches of the subhalo and the
    # main subhalo of its z=0 FoF Host -- then tabulate various properties
    sub_fields  = ['SnapNum', 'SubfindID', 'SubhaloMassInRadType', 'SubhaloStellarPhotometrics', 'GroupFirstSub', 'SubhaloParent',
                   'SubhaloHalfmassRadType', 'SubhaloPos', 'SubhaloGrNr', 'SubhaloSFR',
                   'SubhaloSFRinRad', 'SubhaloMass', 'SubhaloMassType', 
                   'SubhaloBHMdot', 'SubhaloBHMass', 'SubhaloSpin', 'SubhaloVel']
    
    host_fields = ['SnapNum', 'SubfindID', 'SubhaloMassInRadType',
                   'SubhaloHalfmassRadType', 'SubhaloPos', 'SubhaloGrNr',
                   'SubhaloVel', 'Group_M_Crit200', 'Group_R_Crit200']
    
    # load the full subhalo branch
    sub_tree = ru.loadMainTreeBranch(sim, snap, subfindID, fields=sub_fields,
                                    min_snap=min_snap, max_snap=max_snap, treeName=treeName)
    
    # check that the subhalo is in the merger tree
    if not sub_tree:
        return

    # load the host_tree MPB using the GroupFirstSub from the last identified snap of the subhalo        
    host_tree = il.sublink.loadTree(basePath, sub_tree['SnapNum'][0], sub_tree['GroupFirstSub'][0],
                                    treeName=treeName, fields=host_fields, onlyMPB=True)

    # find the snapshots where both the subhalo and host have been identified
    snap_indices, sub_indices, host_indices = ru.find_common_snaps(SnapNums,
                                                                   sub_tree['SnapNum'],
                                                                   host_tree['SnapNum'])

    # calculate the host-centric distance
    a                    = Times[snap_indices]
    boxsizes             = BoxSizes[snap_indices]
    SubPos               = (sub_tree['SubhaloPos'][sub_indices].T * a / h).T
    HostPos              = (host_tree['SubhaloPos'][host_indices].T * a / h).T
    hostcentricdistances = np.zeros(sub_indices.size, dtype=float)

    # find a way to vectorize this... Nx3, Nx3, Nx1 array indexing is hard
    for i, sub_index in enumerate(sub_indices):
        boxsize = boxsizes[i]
        subpos = SubPos[i]
        hostpos = HostPos[i]
        hostcentricdistances[i] = ru.mag(subpos, hostpos, boxsize)

    hostcentricdistances_norm = (hostcentricdistances /
                                 (host_tree['Group_R_Crit200'][host_indices] * a / h))

    dsets = [sub_tree['SubfindID'][sub_indices],
             sub_tree['SubhaloMassInRadType'][sub_indices,star_ptn] * 1.0e10 / h,
             sub_tree['SubhaloStellarPhotometrics'][sub_indices],
             sub_tree['SubhaloHalfmassRadType'][sub_indices,star_ptn] * a / h,
             sub_tree['SubhaloGrNr'][sub_indices], sub_tree['GroupFirstSub'][sub_indices],
             sub_tree['SubhaloParent'][sub_indices],
             sub_tree['SubhaloSFR'][sub_indices], sub_tree['SubhaloSFRinRad'][sub_indices],
             sub_tree['SubhaloMass'][sub_indices] * 1.0e10 / h,
             sub_tree['SubhaloMassType'][sub_indices,dm_ptn] * 1.0e10 / h,
             SubPos, sub_tree['SubhaloBHMdot'][sub_indices] * 10.22,
             sub_tree['SubhaloBHMass'][sub_indices] * 1.0e10 / h,
             sub_tree['SubhaloSpin'][sub_indices] / h,
             sub_tree['SubhaloVel'][sub_indices], host_tree['SubhaloVel'][host_indices],
             HostPos, host_tree['SubfindID'][host_indices],
             host_tree['SubhaloMassInRadType'][host_indices,star_ptn] * 1.0e10 / h,
             host_tree['SubhaloHalfmassRadType'][host_indices,star_ptn] * a / h,
             host_tree['Group_M_Crit200'][host_indices] * 1.0e10 / h,
             host_tree['Group_R_Crit200'][host_indices] * a / h,
             host_tree['SubhaloGrNr'][host_indices],
             hostcentricdistances, hostcentricdistances_norm]
    
    for i, key in enumerate(result_keys):
        if (key in threed_keys) or (key in eightd_keys):
            result[return_key][key][snap_indices,:] = dsets[i]
        else:
            result[return_key][key][snap_indices] = dsets[i]
                                               
    return result

