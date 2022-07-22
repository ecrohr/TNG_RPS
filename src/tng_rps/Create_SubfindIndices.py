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
from importlib import reload
import glob

global sim, basePath, SnapNums, BoxSizes, Times, h
global jellyscore_min, ins_key
global insIDs_dict, jelIDs_dict

jellyscore_min = 16
ins_key = 'inspected'
insIDs_dict = jelIDs_dict = None

def run_zooniverseindices(create_indices_flag=False, mp_flag=False, zooniverse_flag=False):

    global sim, basePath, SnapNums, BoxSizes, Times, h
    global jellyscore_min, ins_key
    global insIDs_dict, jelIDs_dict
    
    basePath = ru.ret_basePath(sim)

    print(sim, basePath)
    print()

    Header = il.groupcat.loadHeader(basePath, 99)
    h = Header['HubbleParam']

    SnapNums = range(99, -1, -1)
    Times = np.zeros(len(SnapNums), dtype=float)
    BoxSizes = np.zeros(len(SnapNums), dtype=float)
    for i, SnapNum in enumerate(SnapNums):
        header = il.groupcat.loadHeader(basePath, SnapNum)
        Times[i] = header['Time']
        BoxSizes[i] = header['BoxSize'] * Times[i] / h

    # using the zooniverse results?
    if (zooniverse_flag):

        # no functionality here
        """
        # create the keys if they haven't already been created
        if (create_indices_flag):
            initialize_zooniverseindices()

        # load the SubfindIDs and SnapNums from the zooniverse results
        infname  = 'zooniverse_%s_%s_keys.hdf5'%(sim, ins_key)
        indirec  = '../Output/zooniverse/'

        outfname = 'zooniverse_%s_%s_branches.hdf5'%(sim, ins_key)
        outdirec = indirec

        with h5py.File(indirec + infname, 'a') as inf:
            group = inf['%s'%ins_key]
            SnapNum = group['SnapNum'][:]
            SubfindID = group['SubfindID'][:]

            inf.close()
            
        # run return_zooniverseindices
        if mp_flag:
            pool = mp.Pool(mp.cpu_count()) # should be 8 if running interactively
            result_list = pool.starmap(return_zooniverseindices, zip(SnapNum,
                                                                     SubfindID))
        else:
            result_list = []
            for index, subfindID in enumerate(subfindID):
                result_list.append(return_zooniverseindices(SnapNum[index],
                                                            SubfindID))
        """

    # not using the zooniverse results -- define subfindIDs somehow else... 
    else:
        SnapNum, SubfindID = initialize_subfindindices()

        outdirec = '../Output/%s_subfindGRP/'%sim
        outfname = 'subfind_%s_branches.hdf5'%(sim)

        # run return_subfindindices
        if mp_flag:
            pool = mp.Pool(mp.cpu_count()) # should be 8 if running interactively
            result_list = pool.starmap(return_subfindindices, zip(SnapNum,
                                                                  SubfindID))
        else:
            result_list = []
            for index, subfindID in enumerate(SubfindID):
                result_list.append(return_subfindindices(SnapNum[index],
                                                         subfindID))
        

    # reformat result and save
    result = {}
    for d in result_list:
        result.update(d)
        
    with h5py.File(outdirec + outfname, 'a') as outf:
        for group_key in result.keys():
            group = outf.require_group(group_key)
            for dset_key in result[group_key].keys():
                dset = result[group_key][dset_key]
                dataset = group.require_dataset(dset_key, shape=dset.shape, dtype=dset.dtype)
                dataset[:] = dset
            
        outf.close()

    if (zooniverse_flag):
        # add jellyfish and inspected flags
        add_flags()
    
    return


def initialize_subfindindices():
    """
    Define SubfindIDs and SnapNums to be tracked.
    Currently runs for the first 10 SubfindIDs at snapshot 99.
    Returns SnapNums, SubfindIDs
    """

    SubfindIDs = np.arange(100)
    SnapNums   = np.ones(len(SubfindIDs), dtype=int) * 99

    return SnapNums, SubfindIDs


def return_subfindindices(snap, subfindID, snapNum=33, max_snap=99):
    """
    Given the snap and subfindID, load the MPB/MDB between snapNum and max_snap.
    Then record various properties at each of these snaps.
    Returns dictionary of these properties.
    """

    # helper function to check if the MDB has issues or not, given the subMPB
    def return_subtree(subMPB):

        subMDB = il.sublink.loadTree(basePath, snap, subfindID, treeName='SubLink_gal',
                                     fields=sub_fields, onlyMDB=True)
        
        # check if there's an issue with the MDB -- if the MDB reaches z=0
        # if so, then only use the MPB
        if (subMDB['count'] + snap) > (max_snap + 1):
            print('Issue with MDB for %s snap %d subfindID %d'%(sim, snap, subfindID))

            # find where the MDB stops
            stop  = -(max_snap - snapNum + 1)
            start = np.max(np.where((subMDB['SnapNum'][1:] - subMDB['SnapNum'][:-1]) >= 0)) + 1

            for key in sub_fields:
                subMDB[key] = subMDB[key][start:stop]
            subMDB['count'] = len(subMDB[key])

            
        # if the MDB is clean (reaches z=0), combine the MPB and MDB trees
        sub_tree = {}
        for key in subMPB.keys():
            if key == 'count':
                sub_tree[key] = subMDB[key] + subMPB[key] - 1
            else:
                sub_tree[key] = np.concatenate([subMDB[key][:-1], subMPB[key]])

        return sub_tree

    # initialize results
    snaps = np.arange(max_snap, snapNum-1, -1)
    
    return_key = '%03d_%08d'%(snap, subfindID)

    result_keys = ['SubfindID',
                   'Subhalo_Mstar_Rgal',
                   'Subhalo_Rgal',
                   'SubhaloGrNr', 'SubGroupFirstSub',
                   'SubhaloSFR', 'SubhaloSFRinRad',
                   'SubhaloMass',
                   'SubhaloPos', 'SubhaloBHMdot',
                   'SubhaloBHMass',
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
                'HostSubfindID', 'HostSubhaloGrNr']

    threed_keys = ['SubhaloPos', 'SubhaloVel',
                   'HostSubhaloVel', 'HostSubhaloPos']

    for key in result_keys:
        if key in int_keys:
            result[return_key][key] = np.ones(len(snaps), dtype=int) * -1
        elif key in threed_keys:
            result[return_key][key] = np.ones((len(snaps), 3), dtype=int) * -1
        else:
            result[return_key][key] = np.ones(len(snaps), dtype=float) * -1.

    result[return_key]['SnapNum'] = snaps


    print('Working on %s snap %03d subfindID %08d'%(sim, snap, subfindID))
    
    starpartnum = il.util.partTypeNum('star')
    gaspartnum = il.util.partTypeNum('gas')

    # now load the whole main progenitor branches of the subhalo and the
    # main subhalo of its z=0 FoF Host -- then tabulate various properties
    sub_fields  = ['SnapNum', 'SubfindID', 'SubhaloMassInRadType', 'GroupFirstSub',
                   'SubhaloHalfmassRadType', 'SubhaloPos', 'SubhaloGrNr', 'SubhaloSFR',
                   'SubhaloSFRinRad', 'SubhaloMass', 'SubhaloBHMdot', 'SubhaloBHMass',
                   'SubhaloVel']
    
    host_fields = ['SnapNum', 'SubfindID', 'SubhaloMassInRadType',
                   'SubhaloHalfmassRadType', 'SubhaloPos', 'SubhaloGrNr',
                   'SubhaloVel', 'Group_M_Crit200', 'Group_R_Crit200']
    
    # load the subhalo MPB
    subMPB = il.sublink.loadTree(basePath, snap, subfindID, treeName='SubLink_gal',
                                 fields=sub_fields, onlyMPB=True)

    # check if the subhalo is in the tree
    if not subMPB:
        return result
        
    # if snap < 99, load the MDB and combine with the MPB
    if snap < 99:
        sub_tree = return_subtree(subMPB)
    else: 
        sub_tree = subMPB
    
    # load the host_tree MPB using the GroupFirstSub from the last identified snap of the subhalo        
    host_tree = il.sublink.loadTree(basePath, sub_tree['SnapNum'][0], sub_tree['GroupFirstSub'][0],
                                    treeName='SubLink_gal', fields=host_fields, onlyMPB=True)

    # find the snapshots where both the subhalo and host have been identified
    sub_indices  = []
    host_indices = []
    snap_indices = []
    for snap_index, SnapNum in enumerate(snaps):
        if ((SnapNum in sub_tree['SnapNum']) & (SnapNum in host_tree['SnapNum'])):
            sub_indices.append(np.where(SnapNum == sub_tree['SnapNum'])[0])
            host_indices.append(np.where(SnapNum == host_tree['SnapNum'])[0])
            snap_indices.append(snap_index)
    # note that sub, host indicies are lists of arrays, while
    # snap indices is a list of ints 
    sub_indices  = np.concatenate(sub_indices)
    host_indices = np.concatenate(host_indices)
    snap_indices = np.array(snap_indices)
    
    # calculate the host-centric distance
    a                    = Times[snap_indices]
    boxsizes             = BoxSizes[snap_indices]
    SubPos               = (sub_tree['SubhaloPos'][sub_indices].T * a / h).T
    HostPos              = (host_tree['SubhaloPos'][host_indices].T * a / h).T
    hostcentricdistances = np.zeros(len(sub_indices), dtype=float)

    # find a way to vectorize this... Nx3, Nx3, Nx1 array indexing is hard
    for i, sub_index in enumerate(sub_indices):
        boxsize = boxsizes[i]
        subpos = SubPos[i]
        hostpos = HostPos[i]
        hostcentricdistances[i] = ru.mag(subpos, hostpos, boxsize)

    hostcentricdistances_norm = (hostcentricdistances /
                                 (host_tree['Group_R_Crit200'][host_indices] * a / h))

    dsets = [sub_tree['SubfindID'][sub_indices],
             sub_tree['SubhaloMassInRadType'][sub_indices,starpartnum] * 1.0e10 / h,
             sub_tree['SubhaloHalfmassRadType'][sub_indices,starpartnum] * a / h,
             sub_tree['SubhaloGrNr'][sub_indices], sub_tree['GroupFirstSub'][sub_indices],
             sub_tree['SubhaloSFR'][sub_indices], sub_tree['SubhaloSFRinRad'][sub_indices],
             sub_tree['SubhaloMass'][sub_indices] * 1.0e10 / h,
             SubPos, sub_tree['SubhaloBHMdot'][sub_indices] * 10.22,
             sub_tree['SubhaloBHMass'][sub_indices] * 1.0e10 / h,
             sub_tree['SubhaloVel'][sub_indices], host_tree['SubhaloVel'][host_indices],
             HostPos, host_tree['SubfindID'][host_indices],
             host_tree['SubhaloMassInRadType'][host_indices,starpartnum] * 1.0e10 / h,
             host_tree['SubhaloHalfmassRadType'][host_indices,starpartnum] * a / h,
             host_tree['Group_M_Crit200'][host_indices] * 1.0e10 / h,
             host_tree['Group_R_Crit200'][host_indices] * a / h,
             host_tree['SubhaloGrNr'][host_indices],
             hostcentricdistances, hostcentricdistances_norm,]
    
    for i, key in enumerate(result_keys):
        if key in threed_keys:
            result[return_key][key][snap_indices,:] = dsets[i]
        else:
            result[return_key][key][snap_indices] = dsets[i]
                                               
    return result


sims = ['TNG50-4']
create_indices_flag = True
mp_flag = False
zooniverse_flag = False

for sim in sims:
    sim = sim
    run_zooniverseindices(create_indices_flag=create_indices_flag,
                          mp_flag=mp_flag,
                          zooniverse_flag=zooniverse_flag)
    
    insIDs_dict = jelIDs_dict = None

