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

def run_subfindindices(mp_flag=False, zooniverse_flag=False):

    global sim, basePath, SnapNums, BoxSizes, Times, h
    
    basePath = ru.ret_basePath(sim)

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

        SnapNum, SubfindID = initialize_zooniverseindices()

        ins_key = 'inspected'

        outfname = 'zooniverse_%s_%s_branches.hdf5'%(sim, ins_key)
        outdirec = '../Output/%s_subfindGRP/'%sim

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
    
    return


def initialize_subfindindices():
    """
    Define SubfindIDs and SnapNums to be tracked.
    Returns SnapNums, SubfindIDs
    """

    SubfindIDs = [30, 282800, 363014]
    SnapNums   = np.ones(len(SubfindIDs), dtype=int) * 99

    return SnapNums, SubfindIDs


def initialize_zooniverseindices():
    """
    Load all zooniverse output catalogs for the given simulation and determine 
    which galaxies have been inspected at multiple snapshots. Then tabulates the last 
    snapshot at which the galaxy was inspected, and the subfindID at that snapshot. 
    Returns SnapNums, SubfindIDs
    """

    # load the inspected IDs dictionary
    insIDs_dict = load_zooniverseIDs()

    # create empty lists 
    snapnums   = []
    subfindids = []

    # initialize dictionary with empty lists that will hold the SubfindIDs, 
    # corresponding to branches that have already been cataloged 
    out_dict = {}
    for i in range(0, 100):
        out_dict['%03d'%i] = []
    
    # load in each file consecutively, and check for every done subhalo whether
    # it has already been cataloged. if not, add the key to the empty lists
    # start at snap 99 and work backwards, such that we only need to load the MPBs
    for snap_key in insIDs_dict.keys():

        out_dict_snap = out_dict[snap_key]
        subfindIDs    = insIDs_dict[snap_key]

        for i, subfindID in enumerate(subfindIDs):
            subfindID_key = '%08d'%subfindID

            # check if this subfindID at this snap has already been cataloged
            if subfindID_key in out_dict_snap:
                continue
            
            else:
                # load sublink_gal MPB and tabulate
                fields = ['SnapNum', 'SubfindID']
                MPB    = il.sublink.loadTree(basePath, int(snap_key), subfindID,
                                             fields=fields, onlyMPB=True, treeName='SubLink_gal')

                if MPB is None:
                    print('No MPB for %s snap %s subhaloID %s. Continuing'%(sim, snap_key, subfindID))
                    continue

                for j in range(MPB['count']):
                    snapkey       = '%03d'%MPB['SnapNum'][j]
                    subfindid     = MPB['SubfindID'][j]
                    subfindid_key = '%08d'%MPB['SubfindID'][j]
                    out_dict[snapkey].append(subfindid_key)
                # finish loop over the MPB

                snapnums.append(int(snap_key))
                subfindids.append(int(subfindID_key))
                
    # finish loop over the insIDs and save the keys
    snapnums   = np.array(snapnums, dtype=type(snapnums[0]))
    subfindids = np.array(subfindids, dtype=type(subfindids[0]))

    return snapnums, subfindids


def load_zooniverseIDs():
    """
    Load all zooniverse catalogs. Create a dictionary with each snapshot as the key,
    and the entries are the subfindIDs of all inspected galaxies at that snapshot.
    Returns the dictionary.
    """
    
    # load in the filenames for each snapshot, starting at the last snap
    indirec  = '../IllustrisTNG/%s/postprocessing/Zooniverse_CosmologicalJellyfish/flags/'%sim
    infname  = 'cosmic_jellyfish_flags_*.hdf5'
    infnames = glob.glob(indirec + infname)
    infnames.sort(reverse=True)

    # create dictionaries with snapnum as the key and lists of subfindIDs as the entires
    insIDs_dict = {}
    for filename in infnames:
        snap_key = filename[-8:-5]
        f        = h5py.File(filename, 'r')
        done     = f['done'][0]
        Score    = f['Score'][0]
        
        insIDs_dict[snap_key] = np.where(done == 1)[0]

        f.close()
    # finish loop over files
    
    return insIDs_dict


def return_subfindindices(snap, subfindID, min_snap=0, max_snap=99):
    """
    Given the snap and subfindID, load the MPB/MDB between min_snap and max_snap.
    Then record various properties at each of these snaps.
    Returns dictionary of these properties.
    """

    # initialize results
    snaps = np.arange(max_snap, min_snap-1, -1)
    
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
    
    # load the full subhalo branch
    sub_tree = ru.loadMainTreeBranch(sim, snap, subfindID, fields=sub_fields, min_snap=min_snap)
    
    # load the host_tree MPB using the GroupFirstSub from the last identified snap of the subhalo        
    host_tree = il.sublink.loadTree(basePath, sub_tree['SnapNum'][0], sub_tree['GroupFirstSub'][0],
                                    treeName='SubLink_gal', fields=host_fields, onlyMPB=True)

    # find the snapshots where both the subhalo and host have been identified
    snap_indices, sub_indices, host_indices = ru.find_common_snaps(snaps,
                                                                   sub_tree['SnapNum'],
                                                                   host_tree['SnapNum'])

    
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


sims = ['TNG50-1']
mp_flag = True
zooniverse_flag = True

for sim in sims:
    sim = sim
    run_subfindindices(mp_flag=mp_flag,
                       zooniverse_flag=zooniverse_flag)
    
