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
    
    # using the zooniverse results?
    if (Config.zooniverse_flag):
        SnapNum, SubfindID = initialize_zooniverseindices(Config)

    # not using the zooniverse results -- define subfindIDs somehow else... 
    elif (Config.centrals_flag):
        SnapNum, SubfindID = initialize_central_subfindindices(Config)
        
    # TNG-Cluster?
    elif (Config.TNGCluster_flag):
        SnapNum, SubfindID = initialize_TNGCluster_subfindindices(Config)
      
    # general satellites?
    else:
        SnapNum, SubfindID = initialize_subfindindices(Config)

    # run return_subfindindices
    if Config.mp_flag:
        pool = mp.Pool(8) # should be 8 if running interactively
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


def initialize_subfindindices(Config):
    """
    Define SubfindIDs and SnapNums to be tracked.
    Returns all z=0 satellites with Mstar > Mstar_lolim
    and SubhaloFlag == 1. 
    Returns SnapNums, SubfindIDs
    """
    
    basePath = Config.basePath
    star_ptn = Config.star_ptn
    Mstar_lolim = Config.Mstar_lolim
    h = Config.h
    
    subhalo_fields = ['SubhaloFlag', 'SubhaloMassInRadType']
    subhalos = il.groupcat.loadSubhalos(basePath, 99, fields=subhalo_fields)

    halo_fields = ['GroupFirstSub']
    GroupFirstSub = il.groupcat.loadHalos(basePath, 99, fields=halo_fields)

    Mstar = subhalos['SubhaloMassInRadType'][:,star_ptn] * 1.0e10 / h
    subfind_indices = np.where((subhalos['SubhaloFlag']) & (Mstar >= Mstar_lolim))[0]
    indices = np.isin(subfind_indices, GroupFirstSub)
    SubfindIDs = subfind_indices[~indices]
    SnapNums = np.ones(SubfindIDs.size, dtype=int) * 99 

    return SnapNums, SubfindIDs


def initialize_central_subfindindices(Config):
    """
    Define SubfindIDs and SnapNums to be tracked.
    Returns the most massive z=0 central subhalos.
    """
    
    halo_fields = ['Group_M_Crit200','GroupFirstSub']
    halos = il.groupcat.loadHalos(Sim.basePath, 99, fields=halo_fields)
    M200c = halos['Group_M_Crit200'] * 1.0e10 / h
    indices = M200c >= 10.0**(11.5)

    GroupFirstSub = halos['GroupFirstSub']
    SubfindIDs = GroupFirstSub[indices]
    SnapNums = np.ones(SubfindIDs.size, dtype=int) * 99 

    return SnapNums, SubfindIDs
    
    
def initialize_TNGCluster_subfindindices(Config):
    """
    Define the SubfindIDs at z=0 to be tracked.
    """
    
    basePath = Config.basePath
    max_snap = Config.max_snap
    star_ptn = Config.star_ptn
    h = Config.h
    Mstar_lolim = Config.Mstar_lolim
    centrals_flag = Config.centrals_flag
    
    # load all halos and find the primary zoom target IDs
    halo_fields = ['Group_M_Crit200', 'GroupFirstSub', 'GroupPrimaryZoomTarget']
    halos = il.groupcat.loadHalos(basePath, max_snap, fields=halo_fields)
    haloIDs = np.where(halos['GroupPrimaryZoomTarget'])[0]
    GroupFirstSub = halos['GroupFirstSub'][haloIDs]
    
    # load all subhalos and find which ones:
    # 1) are z=0 satellites of primary zooms
    # 2) have Mstar(z=0) > 10^10 Msun
    subhalo_fields = ['SubhaloGrNr', 'SubhaloMassInRadType']
    subhalos = il.groupcat.loadSubhalos(basePath, max_snap, fields=subhalo_fields)
    subhalo_indices_massive = subhalos['SubhaloMassInRadType'][:,star_ptn] * 1.0e10 / h > Mstar_lolim
    
    _, subhalo_match_indices = ru.match3(haloIDs, subhalos['SubhaloGrNr'][subhalo_indices_massive])
    
    # remove the central galaxies
    subhaloIDs = np.where(subhalo_indices_massive)[0][subhalo_match_indices]
    isin = np.isin(subhaloIDs, GroupFirstSub, assume_unique=True)
    
    if centrals_flag:
        subfindIDs = subhaloIDs[isin]
    else:
        subfindIDs = subhaloIDs[~isin]
        
    snaps = np.ones(subfindIDs.size, dtype=subfindIDs.dtype) * max_snap
    
    return snaps, subfindIDs

    
def initialize_zooniverseindices(Config):
    """
    Load all zooniverse output catalogs for the given simulation and determine 
    which galaxies have been inspected at multiple snapshots. Then tabulates the last 
    snapshot at which the galaxy was inspected, and the subfindID at that snapshot. 
    Returns SnapNums, SubfindIDs
    """

    # load the inspected IDs dictionary
    insIDs_dict = load_zooniverseIDs(Config)

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
                MPB    = il.sublink.loadTree(Config.basePath, int(snap_key), subfindID,
                                             fields=fields, onlyMPB=True, treeName=Config.treeName)

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


def load_zooniverseIDs(Config):
    """
    Load all zooniverse catalogs. Create a dictionary with each snapshot as the key,
    and the entries are the subfindIDs of all inspected galaxies at that snapshot.
    Returns the dictionary.
    """
    
    # load in the filenames for each snapshot, starting at the last snap
    indirec  = '../IllustrisTNG/%s/postprocessing/Zooniverse_CosmologicalJellyfish/flags/'%Config.sim
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
            result[return_key][key] = np.ones(SnapNums.size, dtype=int) * -1
        elif key in threed_keys:
            result[return_key][key] = np.ones((SnapNums.size, 3), dtype=float) * -1
        else:
            result[return_key][key] = np.ones(SnapNums.size, dtype=float) * -1.

    result[return_key]['SnapNum'] = SnapNums


    print('Working on %s snap %03d subfindID %08d'%(sim, snap, subfindID))
    
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
    sub_tree = ru.loadMainTreeBranch(sim, snap, subfindID, fields=sub_fields,
                                    min_snap=min_snap, max_snap=max_snap, treeName=treeName)
    
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
             sub_tree['SubhaloHalfmassRadType'][sub_indices,star_ptn] * a / h,
             sub_tree['SubhaloGrNr'][sub_indices], sub_tree['GroupFirstSub'][sub_indices],
             sub_tree['SubhaloSFR'][sub_indices], sub_tree['SubhaloSFRinRad'][sub_indices],
             sub_tree['SubhaloMass'][sub_indices] * 1.0e10 / h,
             SubPos, sub_tree['SubhaloBHMdot'][sub_indices] * 10.22,
             sub_tree['SubhaloBHMass'][sub_indices] * 1.0e10 / h,
             sub_tree['SubhaloVel'][sub_indices], host_tree['SubhaloVel'][host_indices],
             HostPos, host_tree['SubfindID'][host_indices],
             host_tree['SubhaloMassInRadType'][host_indices,star_ptn] * 1.0e10 / h,
             host_tree['SubhaloHalfmassRadType'][host_indices,star_ptn] * a / h,
             host_tree['Group_M_Crit200'][host_indices] * 1.0e10 / h,
             host_tree['Group_R_Crit200'][host_indices] * a / h,
             host_tree['SubhaloGrNr'][host_indices],
             hostcentricdistances, hostcentricdistances_norm]
    
    for i, key in enumerate(result_keys):
        if key in threed_keys:
            result[return_key][key][snap_indices,:] = dsets[i]
        else:
            result[return_key][key][snap_indices] = dsets[i]
                                               
    return result

