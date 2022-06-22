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

def run_zooniverseindices(create_indices_flag=False, mp_flag=False):

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

    if create_indices_flag:
        create_zooniverseindices()
    
    indirec = '../Output/'
    infname = 'zooniverse_%s_%s_keys.hdf5'%(sim, ins_key)

    with h5py.File(indirec + infname, 'a') as inf:
        group = inf['%s'%ins_key]
        group_SnapNum = group['SnapNum'][:]
        group_SubfindID = group['SubfindID'][:]

        inf.close()
        
    if mp_flag:
        pool = mp.Pool(mp.cpu_count()) # should be 8 if running interactively
        result_list = pool.starmap(return_zooniverseindices, zip(group_SnapNum,
                                                              group_SubfindID))
    else:
        result_list = []
        for index, subfindID in enumerate(group_SubfindID):
            result_list.append(return_zooniverseindices(group_SnapNum[index],
                                                        subfindID))
    result = {}
    for d in result_list:
        result.update(d)

    outdirec = indirec
    outfname = 'zooniverse_%s_%s_branches.hdf5'%(sim, ins_key)
        
    with h5py.File(outdirec + outfname, 'a') as outf:
        for group_key in result.keys():
            group = outf.require_group(group_key)
            for dset_key in result[group_key].keys():
                dset = result[group_key][dset_key]
                dataset = group.require_dataset(dset_key, shape=dset.shape, dtype=dset.dtype)
                dataset[:] = dset
            
        outf.close()

    # add jellyfish and inspected flags
    add_flags()
    
    return 


def create_zooniverseindices():

    global sim, basePath, SnapNums, BoxSizes, Times, h
    global jellyscore_min, ins_key
    global insIDs_dict, jelIDs_dict

    if insIDs_dict is None:
        load_zooniverseIDs()

    # create empty lists to hold the three sets of keys
    full_keys  = []
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

                # full_key is the snapshot + subfindID at the last snapshot where
                # the branch was inspected in zooniverse. Note that the MDB may
                # reach a later time.
                full_keys.append(snap_key + '_' +  subfindID_key)
                snapnums.append(int(snap_key))
                subfindids.append(int(subfindID_key))
                
    # finish loop over the insIDs and save the keys
    snapnums   = np.array(snapnums, dtype=type(snapnums[0]))
    subfindids = np.array(subfindids, dtype=type(subfindids[0]))
    
    outdirec = '../Output/'
    outfname = 'zooniverse_%s_%s_keys.hdf5'%(sim, ins_key)
    dset_keys = ['SnapNum', 'SubfindID']

    with h5py.File(outdirec + outfname, 'a') as outf:
        
        group = outf.require_group(ins_key)
        dsets = [snapnums, subfindids]
        for dset_i, dset_key in enumerate(dset_keys):
            dset       = dsets[dset_i]
            dataset    = group.require_dataset(dset_key, shape=dset.shape, dtype=dset.dtype)
            dataset[:] = dset

        outf.close()        

    return


def load_zooniverseIDs():

    global sim, basePath, SnapNums, BoxSizes, Times, h
    global jellyscore_min, ins_key
    global insIDs_dict, jelIDs_dict
    
    # load in the filenames for each snapshot, starting at snap 99 
    indirec  = '../IllustrisTNG/%s/postprocessing/Zooniverse_CosmologicalJellyfish/flags/'%sim
    infname  = 'cosmic_jellyfish_flags_*.hdf5'
    infnames = glob.glob(indirec + infname)
    infnames.sort(reverse=True)

    # create dictionaries with snapnum as the key and lists of subfindIDs as the entires
    insIDs_dict = {}
    jelIDs_dict = {}
    for filename in infnames:
        snap_key = filename[-8:-5]
        f        = h5py.File(filename, 'r')
        done     = f['done'][0]
        Score    = f['Score'][0]
        
        insIDs_dict[snap_key] = np.where(done == 1)[0]
        jelIDs_dict[snap_key] = np.where(Score >= jellyscore_min)[0]

        f.close()

    return 


def return_zooniverseindices(snap, subfindID):

    global sim, basePath, SnapNums, BoxSizes, Times, h
    global jellyscore_min, ins_key
    global insIDs_dict, jelIDs_dict
    
    # helper function to check if the MDB has issues or not, given the subMPB
    def return_subtree(subMPB):

        subMDB = il.sublink.loadTree(basePath, snap, subfindID, treeName='SubLink_gal',
                                     fields=sub_fields, onlyMDB=True)
        
        # check if there's an issue with the MDB -- if the MDB reaches z=0
        # if so, then only use the MPB
        if (subMDB['count'] + snap) > 100:
            print('Issue with MDB for %s snap %d subfindID %d'%(sim, snap, subfindID))

            sub_tree  = subMPB
            tree_flag = np.array([0], dtype=int)
            
            sub_tree[tree_flag_key] = tree_flag
            
            return sub_tree
        
        # if the MDB is clean (reaches z=0), combine the MPB and MDB trees
        sub_tree = {}
        for key in subMPB.keys():
            if key == 'count':
                sub_tree[key] = subMDB[key] + subMPB[key] - 1
            else:
                sub_tree[key] = np.concatenate([subMDB[key][:-1], subMPB[key]])

        tree_flag = np.array([1], dtype=int)
        sub_tree[tree_flag_key] = tree_flag
        return sub_tree

    return_key = '%03d_%08d'%(snap, subfindID)
    tree_flag_key = 'tree_flag'

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
        
    # if snap < 99, load the MDB and combine with the MPB
    if snap < 99:
        sub_tree = return_subtree(subMPB)
    else: 
        sub_tree = subMPB
        sub_tree[tree_flag_key] = np.array([1], dtype=int)
    
    # load the host_tree MPB using the GroupFirstSub from the last identified snap of the subhalo        
    host_tree = il.sublink.loadTree(basePath, sub_tree['SnapNum'][0], sub_tree['GroupFirstSub'][0],
                                    treeName='SubLink_gal', fields=host_fields, onlyMPB=True)

    # find the snapshots where both the subhalo and host have been identified
    sub_indices  = []
    host_indices = []
    snap_indices = []
    for snap_index, SnapNum in enumerate(SnapNums):
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

    result_keys = ['SnapNum', 'SubfindID',
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
                   'HostCentricDistance_phys', 'HostCentricDistance_norm',
                   'tree_flag']

    dsets = [sub_tree['SnapNum'][sub_indices], sub_tree['SubfindID'][sub_indices],
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
             hostcentricdistances, hostcentricdistances_norm,
             sub_tree[tree_flag_key]]

    result = {}
    result[return_key] = {}
    for i, key in enumerate(result_keys):
        result[return_key][key] = dsets[i]
                                               
    return result


def add_flags():

    global sim, basePath, SnapNums, BoxSizes, Times, h
    global jellyscore_min, ins_key
    global insIDs_dict, jelIDs_dict

    if insIDs_dict is None:
        load_zooniverseIDs()

    fname = 'zooniverse_%s_%s_branches.hdf5'%(sim, ins_key)
    f     = h5py.File('../Output/'+fname, 'a')

    keys = ['ins_flags', 'jel_flags']

    for group_key in f.keys():
        group     = f[group_key]
        SnapNum   = group['SnapNum'][:]
        SubfindID = group['SubfindID'][:]
        ins_flags = np.zeros(len(SnapNum), dtype=int)
        jel_flags = np.zeros(len(SnapNum), dtype=int)

        for index, snap in enumerate(SnapNum):
            snap_key = '%03d'%snap
            try:
                if SubfindID[index] in insIDs_dict[snap_key]:
                    ins_flags[index] = 1
                    if SubfindID[index] in jelIDs_dict[snap_key]:
                        jel_flags[index] = 1
            except KeyError: # no zooniverse classifications as this snap
                continue

        # finish loop over SnapNum
        dsets = [ins_flags, jel_flags]
        for i, key in enumerate(keys):
            dset       = dsets[i]
            dataset    = group.require_dataset(key, shape=dset.shape, dtype=dset.dtype)
            dataset[:] = dset

    # finish loop over groups
    f.close()


sims = ['TNG50-1', 'TNG100-1']
create_indices_flag = True
mp_flag = True

for sim in sims:
    sim = sim
    run_zooniverseindices(create_indices_flag=create_indices_flag, mp_flag=mp_flag)
    insIDs_dict = jelIDs_dict = None


# attempt to examine the bad MDBs, but unsuccessful 
"""
sim = 'TNG50-1'
key = 'control'
basePath = '../../IllustrisTNG/TNG50-1/output'

infname = 'zooniverse_%s_%s_branches.hdf5'%(sim, key)

inf = h5py.File('../../Output/' + infname, 'r')
keys = np.array(list(inf.keys()))

tree_flags = np.zeros(len(keys))
for index, key in enumerate(keys):
    tree_flags[index] = inf[key]['tree_flag'][0]

bad_keys = keys[tree_flags == 0]
D_flags = np.zeros(len(bad_keys), dtype = int)

onlyMDB  = True
treeName = 'Sublink_gal'

for index, key in enumerate(bad_keys):
    snapNum = int(key[:3])
    id = int(key[4:])

    # get RowNum, LastProgID, and SubhaloID
    with h5py.File(offsetPath(basePath, snapNum), 'r') as f:
        groupFileOffsets = f['FileOffsets/Subhalo'][()]
        f.close()
        
    offsetFile = offsetPath(basePath, snapNum)
    prefix = 'Subhalo/' + treeName + '/'

    groupOffset = id

    with h5py.File(offsetFile, 'r') as f:
        # load the merger tree offsets of this subgroup
        RowNum     = f[prefix+'RowNum'][groupOffset]
        LastProgID = f[prefix+'LastProgenitorID'][groupOffset]
        SubhaloID  = f[prefix+'SubhaloID'][groupOffset]
        f.close()
        
    rowStart = RowNum
    rowEnd   = RowNum + (LastProgID - SubhaloID)

    # get offsets
    cache = True
    offsetCache = dict()
    cache = offsetCache

    search_path = treePath(basePath, treeName, '*')
    numTreeFiles = len(glob.glob(search_path))
    if numTreeFiles == 0:
        raise ValueError("No tree files found! for path '{}'".format(search_path))
    offsets = np.zeros(numTreeFiles, dtype='int64')

    for i in range(numTreeFiles-1):
        with h5py.File(treePath(basePath, treeName, i), 'r') as f:
            offsets[i+1] = offsets[i] + f['SubhaloID'].shape[0]
            f.close()

    rowOffsets = rowStart - offsets

    fileNum = np.max(np.where(rowOffsets >= 0))

    fileOff = rowOffsets[fileNum]

    treef = h5py.File(treePath(basePath, treeName, fileNum), 'r')

    # find the subfindID and snapNum of the descendant
    DescendantID = treef['DescendantID'][fileOff]
    index_descendant = np.where(treef['SubhaloID'][:] == DescendantID)[0]

    SubfindID_descendant = treef['SubfindID'][index_descendant]
    SnapNum_descendant   = treef['SnapNum'][index_descendant]

    # load the MPB from the root descendant
    index_RD  = int(np.where(treef['SubhaloID'][:] == treef['RootDescendantID'][fileOff])[0])
    index_max = index_RD + (99 - snapNum)
    index_D   = index_RD + np.where(treef['SnapNum'][index_RD:index_max] == SnapNum_descendant)[0]

    SubfindID_D = treef['SubfindID'][index_D]

    treef.close()

    if SubfindID_descendant == SubfindID_D: # descendant is on MPB -- not helpful
        D_flags[index] = 0
    else:
        D_flags[index] = 1
             
"""    
    
