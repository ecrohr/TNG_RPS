### import modules
import illustris_python as il
import multiprocessing as mp
import numpy as np
import h5py
import rohr_utils as ru 
from importlib import reload
import glob
import os
from globals import *

in_tree_key    = 'in_tree'
central_key    = 'central'
in_z0_host_key = 'in_z0_host'
host_m200c_key = 'host_m200c'

true_return  = np.ones(SnapNums.size, dtype=int)
false_return = true_return.copy() * 0
inval_return = true_return.copy() * -1
float_return = true_return.copy() * -1.

keys = [in_tree_key,
        central_key,
        in_z0_host_key,
        host_m200c_key]

subfindsnapshot_outdirec = '../Output/%s_subfindsnapshotflags/'%sim
subfindsnapshot_outfname = 'subfindsnapshot_flags.hdf5'

if (os.path.isdir(subfindsnapshot_outdirec)):
    print('Directory %s exists.'%subfindsnapshot_outdirec)
    if os.path.isfile(subfindsnapshot_outdirec + subfindsnapshot_outfname):
        print('File %s exists. Overwriting.'%(subfindsnapshot_outdirec+subfindsnapshot_outfname))
    else:
        print('File %s does not exists. Writing.'%(subfindsnapshot_outdirec+subfindsnapshot_outfname))
else:
    print('Directory %s does not exist. Creating it now.'%subfindsnapshot_outdirec)
    os.system('mkdir %s'%subfindsnapshot_outdirec)


def run_subfindsnapshot_flags():
    """
    Create the FoF membership flags for all subhalos at each snapshot.
    Start by initializing the output and task manager, and then saves the output.
    Note that the number of subhalos of interest and the final result may have
    different shapes. That is, while subfindIDs.size == len(result_list), the shape
    of the result may not be (subfindIDs.size, 100). See initialize_TNGCluster() for
    an example.
    """
        
    subfindIDs, result = initialize_allsubhalos()
        
    print('Number of subhalos: %d'%subfindIDs.size)

    if mp_flag:
        pool = mp.Pool(mp.cpu_count(8)) # should be 8 if running interactively
        result_list = pool.map(create_flags, subfindIDs)
    else:
        result_list = []
        for subfindID in subfindIDs:
            result_list.append(create_flags(subfindID))

    for i, subfindID in enumerate(subfindIDs):
        result_dic = result_list[i]
        for key in result_dic.keys():
            result[key][subfindID] = result_dic[key]
        
    with h5py.File(subfindsnapshot_outdirec + subfindsnapshot_outfname, 'a') as outf:
        group = outf.require_group('group')
        for dset_key in result.keys():
            dset = result[dset_key]
            dataset = group.require_dataset(dset_key, shape=dset.shape, dtype=dset.dtype)
            dataset[:] = dset
            
        outf.close()
    
    return result


def initialize_allsubhalos():
    """
    Create a list of the subfindIDs for all subhalos in the simulation.
    """
    
    Nsubhalos = il.groupcat.loadSubhalos(basePath, max_snap, fields='SubhaloFlag').size
    subfindIDs = np.arange(Nsubhalos)
    
    # initlaize and fill finalize result
    result = {}
    for key in keys:
        if key == host_m200c_key:
            result[key] = np.zeros((Nsubhalos, SnapNums.size), dtype=float) - 1.
        else:
            result[key] = np.zeros((Nsubhalos, SnapNums.size), dtype=int) - 1
    
    return subfindIDs, result


def initialize_TNGCluster():
    """
    Create a list of all TNGCluster subfindIDs of interest, namely those
    defined in Create_SubfindIndices. However, we want to use the subfindID
    as the index into the final result. So we create the result to be of size
    all subfindIDs (with all -1 values), and later calculate the flags only
    for the subhalos of interest.
    """
    
    # use the function defined in Create_SubfindIndices to define the subhalos of interest
    from Create_SubfindIndices import initialize_TNGCluster_subfindindices
    _, subfindIDs = initialize_TNGCluster_subfindindices()
    
    # use the full catalog for the number of subhalos
    Nsubhalos = il.groupcat.loadSubhalos(basePath, max_snap, fields='SubhaloFlag').size
    
    # initlaize and fill finalize result
    result = {}
    for key in keys:
        if key == host_m200c_key:
            result[key] = np.zeros((Nsubhalos, SnapNums.size), dtype=float) - 1.
        else:
            result[key] = np.zeros((Nsubhalos, SnapNums.size), dtype=int) - 1
    
    return subfindIDs, result


def create_flags(subfindID):
    """
    Given the subfindID (at snapshot 99), load the subhalo tree,
    and potentially it's z=0 host's branch. For each snapshot,
    assign values for the flags, and return the result.
    """
        
    result = init_result()
    
    fields  = ['SubfindID', 'SnapNum', 'SubhaloGrNr', 'GroupFirstSub', 'Group_M_Crit200']               

    MPB_sub  = il.sublink.loadTree(basePath, max_snap, subfindID,
                                   fields=fields, onlyMPB=True, treeName=treeName)
    
    # if the subhalo is not in the tree, return the default result 
    if (MPB_sub is None):
        return result 
    
    # subhalo exists in the mergertrees
    # at which snapshots does the MPB exist?
    snap_indices, sub_indices, _ = ru.find_common_snaps(SnapNums,
                                                        MPB_sub['SnapNum'],
                                                        MPB_sub['SnapNum'])
                    
    result[in_tree_key][snap_indices] = true_return[sub_indices] 

    # check when the subhalo is central
    central_flags = MPB_sub['SubfindID'] == MPB_sub['GroupFirstSub']
    result[central_key][snap_indices] = central_flags
    
    # load the host halo mass
    result[host_m200c_key][snap_indices] = MPB_sub['Group_M_Crit200'] * 1.0e10 / h
    
    # if the subhalo is a central at z=0, then in z=0 host is the same as central flags 
    if (central_flags[0]):
        result[in_z0_host_key][snap_indices] = central_flags
        
        return result 
        
        
    # subhalo is a satellite at z=0, so load the host's MPB
    MPB_host = il.sublink.loadTree(basePath, max_snap, MPB_sub['GroupFirstSub'][0],
                                   fields=fields, onlyMPB=True, treeName=treeName)
    
    # if the host MPB does not exist in the tree (shouldn't occur), then return 
    if (MPB_host is None):
        return result

        
    # find the snaps at which both the sub and host branches exist
    snap_indices, sub_indices, host_indices = ru.find_common_snaps(SnapNums,
                                                                   MPB_sub['SnapNum'],
                                                                   MPB_host['SnapNum'])
    
    # if there are no common snaps (shouldn't occur), then return
    if len(sub_indices) == 0:
        return result
    
    # check when the subhalo was a member of its z=0 host
    in_z0_host_flags = MPB_sub['SubhaloGrNr'][sub_indices] == MPB_host['SubhaloGrNr'][host_indices]
    result[in_z0_host_key][snap_indices] = in_z0_host_flags
    
    return result 


def init_result():
    """
    Initialize the result.
    """
    init_result = {}
    for key in keys:
        if key == in_tree_key:
            init_result[key] = false_return.copy()
        elif key == host_m200c_key:
            init_result[key] = float_return.copy()
        else:
            init_result[key] = inval_return.copy()
            
    return init_result
