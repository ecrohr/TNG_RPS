### import modules
import illustris_python as il
import multiprocessing as mp
import numpy as np
import h5py
import rohr_utils as ru 
from functools import partial
import glob
import os

in_tree_key    = 'in_tree'
central_key    = 'central'
in_z0_host_key = 'in_z0_host'
host_m200c_key = 'host_m200c'

keys = [in_tree_key,
        central_key,
        in_z0_host_key,
        host_m200c_key]
    

def run_subfindsnapshot_flags(Config):
    """
    Create the FoF membership flags for all subhalos at each snapshot.
    Start by initializing the output and task manager, and then saves the output.
    Note that the number of subhalos of interest and the final result may have
    different shapes. That is, while subfindIDs.size == len(result_list), the shape
    of the result may not be (subfindIDs.size, 100). See initialize_TNGCluster() for
    an example.
    """
    
    # check the output directory and fname
    subfindsnapshot_outdirec, subfindsnapshot_outfname = return_outdirec_outfname(Config, snapshotflags=True)
            
    # initialize the subfindIDs of interest and final result shape
    if Config.TNGCluster_flag:
        subfindIDs, result = initialize_TNGCluster(Config)
    else:
        subfindIDs, result = initialize_allsubhalos(Config)
        
    print('Number of subhalos of interest: %d'%subfindIDs.size)
    
    if Config.run_SS:

        if Config.mp_flag:
            pool = mp.Pool(Config.Nmpcores)
            result_list = pool.map(partial(create_flags, Config=Config), subfindIDs)
        else:
            result_list = []
            for subfindID in subfindIDs:
                result_list.append(create_flags(subfindID, Config))

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
    
    if Config.run_SS_PP:
        
        # post process the catalog to create one flag per z=0 subhalo of interest
        result = postprocess_flags(subfindIDs, Config)
        subfindsnapshot_outdirec, subfind_outfname = return_outdirec_outfname(Config, snapshotflags=False)
        with h5py.File(subfindsnapshot_outdirec + subfind_outfname, 'a') as outf:
            group = outf.require_group('group')
            for dset_key in result.keys():
                dset = result[dset_key]
                dataset = group.require_dataset(dset_key, shape=dset.shape, dtype=dset.dtype)
                dataset[:] = dset
                
            outf.close()
    
    return result


def initialize_allsubhalos(Config):
    """
    Create a list of the subfindIDs for all subhalos in the simulation.
    """
    
    subfindIDs = Config.SubfindIDs
    
    # initlaize and fill finalize result
    result = {}
    for key in keys:
        if key == host_m200c_key:
            result[key] = np.zeros((subfindIDs.size, Config.SnapNums.size), dtype=float) - 1.
        else:
            result[key] = np.zeros((subfindIDs.size, Config.SnapNums.size), dtype=int) - 1
    
    return subfindIDs, result


def initialize_TNGCluster(Config):
    """
    Create a list of all TNGCluster subfindIDs of interest, namely those
    defined in Create_SubfindIndices. However, we want to use the subfindID
    as the index into the final result. So we create the result to be of size
    all subfindIDs (with all -1 values), and later calculate the flags only
    for the subhalos of interest.
    """
    
    SnapNums = Config.SnapNums
    subfindIDs = Config.SubfindIDs
    
    # use the full catalog for the number of subhalos
    Nsubhalos = il.groupcat.loadSubhalos(Config.basePath, Config.max_snap, fields='SubhaloGrNr').size
    
    # initlaize and fill finalize result
    result = {}
    # check if the files already exist, and if so, overwrite
    subfindsnapshot_outdirec, subfindsnapshot_outfname = return_outdirec_outfname(Config, snapshotflags=True)
    if os.path.isfile(subfindsnapshot_outdirec + subfindsnapshot_outfname):
        with h5py.File(subfindsnapshot_outdirec + subfindsnapshot_outfname, 'r') as f:
            group = f['group']
            for key in group.keys():
                result[key] = group[key][:]
            f.close()
        return subfindIDs, result

    # file does not exist, so initialize result
    for key in keys:
        if key == host_m200c_key:
            result[key] = np.zeros((Nsubhalos, SnapNums.size), dtype=float) - 1.
        else:
            result[key] = np.zeros((Nsubhalos, SnapNums.size), dtype=int) - 1
    
    return subfindIDs, result


def create_flags(subfindID, Config):
    """
    Given the subfindID (at snapshot 99), load the subhalo tree,
    and potentially it's z=0 host's branch. For each snapshot,
    assign values for the flags, and return the result.
    """
        
    result = init_result(Config)
    
    SnapNums = Config.SnapNums
    basePath = Config.basePath
    max_snap = Config.max_snap
    treeName = Config.treeName
    
    true_return  = np.ones(SnapNums.size, dtype=int)
    
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
    result[host_m200c_key][snap_indices] = MPB_sub['Group_M_Crit200'] * 1.0e10 / Config.h
    
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
    

def postprocess_flags(subfindIDs, Config):
    """
    Based on the created flags, create a set of flags only at z=0 that state
    if the given subhalo is a central, and if so is a backsplash galaxy, or
    is a satellite, and if so, if it was pre-processed or previously a backsplash.
    """
    
    Nsnaps_PP = Config.Nsnaps_PP
    M200c0_lolim_PP = Config.M200c0_lolim_PP

    # load the catalogs
    subfindsnapshot_outdirec, subfindsnapshot_outfname = return_outdirec_outfname(Config)
    f = h5py.File(subfindsnapshot_outdirec + subfindsnapshot_outfname, 'r')
    group = f['group']
    
    classified_flag = 'classified'
    central_z0_flag = 'central_z0'
    backsplash_z0_flag = 'backsplash_z0'
    backsplash_prev_flag = 'backsplash_prev'
    preprocessed_flag = 'preprocessed'
    flags = [classified_flag, central_z0_flag, backsplash_z0_flag,
             backsplash_prev_flag, preprocessed_flag]
             
    # initialize result
    result = {}
    # first check if the files already exist, and if so, append to them
    subfindsnapshot_outdirec, subfindsnapshot_outfname = return_outdirec_outfname(Config, snapshotflags=False)
    if os.path.isfile(subfindsnapshot_outdirec + subfindsnapshot_outfname):
        with h5py.File(subfindsnapshot_outdirec + subfindsnapshot_outfname, 'r') as f:
            group = f['group']
            for key in group.keys():
                result[key] = group[key][:]
            f.close()

    for flag in flags:
        result[flag] = np.zeros(len(group[in_tree_key]), dtype=int)
        if flag != classified_flag:
            result[flag] -= 1
    
    # loop over subfindIDs of interest
    # NB: group[in_tree_key].size is not necessarily equal to SubfindIDs.size
    # group[in_tree_key].size is equal to the number of subhalos in the simulation
    # and SubfindIDs.size is equal to the number of subhalos of interest
    for i, subfindID in enumerate(subfindIDs):
        
        # we only care about snapshots when the subhalo was identified in the merger trees
        in_tree = group[in_tree_key][subfindID]
        intree_indices = in_tree == 1
        
        if intree_indices[intree_indices].size < Nsnaps_PP:
            continue
            
        result[classified_flag][subfindID] = 1
        
        central = group[central_key][subfindID][intree_indices]
        in_z0_host = group[in_z0_host_key][subfindID][intree_indices]
        host_m200c = group[host_m200c_key][subfindID][intree_indices] >= M200c0_lolim_PP
        
        # is the subhalo a central at z=0?
        if (central[0]):
            result[central_z0_flag][subfindID] = 1
            # yes! Is the subhalo a backsplash galaxy?
            backsplash_indices = ~central & ~in_z0_host & host_m200c
            backsplash_check = [True] * Config.Nsnaps_PP
            if (ru.is_slice_in_list(backsplash_check, backsplash_indices)):
                result[backsplash_z0_flag][subfindID] = 1
            else:
                result[backsplash_z0_flag][subfindID] = 0
                
            # leave the pre-processed and previous backsplash flags as -1 for centrals,
            # and continue to the next subhalo
            continue
            
        # no, the subhalo is a satellite at z=0
        result[central_z0_flag][subfindID] = 0
        # was the subhalo pre-processed?
        preprocessed_indices = ~central & ~in_z0_host & host_m200c
        preprocessed_check = [True] * Nsnaps_PP
        if (ru.is_slice_in_list(preprocessed_check, preprocessed_indices)):
            result[preprocessed_flag][subfindID] = 1
        else:
            result[preprocessed_flag][subfindID] = 0
            
        # was the subhalo previously a backsplash?
        # first, find the first time that the subhalo spent at least Nsnaps_PP in a massive host
        satellite_indices = ~central & host_m200c
        satellite_check = [True] * Nsnaps_PP
        satellite_indices_bool = ru.where_is_slice_in_list(satellite_indices, satellite_check)
        if any(satellite_indices_bool):
            # from this first time that the subhalo was a satellite, find the index of
            # the last conescutive snapshot.
            end_index = np.where(satellite_indices_bool)[0].max()
            
            if end_index > 0:
                # after this time, was the galaxy a central for at least Nsnaps_PP?
                central_indices = central[:end_index]
                central_check = [True] * Nsnaps_PP
                if central_indices.size >= Nsnaps_PP:
                    if ru.is_slice_in_list(central_check, central_indices):
                        result[backsplash_prev_flag][subfindID] = 1
                        continue
        result[backsplash_prev_flag][subfindID] = 0
                
    # finish loop over SubfindIDs
    f.close()

    # return catalog to main function to save

    return result


def init_result(Config):
    """
    Initialize the result for a single subhalo.
    """
    
    false_return = np.zeros(Config.SnapNums.size, dtype=int)
    inval_return = false_return.copy() * -1
    float_return = false_return.copy() * -1.

    init_result = {}
    for key in keys:
        if key == in_tree_key:
            init_result[key] = false_return.copy()
        elif key == host_m200c_key:
            init_result[key] = float_return.copy()
        else:
            init_result[key] = inval_return.copy()
            
    return init_result


def return_outdirec_outfname(Config, snapshotflags=True):
    """
    Helper function to determine the directory and filename for the outputs.
    """
    
    outdirec = '../Output/%s_subfindsnapshotflags/'%(Config.sim)
    if snapshotflags:
        outfname = 'subfindsnapshot_flags.hdf5'
    else:
        outfname = 'subfind_flags.hdf5'

    if (os.path.isdir(outdirec)):
        print('Directory %s exists.'%outdirec)
        if os.path.isfile(outdirec + outfname):
            print('File %s exists. Overwriting.'%(outdirec+outfname))
        else:
            print('File %s does not exists. Writing.'%(outdirec+outfname))
    else:
        print('Directory %s does not exist. Creating it now.'%subfindsnapshot_outdirec)
        os.system('mkdir %s'%subfindsnapshot_outdirec)
    
    return outdirec, outfname
