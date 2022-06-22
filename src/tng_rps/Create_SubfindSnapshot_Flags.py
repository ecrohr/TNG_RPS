### import modules
import illustris_python as il
import multiprocessing as mp
import numpy as np
import h5py
import rohr_utils as ru 
from importlib import reload
import glob

global treeName, snapNum, h, SnapNums
global in_tree_key, central_key, in_z0_host_key, host_m200c0_key
global true_return, false_return, inval_return, float_return
global sim, basePath

treeName      = 'SubLink_gal'
snapNum       = 99
h             = il.groupcat.loadHeader(ru.ret_basePath('TNG50-1'), snapNum)['HubbleParam']
SnapNums      = np.arange(99, -1, -1)

in_tree_key    = 'in_tree'
central_key    = 'central'
in_z0_host_key = 'in_z0_host'
host_m200c_key = 'host_m200c'

true_return  = np.ones(len(SnapNums), dtype=int)
false_return = true_return.copy() * 0
inval_return = true_return.copy() * -1
float_return = true_return.copy() * -1.

keys = [in_tree_key,
        central_key,
        in_z0_host_key,
        host_m200c_key]

def create_subfindsnapshot_flags(mp_flag=False):
           

    Nsubhalos = len(il.groupcat.loadSubhalos(basePath, snapNum, fields='SubhaloFlag'))
    subfindIDs = range(Nsubhalos)
    
    del Nsubhalos
    
    print('Number of subhalos: %d'%len(subfindIDs))

    if mp_flag:
        pool = mp.Pool(mp.cpu_count(8)) # should be 8 if running interactively
        result_list = pool.map(return_flags, subfindIDs)
    else:
        result_list = []
        for subfindID in subfindIDs:
            result_list.append(return_flags(subfindID))
            
    # initlaize and fill finalize result
    result = {}
    for key in keys:
        if key == host_m200c_key:
            result[key] = np.zeros((len(result_list), len(SnapNums)), dtype=float)
        else:
            result[key] = np.zeros((len(result_list), len(SnapNums)), dtype=int)

    for i, result_dic in enumerate(result_list):
        for key in result_dic.keys():
            result[key][i] = result_dic[key]
            
    # save the flags 
    outdirec = '../Output/'
    outfname = 'subfindsnapshot_flags_%s.hdf5'%(sim)
        
    with h5py.File(outdirec + outfname, 'a') as outf:
        group = outf.require_group('group')
        for dset_key in result.keys():
            dset = result[dset_key]
            dataset = group.require_dataset(dset_key, shape=dset.shape, dtype=dset.dtype)
            dataset[:] = dset
            
        outf.close()
    
    return result


def return_flags(subfindID):
        
    result = init_result()
    
    fields  = ['SubfindID', 'SnapNum', 'SubhaloGrNr', 'GroupFirstSub', 'Group_M_Crit200']               

    MPB_sub  = il.sublink.loadTree(basePath, snapNum, subfindID,
                                   fields=fields, onlyMPB=True, treeName=treeName)
    
    # if the subhalo is not in the tree, return the default result 
    if (MPB_sub is None):
        return result 
    
    # subhalo exists in the mergertrees
    # at which snapshots does the MPB exist?
    sub_indices = []
    snap_indices = []
    for snap_index, SnapNum in enumerate(SnapNums):
        if (SnapNum in MPB_sub['SnapNum']):
            sub_indices.append( np.where(SnapNum == MPB_sub['SnapNum'])[0])
            snap_indices.append(snap_index)
    sub_indices  = np.concatenate(sub_indices)
    snap_indices = np.array(snap_indices)
                    
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
    MPB_host = il.sublink.loadTree(basePath, snapNum, MPB_sub['GroupFirstSub'][0],
                                   fields=fields, onlyMPB=True, treeName=treeName)
    
    # if the host MPB does not exist in the tree (shouldn't occur), then return 
    if (MPB_host is None):
        return result
        
    # find the snaps at which both the sub and host branches exist 
    sub_indices  = []
    host_indices = []
    snap_indices = []
    for snap_index, SnapNum in enumerate(SnapNums):
        if ((SnapNum in MPB_sub['SnapNum']) & (SnapNum in MPB_host['SnapNum'])):
            sub_indices.append( np.where(SnapNum == MPB_sub['SnapNum'])[0])
            host_indices.append(np.where(SnapNum == MPB_host['SnapNum'])[0])
            snap_indices.append(snap_index)
    
    # if there are no common snaps (shouldn't occur), then return
    if len(sub_indices) == 0:
        return result
    
    # note that sub, host indicies are lists of arrays, while
    # snap indices is a list of ints 
    sub_indices  = np.concatenate(sub_indices)
    host_indices = np.concatenate(host_indices)
    snap_indices = np.array(snap_indices)
    
    # check when the subhalo was a member of its z=0 host
    in_z0_host_flags = MPB_sub['SubhaloGrNr'][sub_indices] == MPB_host['SubhaloGrNr'][host_indices]
    result[in_z0_host_key][snap_indices] = in_z0_host_flags
    
    return result 


def init_result():
    init_result = {}
    for key in keys:
        if key == in_tree_key:
            init_result[key] = false_return.copy()
        elif key == host_m200c_key:
            init_result[key] = float_return.copy()
        else:
            init_result[key] = inval_return.copy()
            
    return init_result


sims = ['TNG100-1']
for sim in sims:
    basePath = ru.ret_basePath(sim)
    _ = create_subfindsnapshot_flags()
