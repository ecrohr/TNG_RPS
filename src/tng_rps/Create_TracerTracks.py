### import modules
import illustris_python as il
import multiprocessing as mp
import numpy as np
import h5py
import rohr_utils_draco as ru 
from importlib import reload
import glob


global sim, basePath, snapNum, tcoldgas
global tracer_ptn, star_ptn, gas_ptn, bh_ptn, bary_ptns
global part_fields

# set up some global variables
sim        = 'TNG50-4'
basePath   = ru.ret_basePath(sim)
snapNum    = 50
tcoldgas   = 10.**(4.5) # [K]

tracer_ptn = il.util.partTypeNum('tracer')
star_ptn   = il.util.partTypeNum('star')
gas_ptn    = il.util.partTypeNum('gas')
bh_ptn     = il.util.partTypeNum('bh')

bary_pts   = [gas_ptn,
              star_ptn,
              bh_ptn]

gas_fields  = ['InternalEnergy', 'ElectronAbundance', 'StarFormationRate', 'ParticleIDs']

part_fields = ['ParticleIDs']

# begin main function

# define the subhalos we care about 
subfindIDs = [1]

# initialize the outputs and start with the first snapshot snapNum

# initialize the offset dictionary
offsets_subhalo = {}
offsets_subhalo['SubfindID']     = np.zeros(len(subfindIDs), int)
offsets_subhalo['SubhaloOffset'] = np.zeros(len(subfindIDs), int)
offsets_subhalo['SubhaloLength'] = np.zeros(len(subfindIDs), int)

particles = {}
big_array_length = int(1e8)
# rewrite into a 4xbig_array_length array rather than a dictionary
# for a speed increase
particles['TracerIDs']      = np.empty(big_array_length, dtype=int)
particles['TracerIndices']  = np.empty(big_array_length, dtype=int)
particles['ParentIndices']  = np.empty(big_array_length, dtype=int)
particles['ParentPartType'] = np.empty(big_array_length, dtype=int)

# load every tracer particle in the simulation at this snapshot
tracers = il.snapshot.loadSubset(basePath, snapNum, tracer_ptn)

# begin loop over the subhalos at snapshot snapNum
subfind_i = 0
subfindID = subfindIDs[subfind_i]

gas_cells    = il.snapshot.loadSubhalo(basePath, snapNum, subfindID, gas_ptn, fields=gas_fields)
gas_cells    = ru.calc_temp_dict(gas_cells)

# find the local indices and load the global offset for these gas cells
cgas_indices = np.where(gas_cells['Temperature'] <= tcoldgas)[0]
ParticleIDs  = gas_cells['ParticleIDs'][cgas_indices]

# match the tracer ParentID with the cold gas cells ParticleIDs
isin_tracer = np.isin(tracers['ParentID'], ParticleIDs)

# save the tracerIDs and tracer indices at snapshot snapNum
tracer_IDs = tracers['TracerID'][isin_tracer]
tracer_indices = np.where(isin_tracer)[0]

# fill in the offsets dictionary for this subhalo
offsets_subhalo['SubfindID'][subfind_i]     = subfindID
offsets_subhalo['SubhaloLength'][subfind_i] = len(tracer_indices)

if subhalo_i == 0:
    offsets_subhalo['SubhaloOffset'][subfind_i] = 0
else:
    offsets_subhalo['SubhaloOffset'][subfind_i] = (offsets_subhalo['SubhaloOffset'][subfind_i-1]
                                                   + offsets_subhalo['SubhaloLength'][subfind_i-1])

# save the corresponding gas cell indices
# get the local cold gas indices with matched tracer particles and include the global offset 
isin_gas = np.isin(ParticleIDs, tracers['ParentID'][isin_tracer])

r           = il.snapshot.getSnapOffsets(basePath, snapNum, subfindID, "Subhalo")
offset      = r['offsetType'][gas_ptn]
gas_indices = offset + cgas_indices[isin_gas]

# note that some of these indices need to be repeated due to having multiple tracers with the same parent
gas_IDs        = ParticleIDs[isin_gas]
parent_IDs     = tracers['ParentID'][isin_tracer]
# find a way to optimize the following line... 
repeat_indices = np.where([parent_ID == gas_IDs for parent_ID in parent_IDs])[1]
gas_indices    = gas_indices[repeat_indices]

# note that the parent type is always gas
parent_ptn = np.ones(len(gas_indices), dtype=int) * gas_ptn

# fill in the particle dictionary for this subhalo
start = offsets_subhalo['SubhaloOffset'][subfind_i]
end   = start + offsets_subhalo['SubhaloLength'][subfind_i]

particles['TracerIDs'][start:end]      = tracer_IDs
particles['TracerIndices'][start:end]  = tracer_indices 
particles['ParentIndices'][start:end]  = gas_indices
particles['ParentPartType'][start:end] = parent_ptn

# finish loop over the subhalos at snapshot snapNum
# reshape the arrays
end_length = offsets_subhalo['SubhaloLength'][-1]
for key in particles.keys():
    particles[key] = particles[key][:end_length]

# save the offsets and particles dictionaries
dicts  = [offsets, particles]
fnames = ['offsets', 'tracers']
for d_i, d in enumerate(dicts):
    fname    = fnames[d_i]
    outfname = '%s_%03d.hdf5'%(snapNum, fname)
    outdirec = '../Output/%s_tracers/'%(sim)
    
    with h5py.File(outdirec + outfname, 'a') as outf:
        group = outf.require_group('group')
        for dset_key in result.keys():
            dset = result[dset_key]
            dataset = group.require_dataset(dset_key, shape=dset.shape, dtype=dset.dtype)
            dataset[:] = dset
            
        outf.close()
        
# finish the first snapshot. move to the next. 








"""


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

"""
