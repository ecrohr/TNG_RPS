### import modules
import illustris_python as il
import multiprocessing as mp
import numpy as np
import h5py
import rohr_utils as ru 
from importlib import reload
import glob
import time
import os

global sim, basePath, snapNum, tcoldgas, max_snap
global tracer_ptn, star_ptn, gas_ptn, bh_ptn, bary_ptns
global gas_fields, part_fields
global big_array_length
global outdirec

def create_tracertracks():
    """
    Run the Create_TracerTracks.py file. 
    Starts at snapNum with a list of subhalo subfindIDs,
    find the descendants, and loops over the snapshots
    matching the tracers from bound cold gas cells.
    Saves the output.
    """

    global sim, basePath, snapNum, tcoldgas, max_snap
    global tracer_ptn, star_ptn, gas_ptn, bh_ptn, bary_ptns
    global gas_fields, part_fields
    global big_array_length
    global outdirec
    
    # define the global variables
    sim        = 'TNG50-2'
    basePath   = ru.ret_basePath(sim)
    snapNum    = 33
    tcoldgas   = 10.**(4.5) # [K]
    max_snap   = 99

    tracer_ptn = il.util.partTypeNum('tracer')
    star_ptn   = il.util.partTypeNum('star')
    gas_ptn    = il.util.partTypeNum('gas')
    bh_ptn     = il.util.partTypeNum('bh')

    bary_ptns   = [gas_ptn,
                  star_ptn,
                  bh_ptn]

    gas_fields  = ['InternalEnergy', 'ElectronAbundance', 'StarFormationRate', 'ParticleIDs']

    part_fields = ['ParticleIDs']

    big_array_length = int(1e8)

    # define the subhalos we care about at snapshot snapNum
    subfindIDs = range(10000)
    #subfindIDs = [30, 282800, 363014]
    
    outdirec = '../Output/%s_tracers_%d-%d/'%(sim,subfindIDs[0],subfindIDs[-1])
    #outdirec = '../Output/%s_tracers_zooniverse/'%(sim)
    print(outdirec)
    if not os.path.isdir(outdirec):
        os.system('mkdir %s'%outdirec)

    # find the corresponding subfindIDs at the next snapshots
    track_subfindIDs(subfindIDs)
    
    # now track tracers from snapNum until max_snap
    for snap in range(min_snap, max_snap+1):
        start = time.time()
        track_tracers(snap)
        end = time.time()
        print('%s snap %03d track_tracers: %.3g [s]'%(sim, snap, (end-start)))
              
    # and find the unmatched tracers from snapNum + 1 until max_snap
    for snap in range(snapNum+1, max_snap+1):
        start = time.time()
        find_unmatched_tracers(snap)
        end = time.time()
        print('%s snap %03d find_unmatched_tracers: %.3g [s]'%(sim, snap, (end-start)))

    return


def track_tracers(snap):
    """
    Match the bound cold gas cells of the subfindIDs at snap to the tracers.
    If snap < snapNum, then loads the previous tracers and checks if they still
    belong to the same subhalo. 
    Saves offsets and tracers catalogs, but does not find the unmatched tracers -- 
    i.e., does not search for tracers whose parents were bound cold gas cells but
    no longer are (group 3 tracers in the code below).
    No returns.
    """

    print('Working on %s snap %03d'%(sim, snap))
       
    # load the subfindIDs
    with h5py.File(outdirec + 'offsets_%03d.hdf5'%snap, 'r') as f:
        subfindIDs = f['group']['SubfindID'][:]
        f.close()
      
    # initialize the outputs
    offsets_subhalo, tracers_subhalo = initialize_outputs(len(subfindIDs))
    
    # if snap > snapNum: load previous tracers
    if (snap > snapNum):
        offsets_past = []
        tracers_past = []
        for i in range(1, 4):
            if (snap - i) < snapNum:
                break
            offsets, tracers = load_catalogs(snap - i)
            offsets_past.append(offsets)
            tracers_past.append(tracers)

    else:
        offsets_past = None
        tracers_past = None
        
    # find the gas cells of interest
    ParticleIDs, Particle_indices, Temperatures, offsets, lengths = find_coldgascells(subfindIDs, snap)

    # load all tracers in the simulation
    tracers = il.snapshot.loadSubset(basePath, snap, tracer_ptn)
    
    # match the ParticleIDs to the tracer ParentIDs
    indices1, indices2 = match3(ParticleIDs, tracers['ParentID'])
    
    # slice the the relevant arrays
    tracer_IDs = tracers['TracerID'][indices2]
    tracer_indices = indices2
    
    particle_IDs = ParticleIDs[indices1]
    particle_indices = Particle_indices[indices1]
    temperatures = Temperatures[indices1]
    
    # for each subhalo, find the relevant particles + tracers and save
    for subfind_i, subfindID in enumerate(subfindIDs):
        
        if subfind_i == 0:
            offsets_subhalo['SubhaloOffset'][subfind_i] = 0
        else:
            offsets_subhalo['SubhaloOffset'][subfind_i] = (offsets_subhalo['SubhaloOffset'][subfind_i-1]
                                                           + offsets_subhalo['SubhaloLength'][subfind_i-1])

        start = offsets[subfind_i]
        stop = start + lengths[subfind_i]

        if stop == start:
            continue

        sub_indices = ((indices1 >= start) & (indices1 < stop))
        Ntracers = len(sub_indices[sub_indices])
        offsets_subhalo['SubhaloLength'][subfind_i]            = Ntracers
        offsets_subhalo['SubhaloLengthColdGas'][subfind_i]     = Ntracers
        offsets_subhalo['SubhaloLengthColdGas_new'][subfind_i] = Ntracers
        
        tracer_IDs_sub = tracer_IDs[sub_indices]
        tracer_indices_sub = tracer_indices[sub_indices]

        # if snap > SnapNum: cross match with previous snapshot(s)
        if snap > snapNum:
            ### check which cold gas cell tracers from previous snap are still here ###
            i = 1
            while ((snap - i) >= snapNum) & (i < 4):
                if offsets_past[i-1]['SubfindID'][subfind_i] != -1:
                    start = offsets_past[i-1]['SubhaloOffset'][subfind_i]
                    end   = start + offsets_past[i-1]['SubhaloLengthColdGas'][subfind_i]

                    IDs_past = tracers_past[i-1]['TracerIDs'][start:end]
                    indices_past = tracers_past[i-1]['TracerIndices'][start:end]
                    break
                else:
                    i += 1
                # if the subhalo isn't in the merger trees the past few snaps, move on
                if i == 4:
                    IDs_past = indices_past = np.ones(1, dtype=int) * -1
                    break
            
            # reorder tracer_IDs and tracer_indices such that:
            # 1st: cold gas tracers found in both snapshots
            # 2nd: new cold gas tracers (not in the previous snapshot)
            # 3rd: no longer cold gas tracers (was in previous snapshot but no longer)
            isin_now  = np.isin(tracer_IDs_sub, IDs_past)
            isin_past = np.isin(IDs_past, tracer_IDs_sub)
            
            IDs_group1 = tracer_IDs_sub[isin_now]
            IDs_group2 = tracer_IDs_sub[~isin_now]
            IDs_group3 = IDs_past[~isin_past]
            
            indices_group1 = tracer_indices_sub[isin_now]
            indices_group2 = tracer_indices_sub[~isin_now]
            # find the indices of group3 tracers later
            # the tracer indices vary with each snapshot, so we can't use the indices from the previous snaps
            # save -1 now, such that we only need to find the indices once per snapshot (rather than per subhalo)
            indices_group3 = np.ones(len(IDs_group3), dtype=int) * -1
            
            tracer_IDs_sub = np.concatenate([IDs_group1, IDs_group2, IDs_group3])
            tracer_indices_sub = np.concatenate([indices_group1, indices_group2, indices_group3])
            
            # save the info in the offsets catalog
            offsets_subhalo['SubhaloLength'][subfind_i] = len(tracer_IDs_sub) # all three groups
            offsets_subhalo['SubhaloLengthColdGas'][subfind_i] = len(IDs_group1) + len(IDs_group2) # groups 1 and 2
            offsets_subhalo['SubhaloLengthColdGas_new'][subfind_i] = len(IDs_group2) # group 2
        
        # end cross match with previous snapshot(s)
        
        start       = offsets_subhalo['SubhaloOffset'][subfind_i]
        length      = offsets_subhalo['SubhaloLength'][subfind_i]
        length_cgas = offsets_subhalo['SubhaloLengthColdGas'][subfind_i]
        
        # save the tracerIDs and indices
        tracers_subhalo['TracerIDs'][start:start+length] = tracer_IDs_sub
        tracers_subhalo['TracerIndices'][start:start+length] = tracer_indices_sub

        # find the appropriate particles indices for groups 1, 2
        part_indices = particle_indices[sub_indices]
        tracers_subhalo['ParentIndices'][start:start+length_cgas] = part_indices
        tracers_subhalo['ParentPartType'][start:start+length_cgas] = np.ones(len(part_indices), dtype=int) * gas_ptn
        tracers_subhalo['ParentGasTemp'][start:start+length_cgas] = temperatures[sub_indices]

        # for now, make the parent indices and parent part type -1
        # then later load the other baryonic particles in the sim and match
        # their particle IDs with all unmatched tracers
        tracers_subhalo['ParentIndices'][start+length_cgas:start+length]  = np.ones((length - length_cgas), dtype=int) * -1
        tracers_subhalo['ParentPartType'][start+length_cgas:start+length] = np.ones((length - length_cgas), dtype=int) * -1
        tracers_subhalo['ParentGasTemp'][start+length_cgas:start+length]  = np.ones((length - length_cgas), dtype=float) * -1
    
    # finish loop over subfindIDs    

    # reshape the tracers_subhalo arrays
    end = offsets_subhalo['SubhaloOffset'][-1] + offsets_subhalo['SubhaloLength'][-1]
    for key in tracers_subhalo.keys():
        tracers_subhalo[key] = tracers_subhalo[key][:end]
        
    save_catalogs(offsets_subhalo, tracers_subhalo, snap)
            
    return


def find_coldgascells(subfindIDs, snap):
    """
    Find the cold gas cells bound to the subfindIDs at snap.
    Loads all gas cells at snap, and then loops over the subfindIDs 
    to find the appropriate slices.
    Returns are come in two flavors: 
    ParticleIDs, Particle_indices, and Temperatures contain information
    on the cell by cell basis. i.e., when loading all gas cells in the sim., 
    gas_cells['Particle_indices[i]'] has ParticleID ParticleIDs[i] and 
    temperature Temperatures[i].
    offsets, lengths contain information on the subhalo by subhalo basis. 
    The cold gas cells belonging to subhalo subfindIDs[i] of the first 
    three returns are the slice [offsets[i]:offsets[i]+lengths[i]]. 
    
    Returns ParticleIDs, Particle_indices, Temperatures, offsets, lengths.
    """
    
    a = time.time()
    gas_cells = il.snapshot.loadSubset(basePath, snap, gas_ptn, fields=gas_fields)
    b = time.time()

    gas_cells = ru.calc_temp_dict(gas_cells)

    c = time.time()

    dtype = int
    offsets = np.zeros(len(subfindIDs), dtype=dtype)
    lengths = np.zeros(len(subfindIDs), dtype=dtype)

    dtype = gas_cells['ParticleIDs'].dtype
    ParticleIDs = np.zeros(big_array_length, dtype=dtype)
    Particle_indices = np.zeros(big_array_length, dtype=int)

    tcoldgas = 10.**(4.5) # [K]

    for i, subfindID in enumerate(subfindIDs):

        if i > 0:
            offsets[i] = offsets[i-1] + lengths[i-1]
            
        if subfindID == -1:
            continue
            
        r = il.snapshot.getSnapOffsets(basePath, snap, subfindID, 'Subhalo')
        start = r['offsetType'][gas_ptn]
        end = r['lenType'][gas_ptn]

        temps = gas_cells['Temperature'][start:start+end]
        indices = temps <= tcoldgas

        lengths[i] = len(indices[indices])

        # check if there are any bound gas cells
        if end == 0:
            continue
            
        ParticleIDs[offsets[i]:offsets[i]+lengths[i]] = gas_cells['ParticleIDs'][start:start+end][indices]
        Particle_indices[offsets[i]:offsets[i]+lengths[i]] = start + np.where(indices)[0]

    stop = offsets[-1] + lengths[-1]
    ParticleIDs = ParticleIDs[:stop]
    Particle_indices = Particle_indices[:stop]
    Temperatures = gas_cells['Temperature'][Particle_indices]
    d = time.time()

    if np.min(ParticleIDs) <= 0:
        print('Warning')
    
    return ParticleIDs, Particle_indices, Temperatures, offsets, lengths


# from Dylan Nelson
def match3(ar1, ar2, firstSorted=False, parallel=False):
    """ Returns index arrays i1,i2 of the matching elements between ar1 and ar2. While the elements of ar1 
        must be unique, the elements of ar2 need not be. For every matched element of ar2, the return i1 
        gives the index in ar1 where it can be found. For every matched element of ar1, the return i2 gives 
        the index in ar2 where it can be found. Therefore, ar1[i1] = ar2[i2]. The order of ar2[i2] preserves 
        the order of ar2. Therefore, if all elements of ar2 are in ar1 (e.g. ar1=all TracerIDs in snap, 
        ar2=set of TracerIDs to locate) then ar2[i2] = ar2. The approach is one sort of ar1 followed by 
        bisection search for each element of ar2, therefore O(N_ar1*log(N_ar1) + N_ar2*log(N_ar1)) ~= 
        O(N_ar1*log(N_ar1)) complexity so long as N_ar2 << N_ar1. """
    if not isinstance(ar1,np.ndarray): ar1 = np.array(ar1)
    if not isinstance(ar2,np.ndarray): ar2 = np.array(ar2)
    assert ar1.ndim == ar2.ndim == 1
    
    if not firstSorted:
        # need a sorted copy of ar1 to run bisection against
        if parallel:
            index = p_argsort(ar1)
        else:
            index = np.argsort(ar1)
        ar1_sorted = ar1[index]
        ar1_sorted_index = np.searchsorted(ar1_sorted, ar2)
        ar1_sorted = None
        ar1_inds = np.take(index, ar1_sorted_index, mode="clip")
        ar1_sorted_index = None
        index = None
    else:
        # if we can assume ar1 is already sorted, then proceed directly
        ar1_sorted_index = np.searchsorted(ar1, ar2)
        ar1_inds = np.take(np.arange(ar1.size), ar1_sorted_index, mode="clip")

    mask = (ar1[ar1_inds] == ar2)
    ar2_inds = np.where(mask)[0]
    ar1_inds = ar1_inds[ar2_inds]

    if not len(ar1_inds):
        return None,None

    return ar1_inds, ar2_inds


def find_unmatched_tracers(snap):
    """
    For all unmatched tracers at snap, find their parents.
    """

    # load the offsets, tracers at snap
    a = time.time()
    offsets_subhalo, tracers_subhalo = load_catalogs(snap)
    b = time.time()

    # find the unmatched tracers
    unmatched_indices = tracers_subhalo['ParentPartType'] == -1
    unmatched_TracerIDs = tracers_subhalo['TracerIDs'][unmatched_indices]
    c = time.time()

    # load all tracers at snap
    tracers = il.snapshot.loadSubset(basePath, snap, tracer_ptn)

    # match unmatched tracerIDs to all tracers at snap to save indices and ParentIDs
    # NB: unmatched_tracerIDs is not necessarily unique
    tracers_indices, matched_unmatched_indices = match3(tracers['TracerID'], unmatched_TracerIDs)
    unmatched_ParentIDs = tracers['ParentID'][tracers_indices]

    # del simulation tracers before loading baryonic particles
    del tracers

    # loop over each baryonic particle type, searching for the parents

    for i, ptn in enumerate(bary_ptns):

        if ptn == gas_ptn:
            fields = gas_fields
        else:
            fields = part_fields

        # load all particles at snap
        Particles   = il.snapshot.loadSubset(basePath, snap, ptn, fields=fields, sq=False)
        ParticleIDs = Particles['ParticleIDs']

        # match ParentIDs with unmatched_ParentIDs
        # reminder that unmatched_ParentIDs are not unique
        Particle_indices, Parent_indices = match3(ParticleIDs, unmatched_ParentIDs)

        if Particle_indices is None:
            print('Warning, no matched parents for part type %d. Continuing to the next bary_ptn'%ptn)
            continue

        parent_indices = Particle_indices
        parent_ptn = np.ones(len(parent_indices), dtype=int) * ptn

        # calculate the temperature for the gas cells
        if ptn == gas_ptn:
            for key in fields:
                Particles[key] = Particles[key][parent_indices]
            Particles['count'] = len(parent_indices)
            Particles = ru.calc_temp_dict(Particles)
            temps = Particles['Temperature']
        else:
            temps = np.ones(len(parent_indices), dtype=float) * -1.

        # calculate the tracer catalog indices for the unmatched tracers matched to this part type
        save_indices = np.where(unmatched_indices)[0][matched_unmatched_indices][Parent_indices]

        # save the parent part types, indices, and the temperatures (temps only for gas cells)
        tracers_subhalo['ParentIndices'][save_indices]  = parent_indices
        tracers_subhalo['ParentPartType'][save_indices] = parent_ptn
        tracers_subhalo['ParentGasTemp'][save_indices]  = temps
    # finish loop over bary_ptns
    
    # test -- ensure that all part types were found
    unfound_indices = tracers_subhalo['ParentPartType'] == -1
    if len(unfound_indices[unfound_indices]) != 0:
        print('Warning, not all parents found!')
        
    # save catalogs and return 
    save_catalogs(offsets_subhalo, tracers_subhalo, snap)
        
    return


def track_subfindIDs(subfindIDs, z0_flag=True):
    """
    Given the list of subhalo subfindIDs at either z0 (default) or 
    at snapNum, use either the MPB (default) or MDB to find the 
    corresponding subfindIDs at the other snapshots between snapNum and 99. 
    Be careful at subhalos that don't exist in the trees or skip snaps.
    """

    # initialize result 
    snaps    = np.arange(max_snap, snapNum-1, -1)
    n_snaps  = len(snaps)
    result   = np.ones((len(subfindIDs), n_snaps), dtype=int) * -1

    fields   = ['SubfindID', 'SnapNum']
    treeName = 'SubLink_gal'
    
    # begin loop over subfindIDs
    for i, subfindID in enumerate(subfindIDs):
        
        if (z0_flag):
            tree = ru.loadMainTreeBranch(max_snap, subfindID, sim=sim, treeName=treeName,
                                         fields=fields, min_snap=snapNum, max_snap=max_snap)
        else:
            tree = ru.loadMainTreeBranch(snapNum, subfindID, sim=sim, treeName=treeName,
                                         fields=fields, min_snap=snapNum, max_snap=max_snap)
            
        if not tree:
            continue

        # find at which snaps the subhalo was identified
        isin = np.isin(snaps, tree['SnapNum'])

        # and record the result
        result[i,isin] = tree['SubfindID']

    # finish loop over subfindIDs
    # save by looping over the snapshots
    
    for i, snap in enumerate(snaps):

        outfname = 'offsets_%03d.hdf5'%(snap)

        dset = result[:,i]
        
        with h5py.File(outdirec + outfname, 'a') as outf:
            group = outf.require_group('group')

            dataset = group.require_dataset('SubfindID', shape=dset.shape, dtype=dset.dtype)
            dataset[:] = dset

            outf.close()
        
    # finish loop over snaps. return to main function
            
    return


def initialize_outputs(Nsubhalos):
    """
    Given the global variable big_array_length, initalize the final outputs.
    """
    
    # initialize the offset dictionary
    offsets_subhalo = {}
    offsets_subhalo['SubhaloOffset']            = np.zeros(Nsubhalos, int)
    offsets_subhalo['SubhaloLength']            = np.zeros(Nsubhalos, int)
    offsets_subhalo['SubhaloLengthColdGas']     = np.zeros(Nsubhalos, int)
    offsets_subhalo['SubhaloLengthColdGas_new'] = np.zeros(Nsubhalos, int)

    tracers_subhalo = {}
    # rewrite into a 4xbig_array_length array rather than a dictionary
    # for a speed increase
    tracers_subhalo['TracerIDs']      = np.empty(big_array_length, dtype=int)
    tracers_subhalo['TracerIndices']  = np.empty(big_array_length, dtype=int)
    tracers_subhalo['ParentIndices']  = np.empty(big_array_length, dtype=int)
    tracers_subhalo['ParentPartType'] = np.empty(big_array_length, dtype=int)
    tracers_subhalo['ParentGasTemp']  = np.empty(big_array_length, dtype=float)
    
    return offsets_subhalo, tracers_subhalo

def load_catalogs(snap):
    """
    Load the offsets and tracers catalogs at snap.
    Returns dicitonaries of the offsets, tracers catalogs.
    """
    result = {}
    fnames = ['offsets', 'tracers']
    for i, fname in enumerate(fnames):
        result[fname] = {}
        with h5py.File(outdirec + '%s_%03d.hdf5'%(fname, snap), 'r') as f:
            group = f['group']
            for key in group.keys():
                result[fname][key] = group[key][:]
        f.close()
        
    return result[fnames[0]], result[fnames[1]]    


def save_catalogs(offsets, tracers, snap):
    """
    Save the offsets and tracers catalogs at snap.
    """
    dicts  = [offsets, tracers]
    fnames = ['offsets', 'tracers']
    for d_i, d in enumerate(dicts):
        fname    = fnames[d_i]
        outfname = '%s_%03d.hdf5'%(fname, snap)

        with h5py.File(outdirec + outfname, 'a') as outf:
            group = outf.require_group('group')
            for dset_key in d.keys():
                dset = d[dset_key]
                dataset = group.require_dataset(dset_key, shape=dset.shape, dtype=dset.dtype)
                dataset[:] = dset

            outf.close()
            
    return


create_tracertracks()

