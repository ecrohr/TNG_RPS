### import modules
import illustris_python as il
import multiprocessing as mp
import numpy as np
import h5py
import rohr_utils as ru 
from importlib import reload
import glob
import time

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
    
    a = time.time()
    
    # define the global variables
    sim        = 'TNG50-3'
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

    big_array_length = int(1e6)

    # define the subhalos we care about at snapshot snapNum
    subfindIDs = range(3)

    outdirec = '../Output/%s_tracers_%d-%d/'%(sim,subfindIDs[0],subfindIDs[-1])
    print(outdirec)
    if not os.path.isdir(outdir):
        os.system('mkdir %s'%outdirec)

    # find the corresponding subfindIDs at the next snapshots
    track_subfindIDs(subfindIDs)

    b = time.time()
    print('create_tracertracks: Initalization: %.2g s'%(b-a))
    
    # now track tracers from snapNum + 1 until snap 99
    for snap in range(snapNum, snapNum+10):
        match_tracers(snap)

    # and find the unmatched tracers from snapNum + 1 until snap 99
    #for snap in range(snapNum+1, max_snap+1):
    #    find_unmatched_tracers(snap)

    return


def match_tracers(snap):
    """
    Match the bound cold gas cells of the subfindIDs at snap to the tracers.
    If snap < snapNum, then loads the previous tracers and checks if they still
    belong to the same subhalo. 
    Saves offsets and tracers catalogs, but does not find the unmatched tracers -- 
    i.e., does not search for tracers whose parents were bound cold gas cells but
    no longer are (group 3 tracers in the code below).
    No returns.
    """
    
    # load the subfindIDs
    with h5py.File(outdirec + 'offsets_%03d.hdf5'%snap, 'r') as f:
        subfindIDs = f['group']['SubfindID'][:]
        f.close()
      
    # initialize the outputs
    offsets_subhalo, tracers_subhalo = initialize_outputs(len(subfindIDs))
    
    # if snap > snapNum: load previous tracers
    if (snap > snapNum):
        offsets_past = h5py.File(outdirec + 'offsets_%03d.hdf5'%(snap - 1), 'r')
        tracers_past = h5py.File(outdirec + 'tracers_%03d.hdf5'%(snap - 1), 'r')
        
    # find the gas cells of interest
    a = time.time()
    ParticleIDs, Particle_indices, Temperatures, offsets, lengths = find_coldgascells(subfindIDs, snap)
    b = time.time()

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

        indices = ((indices1 >= start) & (indices1 < stop))
        Ntracers = len(indices[indices])
        offsets_subhalo['SubhaloLength'][subfind_i]            = Ntracers
        offsets_subhalo['SubhaloLengthColdGas'][subfind_i]     = Ntracers
        offsets_subhalo['SubhaloLengthColdGas_new'][subfind_i] = Ntracers
        
        # if snap > SnapNum: cross match with previous snapshot(s)
        if snap > snapNum:
            ### check which cold gas cell tracers from previous snap are still here ###

            # check if the subhalo was defined at the previous snap
            if offsets_past['group']['SubfindID'][subfind_i] != -1:
                start = offsets_past['group']['SubhaloOffset'][subfind_i]
                end   = start + offsets_past['group']['SubhaloLengthColdGas'][subfind_i]

            # if not, then try the previous snapshots
            else:
                i = 2
                while ((snap - i) >= snapNum) & (i < 4):
                    with h5py.File(outdirec + 'offsets_%03d.hdf5'%(snap - i), 'r') as f:
                        if f['group']['SubfindID'][subfind_i] == -1:
                            f.close()
                            i += 1
                        else:
                            start = f['group']['SubhaloOffset'][subfind_i]
                            end   = start + f['group']['SubhaloLengthColdGas'][subfind_i]
                            f.close()
                            break
                    # if the subhalo isn't in the merger trees the past few snaps, move on
                    if i == 4:
                        start = end = 0

            IDs_past = tracers_past['group']['TracerIDs'][start:end]
            indices_past = tracers_past['group']['TracerIndices'][start:end]

            isin_now  = np.isin(tracer_IDs, IDs_past)
            isin_past = np.isin(IDs_past, tracer_IDs)

            # reorder tracer_IDs and tracer_indices such that:
            # 1st: cold gas tracers found in both snapshots
            # 2nd: new cold gas tracers (not in the previous snapshot)
            # 3rd: no longer cold gas tracers (was in previous snapshot but no longer)

            # find the indices of group 3
            isin_group3    = np.isin(tracers['TracerID'], IDs_past[~isin_past])
            IDs_group3     = tracers['TracerID'][isin_group3]
            indices_group3 = np.where(isin_group3)[0]

            IDs = np.concatenate([tracer_IDs[isin_now],
                                  tracer_IDs[~isin_now],
                                  IDs_group3])

            indices = np.concatenate([tracer_indices[isin_now],
                                      tracer_indices[~isin_now],
                                      indices_group3])

            # save the offset information
            offsets_subhalo['SubhaloLength'][subfind_i]            = len(IDs)
            offsets_subhalo['SubhaloLengthColdGas'][subfind_i]     = len(tracer_IDs)
            offsets_subhalo['SubhaloLengthColdGas_new'][subfind_i] = len(tracer_IDs[~isin_now])
        # end cross match with previous snapshot(s)

        start       = offsets_subhalo['SubhaloOffset'][subfind_i]
        length      = offsets_subhalo['SubhaloLength'][subfind_i]
        length_cgas = offsets_subhalo['SubhaloLengthColdGas'][subfind_i]

        # save the tracerIDs and indices
        tracers_subhalo['TracerIDs'][start:start+length] = tracer_IDs[indices]
        tracers_subhalo['TracerIndices'][start:start+length] = tracer_indices[indices]

        # find the appropriate particles indices
        part_indices = particle_indices[indices]
        tracers_subhalo['ParentIndices'][start:start+length_cgas] = part_indices
        tracers_subhalo['ParentPartType'][start:start+length_cgas] = np.ones(len(part_indices), dtype=int) * gas_ptn
        tracers_subhalo['ParentGasTemp'][start:start+length_cgas] = temperatures[indices]

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

    # close previous offsets and tracers files
    if offsets_past:
        offsets_past.close()
        tracers_past.close()
        
    # save the offsets and particles dictionaries
    dicts  = [offsets_subhalo, tracers_subhalo]
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

def match_subhalo_tracers(subfind_i, subfindID, snap, tracers, offsets_subhalo, tracers_subhalo,
                          offsets_past=None, tracers_past=None):
    """
    Match the bound gas cells of a given subhalo to the tracerrs at snap.
    Returns the offsets_subahlo, tracers_subhalo dictionaries.
    """
    
    print('Working on %s snapshot %d subfindID %d'%(sim, snap, subfindID))
    
    # load offsets_past, tracers_past if snap > snapNum and not already done
    if (snap > snapNum) and not offsets_past and not tracers_past:
        offsets_past = h5py.File(outdirec + 'offsets_%03d.hdf5'%(snap - 1), 'r')
        tracers_past = h5py.File(outdirec + 'tracers_%03d.hdf5'%(snap - 1), 'r')

    if subfind_i == 0:
        offsets_subhalo['SubhaloOffset'][subfind_i] = 0
    else:
        offsets_subhalo['SubhaloOffset'][subfind_i] = (offsets_subhalo['SubhaloOffset'][subfind_i-1]
                                                       + offsets_subhalo['SubhaloLength'][subfind_i-1])
    # check that the subhalo is identified at snapNum
    if subfindID == -1:
        return offsets_subhalo, tracers_subhalo
    
    a = time.time()
    gas_cells = il.snapshot.loadSubhalo(basePath, snap, subfindID, gas_ptn, fields=gas_fields)
    b = time.time()
    print('match_subhalo_tracers: Load subhalo gas cells: %.2g s'%(b-a))
    
    # check if there are any gas cells
    if gas_cells['count'] == 0:
        return offsets_subhalo, tracers_subhalo
            
    gas_cells = ru.calc_temp_dict(gas_cells)
    c = time.time()
    print('match_subhalo_tracers: Calc gas cell temps: %.2g s'%(c-b))
    
    # find the local indices and load the global offset for these gas cells
    cgas_indices = np.where(gas_cells['Temperature'] <= tcoldgas)[0]
    if len(cgas_indices) == 0:
        return offsets_subhalo, tracers_subhalo
    d = time.time()
    print('match_subhalo_tracers: Find cold gas cell indices: %.2g s'%(d-c))
    
    ParticleIDs = gas_cells['ParticleIDs'][cgas_indices]

    # match the tracer ParentID with the cold gas cells ParticleIDs
    isin_tracer = np.isin(tracers['ParentID'], ParticleIDs)
    e = time.time()
    print('match_subhalo_tracers: Match tracers with gas cells: %.2g s'%(e-d))

    # save the tracerIDs and tracer indices at snapshot snapNum
    tracer_IDs = tracers['TracerID'][isin_tracer]
    tracer_indices = np.where(isin_tracer)[0]
    
    offsets_subhalo['SubhaloLength'][subfind_i]            = len(tracer_indices)
    offsets_subhalo['SubhaloLengthColdGas'][subfind_i]     = len(tracer_indices)
    offsets_subhalo['SubhaloLengthColdGas_new'][subfind_i] = len(tracer_indices)
    f = time.time()
    print('match_subhalo_tracers: Save offsets info: %.2g s'%(f-e))
    
    IDs = tracer_IDs
    indices = tracer_indices
    
    if snap > snapNum:
        ### check which cold gas cell tracers from previous snap are still here ###

        # check if the subhalo was defined at the previous snap
        if offsets_past['group']['SubfindID'][subfind_i] != -1:
            start = offsets_past['group']['SubhaloOffset'][subfind_i]
            end   = start + offsets_past['group']['SubhaloLengthColdGas'][subfind_i]

        # if not, then try the previous snapshots
        else:
            i = 2
            while ((snap - i) >= snapNum) & (i < 4):
                with h5py.File(outdirec + 'offsets_%03d.hdf5'%(snap - i), 'r') as f:
                    if f['group']['SubfindID'][subfind_i] == -1:
                        f.close()
                        i += 1
                    else:
                        start = f['group']['SubhaloOffset'][subfind_i]
                        end   = start + f['group']['SubhaloLengthColdGas'][subfind_i]
                        f.close()
                        break
                # if the subhalo isn't in the merger trees the past few snaps, move on
                if i == 4:
                    start = end = 0
            
        IDs_past = tracers_past['group']['TracerIDs'][start:end]
        indices_past = tracers_past['group']['TracerIndices'][start:end]

        isin_now  = np.isin(tracer_IDs, IDs_past)
        isin_past = np.isin(IDs_past, tracer_IDs)

        # reorder tracer_IDs and tracer_indices such that:
        # 1st: cold gas tracers found in both snapshots
        # 2nd: new cold gas tracers (not in the previous snapshot)
        # 3rd: no longer cold gas tracers (was in previous snapshot but no longer)

        # find the indices of group 3
        isin_group3    = np.isin(tracers['TracerID'], IDs_past[~isin_past])
        IDs_group3     = tracers['TracerID'][isin_group3]
        indices_group3 = np.where(isin_group3)[0]

        IDs = np.concatenate([tracer_IDs[isin_now],
                              tracer_IDs[~isin_now],
                              IDs_group3])
                
        indices = np.concatenate([tracer_indices[isin_now],
                                  tracer_indices[~isin_now],
                                  indices_group3])
        
        # save the offset information
        offsets_subhalo['SubhaloLength'][subfind_i]            = len(IDs)
        offsets_subhalo['SubhaloLengthColdGas'][subfind_i]     = len(tracer_IDs)
        offsets_subhalo['SubhaloLengthColdGas_new'][subfind_i] = len(tracer_IDs[~isin_now])

    
    # save the tracer IDs and indices
    start       = offsets_subhalo['SubhaloOffset'][subfind_i]
    length      = offsets_subhalo['SubhaloLength'][subfind_i]
    length_cgas = offsets_subhalo['SubhaloLengthColdGas'][subfind_i]
    
    # check that there are cold gas cells with tracers
    if length_cgas == 0:
        return offsets_subhalo, tracers_subhalo
    
    # for the cold gas cell tracers, save the parent IDs and tracers
    # get the local cold gas indices with matched tracer particles and include the global offset
    parent_IDs  = tracers['ParentID'][indices[:length_cgas]]
    isin_gas    = np.isin(ParticleIDs, parent_IDs)

    r           = il.snapshot.getSnapOffsets(basePath, snap, subfindID, "Subhalo")
    offset      = r['offsetType'][gas_ptn]
    gas_indices = offset + cgas_indices[isin_gas]

    # note that some of these indices need to be repeated due to having multiple tracers with the same parent
    gas_IDs        = ParticleIDs[isin_gas]
    g = time.time()
    print('match_subhalo_tracers: Start finding matched gas cell indices: %.2g s'%(g-f))

    # find a way to optimize the following line... 
    repeat_indices = np.where([parent_ID == gas_IDs for parent_ID in parent_IDs])[1]
    h = time.time()
    print('match_subhalo_tracers: Find repeat indices: %.2g s'%(h-g))
    gas_indices    = gas_indices[repeat_indices]

    # note that the parent type is always gas
    parent_ptn = np.ones(len(gas_indices), dtype=int) * gas_ptn

    temps = gas_cells['Temperature'][cgas_indices][isin_gas][repeat_indices]
    
    # fill in the particle dictionary for this subhalo
    tracers_subhalo['TracerIDs'][start:start+length]     = IDs
    tracers_subhalo['TracerIndices'][start:start+length] = indices

    tracers_subhalo['ParentIndices'][start:start+length_cgas]  = gas_indices
    tracers_subhalo['ParentPartType'][start:start+length_cgas] = parent_ptn
    tracers_subhalo['ParentGasTemp'][start:start+length_cgas]  = temps

    # for now, make the parent indices and parent part type -1
    # then later load the other baryonic particles in the sim and match
    # their particle IDs with all unmatched tracers
    tracers_subhalo['ParentIndices'][start+length_cgas:start+length]  = np.ones((length - length_cgas), dtype=int) * -1
    tracers_subhalo['ParentPartType'][start+length_cgas:start+length] = np.ones((length - length_cgas), dtype=int) * -1
    tracers_subhalo['ParentGasTemp'][start+length_cgas:start+length]  = np.ones((length - length_cgas), dtype=float) * -1
    i = time.time()
    print('match_subhalo_tracers: Finish finding gas cell indices + save tracers: %.g s'%(i-h))
    
    return offsets_subhalo, tracers_subhalo


def track_tracers(snap):
    """
    Finds the tracers whose parents are cold gas cells gravitationally 
    bound to subhalos in subfindIDs at snapshot snap
    
    architecture: 
    given the list of subhalo subfindIDs at snap
    load the tracers at this snap
    if snap > snapNum:
        load the tracers from the past snapshot
    for each subhalo in subfindIDs:
        load subfind cold gas cells
        cross match with tracers at this snap
        if snap > snapNum:
            cross match with tracers from previous snap
        save matched tracers for the cold gas cells
    save all info
    return to main function and continue loop over snapshots 
    """
    a = time.time()
    # load the subfindIDs from the offsets file
    with h5py.File(outdirec + 'offsets_%03d.hdf5'%snap, 'r') as f:
        subfindIDs = f['group']['SubfindID'][:]
        f.close()
    b = time.time()
    print('track_tracers: Load subfindIDs: %.2g s'%(b-a))
        
    # initialize the outputs 
    offsets_subhalo, tracers_subhalo = initialize_outputs(len(subfindIDs))
    c = time.time()
    print('track_tracers: Initalize outputs: %.2g s'%(c-b))
    
    # load every tracer particle in the simulation at this snapshot
    tracers = il.snapshot.loadSubset(basePath, snap, tracer_ptn)
    d = time.time()
    print('track_tracers: Load tracers: %.2g s'%(d-c))

    # load the offsets and tracers from the previous snapshot
    if snap > snapNum:
        offsets_past = h5py.File(outdirec + 'offsets_%03d.hdf5'%(snap - 1), 'r')
        tracers_past = h5py.File(outdirec + 'tracers_%03d.hdf5'%(snap - 1), 'r')
        
    else:
        offsets_past = None
        tracers_past = None
    
    # begin loop over the subhalos at snapshot snap
    for subfind_i, subfindID in enumerate(subfindIDs):
        offsets_subhalo, tracers_subhalo = match_subhalo_tracers(subfind_i, subfindID, snap,
                                                                 tracers, offsets_subhalo, tracers_subhalo,
                                                                 offsets_past=offsets_past, tracers_past=tracers_past)
    
    e = time.time()
    print('track_tracers: Finish loop over all subhalos: %.2g s'%(e-d))
    # end loop over subhalos
    # close previous offsets and tracers files
    if offsets_past:
        offsets_past.close()
        tracers_past.close()
    
    # reshape the tracers_subhalo arrays
    end = offsets_subhalo['SubhaloOffset'][-1] + offsets_subhalo['SubhaloLength'][-1]
    for key in tracers_subhalo.keys():
        tracers_subhalo[key] = tracers_subhalo[key][:end]
        
    # save the offsets and particles dictionaries
    dicts  = [offsets_subhalo, tracers_subhalo]
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
            
    f = time.time()
    print('track_tracers: Save all info: %.2g s'%(f-e))

    return 


def find_unmatched_tracers(snap):
    """
    For all unmatched tracers at snap, find their parents.
    """
    
    # load the tracers at snap
    tracers = il.snapshot.loadSubset(basePath, snap, tracer_ptn)
            
    # load the tracers_subhalo catalog at this snap
    tracers_subhalo = {}
    fname = 'tracers'
    outfname = '%s_%03d.hdf5'%(fname, snap)

    with h5py.File(outdirec + outfname, 'a') as outf:
        group = outf.require_group('group')
        for dset_key in group.keys():
            tracers_subhalo[dset_key] = group[dset_key][:]
        outf.close()

    # find the parentIDs of the group 3 tracers, 
    # i.e. the ones whose parents are no longer bound cold gas cells

    unmatched_indices = tracers_subhalo['ParentPartType'] == -1
    tracer_indices    = tracers_subhalo['TracerIndices'][unmatched_indices]
    ParentIDs         = tracers['ParentID'][tracer_indices]

    # close the tracer snapshot data before loading in baryonic snapshot data
    del tracers

    # for each baryonic particle type, load the snapshot data
    for i, ptn in enumerate(bary_ptns):

        if i == 0:
            fields = gas_fields
        else:
            fields = part_fields

        particles   = il.snapshot.loadSubset(basePath, snap, ptn, fields=fields, sq=False)
        ParticleIDs = particles['ParticleIDs']

        isin_tracer    = np.isin(ParentIDs, ParticleIDs)
        isin_particles = np.isin(ParticleIDs, ParentIDs)

        # if there are no matches, then continue
        if len(isin_tracer[isin_tracer]) == 0:
            continue
        
        parent_IDs       = ParentIDs[isin_tracer]
        particle_indices = np.where(isin_particles)[0]

        # note that some of these indices need to be repeated due to having multiple tracers with the same parent
        particle_IDs   = ParticleIDs[isin_particles]
        repeat_indices = np.where([parent_ID == particle_IDs for parent_ID in parent_IDs])[1]

        parent_indices = particle_indices[repeat_indices]
        parent_ptn     = np.ones(len(parent_indices), dtype=int) * ptn

        # for gas cells, also calculate the temperature
        if i == 0:
            for key in particles.keys():
                if key == 'count':
                    particles[key] = len(parent_indices)
                else:
                    particles[key] = particles[key][parent_indices]

            particles = ru.calc_temp_dict(particles)
            temps = particles['Temperature']

        else:
            temps = np.ones(len(parent_indices), dtype=float) * -1

        save_indices = np.where(unmatched_indices)[0][np.where(isin_tracer)[0]]

        # save the parent part types and indices, and the temperatures for parents that are gas cells
        tracers_subhalo['ParentIndices'][save_indices]  = parent_indices
        tracers_subhalo['ParentPartType'][save_indices] = parent_ptn
        tracers_subhalo['ParentGasTemp'][save_indices]  = temps
    # finish loop finding the unmatched tracers

    # save the offsets and tracers_subhalo dictionaries
    d = tracers_subhalo
    
    with h5py.File(outdirec + outfname, 'a') as outf:
        group = outf.require_group('group')
        for dset_key in d.keys():
            dset = d[dset_key]
            dataset = group.require_dataset(dset_key, shape=dset.shape, dtype=dset.dtype)
            dataset[:] = dset

        outf.close()

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



create_tracertracks()

