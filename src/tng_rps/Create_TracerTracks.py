### import modules
import illustris_python as il
import multiprocessing as mp
import numpy as np
import h5py
import rohr_utils as ru 
from importlib import reload
import glob

global sim, basePath, snapNum, tcoldgas
global tracer_ptn, star_ptn, gas_ptn, bh_ptn, bary_ptns
global gas_fields, part_fields
global big_array_length


def create_tracertracks():
    """
    Run the Create_TracerTracks.py file. 
    Starts at snapNum with a list of subhalo subfindIDs,
    find the descendants, and loops over the snapshots
    matching the tracers from bound cold gas cells.
    Saves the output.
    """

    global sim, basePath, snapNum, tcoldgas
    global tracer_ptn, star_ptn, gas_ptn, bh_ptn, bary_ptns
    global gas_fields, part_fields
    global big_array_length

    
    # define the global variables
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

    big_array_length = int(1e8)

    # define the subhalos we care about 
    subfindIDs = np.arange(10)

    # find the corresponding subfindIDs at the next snapshots
    track_subfindIDs(subfindIDs)

    # at snapNum, initialize the tracers we care about
    initialize_coldgastracers(subfindIDs, snapNum)

    # now track tracers from snapNum + 1 until snap 99
    for snap in range(snapNum+1, 100):
        track_tracers(subfindIDs, snap)

    # finish the first snapshot. move to the next.

    return 



def initialize_coldgastracers(subfindIDs, snap):
    """
    Finds the tracers whose parents are cold gas cells gravitationally 
    bound to subhalos in subfindIDs at snapshot snapNum
    
    architecture: 
    given the list of subhalo subfindIDs at snap: 
    load the tracers at this snap;
    for each subhalo in subfindIDs:
        load subfind cold gas cells
        cross match with tracers 
        save matched tracers 
    save all info
    return to main function and continue to next snaps
    """

    # initialize the outputs 

    # initialize the offset dictionary
    offsets_subhalo = {}
    offsets_subhalo['SubhaloOffset']        = np.zeros(len(subfindIDs), int)
    offsets_subhalo['SubhaloLength']        = np.zeros(len(subfindIDs), int)
    offsets_subhalo['SubhaloLengthColdGas'] = np.zeros(len(subfindIDs), int)

    particles = {}
    # rewrite into a 4xbig_array_length array rather than a dictionary
    # for a speed increase
    particles['TracerIDs']      = np.empty(big_array_length, dtype=int)
    particles['TracerIndices']  = np.empty(big_array_length, dtype=int)
    particles['ParentIndices']  = np.empty(big_array_length, dtype=int)
    particles['ParentPartType'] = np.empty(big_array_length, dtype=int)

    # load every tracer particle in the simulation at this snapshot
    tracers = il.snapshot.loadSubset(basePath, snapNum, tracer_ptn)

    # begin loop over the subhalos at snapshot snapNum
    for subfind_i, subfindID in enumerate(subfindIDs):

        gas_cells    = il.snapshot.loadSubhalo(basePath, snapNum, subfindID, gas_ptn, fields=gas_fields)

        # check if there are any gas cells
        if gas_cells['count'] == 0:
            continue
        
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
        offsets_subhalo['SubhaloLength'][subfind_i] = len(tracer_indices)
        offsets_subhalo['SubhaloLengthColdGas'][subfind_i] = len(tracer_indices)

        if subfind_i == 0:
            offsets_subhalo['SubhaloOffset'][subfind_i] = 0
        else:
            offsets_subhalo['SubhaloOffset'][subfind_i] = (offsets_subhalo['SubhaloOffset'][subfind_i-1]
                                                           + offsets_subhalo['SubhaloLength'][subfind_i-1])

        # save the corresponding gas cell indices
        # get the local cold gas indices with matched tracer particles and include the global offset
        parent_IDs  = tracers['ParentID'][isin_tracer]
        isin_gas    = np.isin(ParticleIDs, parent_IDs)

        r           = il.snapshot.getSnapOffsets(basePath, snapNum, subfindID, "Subhalo")
        offset      = r['offsetType'][gas_ptn]
        gas_indices = offset + cgas_indices[isin_gas]

        # note that some of these indices need to be repeated due to having multiple tracers with the same parent
        gas_IDs        = ParticleIDs[isin_gas]
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
    for key in particles.keys():
        particles[key] = particles[key][:end]

    # save the offsets and particles dictionaries
    dicts  = [offsets_subhalo, particles]
    fnames = ['offsets', 'tracers']
    for d_i, d in enumerate(dicts):
        fname    = fnames[d_i]
        outfname = '%s_%03d.hdf5'%(fname, snapNum)
        outdirec = '../Output/%s_tracers/'%(sim)

        with h5py.File(outdirec + outfname, 'a') as outf:
            group = outf.require_group('group')
            for dset_key in d.keys():
                dset = d[dset_key]
                dataset = group.require_dataset(dset_key, shape=dset.shape, dtype=dset.dtype)
                dataset[:] = dset

            outf.close()

    
    return 


def track_tracers(subfindIDs, snap):
    """
    Finds the tracers whose parents are cold gas cells gravitationally 
    bound to subhalos in subfindIDs at snapshot snapNum
    
    architecture: 
    given the list of subhalo subfindIDs at snap: 
    load the tracers at this snap;
    for each subhalo in subfindIDs:
        load subfind cold gas cells
        cross match with tracers at this snap 
        cross match with tracers from previous snap
        save matched tracers for the cold gas cells
        find parents for the unmatched tracers from previous snap
    save all info
    return to main function and continue loop over snapshots 
    """

    # initialize the outputs 

    # initialize the offset dictionary
    offsets_subhalo = {}
    offsets_subhalo['SubhaloOffset']        = np.zeros(len(subfindIDs), int)
    offsets_subhalo['SubhaloLength']        = np.zeros(len(subfindIDs), int)
    offsets_subhalo['SubhaloLengthColdGas'] = np.zeros(len(subfindIDs), int)

    particles = {}
    # rewrite into a 4xbig_array_length array rather than a dictionary
    # for a speed increase
    particles['TracerIDs']      = np.empty(big_array_length, dtype=int)
    particles['TracerIndices']  = np.empty(big_array_length, dtype=int)
    particles['ParentIndices']  = np.empty(big_array_length, dtype=int)
    particles['ParentPartType'] = np.empty(big_array_length, dtype=int)

    # load the offsets and tracers from the previous snapshot
    outdirec     = '../Output/%s_tracers/'%(sim)
    offsets_past = h5py.File(outdirec + 'offsets_%03d.hdf5'%(snap - 1), 'r')
    tracers_past = h5py.File(outdirec + 'tracers_%03d.hdf5'%(snap - 1), 'r')

    # load every tracer particle in the simulation at this snapshot
    tracers = il.snapshot.loadSubset(basePath, snapNum, tracer_ptn)

    # begin loop over the subhalos at snapshot snapNum
    for subfind_i, subfindID in enumerate(subfindIDs):

        print('Working on %s snapshot %d subfindID %d'%(sim, snap, subfindID))

        ### cross match cold gas cells with tracers at this snap ###
        gas_cells    = il.snapshot.loadSubhalo(basePath, snap, subfindID, gas_ptn, fields=gas_fields)

        # check if there are any gas cells
        if gas_cells['count'] == 0:
            continue
        
        gas_cells    = ru.calc_temp_dict(gas_cells)

        # find the local indices and load the global offset for these gas cells
        cgas_indices = np.where(gas_cells['Temperature'] <= tcoldgas)[0]
        ParticleIDs  = gas_cells['ParticleIDs'][cgas_indices]

        # match the tracer ParentID with the cold gas cells ParticleIDs
        isin_tracer = np.isin(tracers['ParentID'], ParticleIDs)

        # save the tracerIDs and tracer indices at snapshot snapNum
        tracer_IDs = tracers['TracerID'][isin_tracer]
        tracer_indices = np.where(isin_tracer)[0]

        ### check which cold gas cell tracers from previous snap are still here ###
        start    = offsets_past['group']['SubhaloOffset'][subfind_i]
        end      = start + offsets_past['group']['SubhaloLengthColdGas'][subfind_i]
        IDs_past = tracers_past['group']['TracerIDs'][start:end]
        indices_past = tracers_past['group']['TracerIndices'][start:end]

        isin_now  = np.isin(tracer_IDs, IDs_past)
        isin_past = np.isin(IDs_past, tracer_IDs)
        #_, indices_now, indices_past = np.intersect1d(tracer_IDs, IDs_past, return_indices=True)

        # reorder tracer_IDs and tracer_indices such that:
        # 1st: cold gas tracers found in both snapshots
        # 2nd: new cold gas tracers (not in the previous snapshot)
        # 3rd: no longer cold gas tracers (was in previous snapshot but no longer)

        IDs = np.concatenate([tracer_IDs[isin_now],
                              tracer_IDs[~isin_now],
                              IDs_past[~isin_past]])
        
        indices = np.concatenate([tracer_indices[isin_now],
                                  tracer_indices[~isin_now],
                                  indices_past[~isin_past]])
        
        # save the offset information
        offsets_subhalo['SubhaloLength'][subfind_i]        = len(IDs)
        offsets_subhalo['SubhaloLengthColdGas'][subfind_i] = len(tracer_IDs)

        if subfind_i == 0:
            offsets_subhalo['SubhaloOffset'][subfind_i] = 0
        else:
            offsets_subhalo['SubhaloOffset'][subfind_i] = (offsets_subhalo['SubhaloOffset'][subfind_i-1]
                                                           + offsets_subhalo['SubhaloLength'][subfind_i-1])

        # save the tracer IDs and indices
        start       = offsets_subhalo['SubhaloOffset'][subfind_i]
        length      = offsets_subhalo['SubhaloLength'][subfind_i]
        length_cgas = offsets_subhalo['SubhaloLengthColdGas'][subfind_i]

        particles['TracerIDs'][start:start+length]     = IDs
        particles['TracerIndices'][start:start+length] = indices

        # for the cold gas cell tracers, save the parent IDs and tracers
        # get the local cold gas indices with matched tracer particles and include the global offset
        parent_IDs  = tracers['ParentID'][indices[:length_cgas]]

        isin_gas    = np.isin(ParticleIDs, parent_IDs)

        r           = il.snapshot.getSnapOffsets(basePath, snapNum, subfindID, "Subhalo")
        offset      = r['offsetType'][gas_ptn]
        gas_indices = offset + cgas_indices[isin_gas]

        # note that some of these indices need to be repeated due to having multiple tracers with the same parent
        gas_IDs        = ParticleIDs[isin_gas]
        # find a way to optimize the following line... 
        repeat_indices = np.where([parent_ID == gas_IDs for parent_ID in parent_IDs])[1]
        gas_indices    = gas_indices[repeat_indices]

        # note that the parent type is always gas
        parent_ptn = np.ones(len(gas_indices), dtype=int) * gas_ptn

        # fill in the particle dictionary for this subhalo
        particles['ParentIndices'][start:start+length_cgas]  = gas_indices
        particles['ParentPartType'][start:start+length_cgas] = parent_ptn

        # for now, make the parent indices and parent part type -1
        # then later load the other baryonic particles in the sim and match
        # their particle IDs with all unmatched tracers
        particles['ParentIndices'][start+length_cgas:start+length] = np.ones((length - length_cgas), dtype=int) * -1
        particles['ParentPartType'][start+length_cgas:start+length] = np.ones((length - length_cgas), dtype=int) * -1



    # end loop over subhalos
    # close previous offsets and tracers files
    offsets_past.close()
    tracers_past.close()
    
    # reshape the particles arrays
    for key in particles.keys():
        particles[key] = particles[key][:end]

    # save the offsets and particles dictionaries
    dicts  = [offsets_subhalo, particles]
    fnames = ['offsets', 'tracers']
    for d_i, d in enumerate(dicts):
        fname    = fnames[d_i]
        outfname = '%s_%03d.hdf5'%(fname, snap)
        outdirec = '../Output/%s_tracers/'%(sim)

        with h5py.File(outdirec + outfname, 'a') as outf:
            group = outf.require_group('group')
            for dset_key in d.keys():
                dset = d[dset_key]
                dataset = group.require_dataset(dset_key, shape=dset.shape, dtype=dset.dtype)
                dataset[:] = dset

            outf.close()

    
    return 



def track_subfindIDs(subfindIDs):
    """
    Given the list of subhalo subfindIDs at snapNum, use the MDB
    to find the corresponding subfindIDs at the following snapshots.
    Be careful at subhalos that don't exist in the trees or skip snaps.
    """

    # initialize result 
    max_snap = 99
    snaps    = np.arange(max_snap, snapNum-1, -1)
    n_snaps  = len(snaps)
    result   = np.ones((len(subfindIDs), n_snaps), dtype=int) * -1

    fields   = ['SubfindID', 'SnapNum']
    treeName = 'SubLink_gal'
    
    # begin loop over subfindIDs
    for i, subfindID in enumerate(subfindIDs):
        
        # load MDB
        MDB = il.sublink.loadTree(basePath, snapNum, subfindID, treeName=treeName,
                                  onlyMDB=True, fields=fields)
        
        # is subhalo in the tree?
        if not MDB:
            result[i,-1] = subfindID
            continue

        # does MDB have a bad count? (i.e., not reach z=0?)
        if (MDB['count'] + snapNum) > 100:

            # find where the MDB stops
            stop             = -(max_snap - snapNum + 1)
            MDB['SnapNum']   = MDB['SnapNum'][stop:]
            MDB['SubfindID'] = MDB['SubfindID'][stop:]
            
            start            = np.max(np.where((MDB['SnapNum'][1:] - MDB['SnapNum'][:-1]) >= 0)) + 1
            MDB['SnapNum']   = MDB['SnapNum'][start:]
            MDB['SubfindID'] = MDB['SubfindID'][start:]

        # find at which snaps the subhalo was identified
        isin = np.isin(snaps, MDB['SnapNum'])

        # and record the result
        result[i,isin] = MDB['SubfindID']

    # finish loop over subfindIDs
    # save by looping over the snapshots
    outdirec = '../Output/%s_tracers/'%(sim)
    
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


create_tracertracks()

