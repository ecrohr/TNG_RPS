### import modules
import illustris_python as il
import multiprocessing as mp
import numpy as np
import h5py
import rohr_utils as ru 
from importlib import reload
import glob

global sim, basePath, snapNum, tcoldgas, max_snap
global tracer_ptn, star_ptn, gas_ptn, bh_ptn, bary_ptns
global gas_fields
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
    global gas_fields
    global big_array_length
    global outdirec

    
    # define the global variables
    sim        = 'TNG50-4'
    basePath   = ru.ret_basePath(sim)
    snapNum    = 33
    tcoldgas   = 10.**(4.5) # [K]
    max_snap   = 99

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

    outdirec = '../Output/%s_tracers/'%(sim)

    # define the subhalos we care about at snapshot snapNum
    subfindIDs = np.arange(100)

    # already ran for snapshots 33-35; no need to rerun at the moment
    # find the corresponding subfindIDs at the next snapshots
    track_subfindIDs(subfindIDs)

    # at snapNum, initialize the tracers we care about
    initialize_coldgastracers()

    # now track tracers from snapNum + 1 until snap 99
    for snap in range(snapNum+1, max_snap+1):
        track_tracers(snap)
        
    return 



def initialize_coldgastracers():
    """
    Finds the tracers whose parents are cold gas cells gravitationally 
    bound to subhalos in subfindIDs at snapshot snapNum
    
    architecture: 
    load the subhalo subfindIDs at snapNum: 
    load the tracers at snapNum;
    for each subhalo in subfindIDs:
        load subfind cold gas cells
        cross match with tracers 
        save matched tracers 
    save all info
    return to main function and continue to next snaps
    """

    # load the subfindIDs from the offsets file
    with h5py.File(outdirec + 'offsets_%03d.hdf5'%snapNum, 'r') as f:
        subfindIDs = f['group']['SubfindID'][:]
        f.close()

    # initialize the outputs 

    # initialize the offset dictionary
    offsets_subhalo = {}
    offsets_subhalo['SubhaloOffset']            = np.zeros(len(subfindIDs), int)
    offsets_subhalo['SubhaloLength']            = np.zeros(len(subfindIDs), int)
    offsets_subhalo['SubhaloLengthColdGas']     = np.zeros(len(subfindIDs), int)
    offsets_subhalo['SubhaloLengthColdGas_new'] = np.zeros(len(subfindIDs), int)

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

        print('Working on %s snapshot %d subfindID %d'%(sim, snapNum, subfindID))

        if subfind_i == 0:
            offsets_subhalo['SubhaloOffset'][subfind_i] = 0
        else:
            offsets_subhalo['SubhaloOffset'][subfind_i] = (offsets_subhalo['SubhaloOffset'][subfind_i-1]
                                                           + offsets_subhalo['SubhaloLength'][subfind_i-1])

        # check that the subhalo is identified at snapNum
        if subfindID == -1:
            continue

        gas_cells    = il.snapshot.loadSubhalo(basePath, snapNum, subfindID, gas_ptn, fields=gas_fields)

        # check if there are any gas cells
        if gas_cells['count'] == 0:
            continue
        
        gas_cells    = ru.calc_temp_dict(gas_cells)

        # find the local indices and load the global offset for these gas cells
        cgas_indices = np.where(gas_cells['Temperature'] <= tcoldgas)[0]
        if len(cgas_indices) == 0:
            continue
        
        ParticleIDs  = gas_cells['ParticleIDs'][cgas_indices]

        # match the tracer ParentID with the cold gas cells ParticleIDs
        isin_tracer = np.isin(tracers['ParentID'], ParticleIDs)

        # save the tracerIDs and tracer indices at snapshot snapNum
        tracer_IDs = tracers['TracerID'][isin_tracer]
        tracer_indices = np.where(isin_tracer)[0]

        # fill in the offsets dictionary for this subhalo
        offsets_subhalo['SubhaloLength'][subfind_i]            = len(tracer_indices)
        offsets_subhalo['SubhaloLengthColdGas'][subfind_i]     = len(tracer_indices)
        offsets_subhalo['SubhaloLengthColdGas_new'][subfind_i] = len(tracer_indices)

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
    end =  offsets_subhalo['SubhaloOffset'][-1] +  offsets_subhalo['SubhaloLength'][-1]
    for key in particles.keys():
        particles[key] = particles[key][:end]

    # save the offsets and particles dictionaries
    dicts  = [offsets_subhalo, particles]
    fnames = ['offsets', 'tracers']
    for d_i, d in enumerate(dicts):
        fname    = fnames[d_i]
        outfname = '%s_%03d.hdf5'%(fname, snapNum)

        with h5py.File(outdirec + outfname, 'a') as outf:
            group = outf.require_group('group')
            for dset_key in d.keys():
                dset = d[dset_key]
                dataset = group.require_dataset(dset_key, shape=dset.shape, dtype=dset.dtype)
                dataset[:] = dset

            outf.close()

    
    return 


def track_tracers(snap):
    """
    Finds the tracers whose parents are cold gas cells gravitationally 
    bound to subhalos in subfindIDs at snapshot snap
    
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
    
    # load the subfindIDs from the offsets file
    with h5py.File(outdirec + 'offsets_%03d.hdf5'%snap, 'r') as f:
        subfindIDs = f['group']['SubfindID'][:]
        f.close()

    # initialize the outputs 

    # initialize the offset dictionary
    offsets_subhalo = {}
    offsets_subhalo['SubhaloOffset']            = np.zeros(len(subfindIDs), int)
    offsets_subhalo['SubhaloLength']            = np.zeros(len(subfindIDs), int)
    offsets_subhalo['SubhaloLengthColdGas']     = np.zeros(len(subfindIDs), int)
    offsets_subhalo['SubhaloLengthColdGas_new'] = np.zeros(len(subfindIDs), int)

    particles = {}
    # rewrite into a 4xbig_array_length array rather than a dictionary
    # for a speed increase
    particles['TracerIDs']      = np.empty(big_array_length, dtype=int)
    particles['TracerIndices']  = np.empty(big_array_length, dtype=int)
    particles['ParentIndices']  = np.empty(big_array_length, dtype=int)
    particles['ParentPartType'] = np.empty(big_array_length, dtype=int)

    # load the offsets and tracers from the previous snapshot
    offsets_past = h5py.File(outdirec + 'offsets_%03d.hdf5'%(snap - 1), 'r')
    tracers_past = h5py.File(outdirec + 'tracers_%03d.hdf5'%(snap - 1), 'r')

    # load every tracer particle in the simulation at this snapshot
    tracers = il.snapshot.loadSubset(basePath, snap, tracer_ptn)

    # begin loop over the subhalos at snapshot snap
    for subfind_i, subfindID in enumerate(subfindIDs):

        print('Working on %s snapshot %d subfindID %d'%(sim, snap, subfindID))

        # calculate subhalo offset
        if subfind_i == 0:
            offsets_subhalo['SubhaloOffset'][subfind_i] = 0
        else:
            offsets_subhalo['SubhaloOffset'][subfind_i] = (offsets_subhalo['SubhaloOffset'][subfind_i-1]
                                                           + offsets_subhalo['SubhaloLength'][subfind_i-1])

        # check if the subhalo is identified at this snap
        if subfindID == -1:
            continue

        ### cross match cold gas cells with tracers at this snap ###
        gas_cells    = il.snapshot.loadSubhalo(basePath, snap, subfindID, gas_ptn, fields=gas_fields)

        # check if there are any gas cells
        if gas_cells['count'] == 0:
            continue
        
        gas_cells    = ru.calc_temp_dict(gas_cells)

        # find the local indices and load the global offset for these gas cells
        cgas_indices = np.where(gas_cells['Temperature'] <= tcoldgas)[0]
        if len(cgas_indices) == 0:
            continue

        ParticleIDs  = gas_cells['ParticleIDs'][cgas_indices]

        # match the tracer ParentID with the cold gas cells ParticleIDs
        isin_tracer = np.isin(tracers['ParentID'], ParticleIDs)

        # save the tracerIDs and tracer indices at snapshot snap
        tracer_IDs = tracers['TracerID'][isin_tracer]
        tracer_indices = np.where(isin_tracer)[0]

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

        IDs = np.concatenate([tracer_IDs[isin_now],
                              tracer_IDs[~isin_now],
                              IDs_past[~isin_past]])
        
        indices = np.concatenate([tracer_indices[isin_now],
                                  tracer_indices[~isin_now],
                                  indices_past[~isin_past]])
        
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
            continue

        particles['TracerIDs'][start:start+length]     = IDs
        particles['TracerIndices'][start:start+length] = indices

        # for the cold gas cell tracers, save the parent IDs and tracers
        # get the local cold gas indices with matched tracer particles and include the global offset
        parent_IDs  = tracers['ParentID'][indices[:length_cgas]]

        isin_gas    = np.isin(ParticleIDs, parent_IDs)

        r           = il.snapshot.getSnapOffsets(basePath, snap, subfindID, "Subhalo")
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
    end = offsets_subhalo['SubhaloOffset'][-1] + offsets_subhalo['SubhaloLength'][-1]
    for key in particles.keys():
        particles[key] = particles[key][:end]

    # save the offsets and particles dictionaries
    dicts  = [offsets_subhalo, particles]
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
            tree = il.sublink.loadTree(basePath, max_snap, subfindID, treeName=treeName,
                                       onlyMPB=True, fields=fields)

            # is subhalo in the tree?
            if not tree:
                result[i,0] = subfindID
                continue

            # trim tree to stop at snap snapNum
            indices  = tree['SnapNum'] >= snapNum

            for field in fields:
                tree[field] = tree[field][indices]
               
        else:
            tree = il.sublink.loadTree(basePath, snapNum, subfindID, treeName=treeName,
                                       onlyMDB=True, fields=fields)
        
            # is subhalo in the tree?
            if not tree:
                result[i,-1] = subfindID
                continue

            # does MDB have a bad count? (i.e., not reach z=0?)
            if (tree['count'] + snapNum) > max_snap:

                # find where the MDB stops
                stop              = -(max_snap - snapNum + 1)
                tree['SnapNum']   = tree['SnapNum'][stop:]
                tree['SubfindID'] = tree['SubfindID'][stop:]

                start             = np.max(np.where((tree['SnapNum'][1:] - tree['SnapNum'][:-1]) >= 0)) + 1
                tree['SnapNum']   = tree['SnapNum'][start:]
                tree['SubfindID'] = tree['SubfindID'][start:]


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


create_tracertracks()

