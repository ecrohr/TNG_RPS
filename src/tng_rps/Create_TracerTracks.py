### import modules
import illustris_python as il
import multiprocessing as mp
import numpy as np
import h5py
import rohr_utils as ru 
import glob
import time
import os
import argparse
from functools import partial

gas_fields  = ['InternalEnergy', 'ElectronAbundance', 'StarFormationRate', 'ParticleIDs']

part_fields = ['ParticleIDs']

big_array_length = int(1e9)

def create_tracertracks(first_snap, last_snap, Config):
    """
    Run the Create_TracerTracks.py file. 
    Starts at min_snap with a list of subhalo subfindIDs,
    find the descendants, and loops over the snapshots
    matching the tracers from bound cold gas cells.
    Saves the output.
    """
    
    sim = Config.sim
    min_snap = Config.min_snap
    
    for snap in range(first_snap, last_snap+1):
        # for the first snapshot, define some subfindIDs of interest,
        # and determine the subfindID at every snapshot
        ### rewrite into an intialization function
        if snap == min_snap:
            track_subfindIDs(Config)
        start = time.time()
        track_tracers(snap, Config)
        end = time.time()
        print('%s inspected branches track_tracers at snap %03d: %.3g [s]'%(sim, snap, (end-start)))

    # and find the unmatched tracers from min_snap + 1 until max_snap
    for snap in range(first_snap, last_snap+1):
        start = time.time()
        find_unmatched_tracers(snap, Config)
        end = time.time()
        print('%s snap %03d find_unmatched_tracers: %.3g [s]'%(sim, snap, (end-start)))

        if snap == (last_snap - 1):
            # add bound flag for the tracer parents
            snaps = range(min_snap+1, max_snap+1)
            Pool = mp.Pool(8)
            Pool.map(partial(create_bound_flags, Config=Config), snaps)
            Pool.close()
            Pool.join()


    ### SLURM job parser, which needs to be moved elsewhere
    # use the jobid to set the snaps we track
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobid", help="slurm job id")
    args = parser.parse_args()
    jobid = int(args.jobid)
    
    # let's assume we can go through 5 snapshots per job (i.e. 24 hours)
    Nsnapsperjob = 10
    first_snap = min_snap + Nsnapsperjob * jobid
    last_snap = first_snap + Nsnapsperjob
    if last_snap >= max_snap+1:
        last_snap = max_snap+1
        
    for snap in range(first_snap, last_snap):
        if snap == min_snap:
            # define the subhalos we care about -- now the TNG50-1 inspected, cleaned branches
            indirec = '../Output/zooniverse/'
            infname = 'zooniverse_TNG50-1_inspected_clean_tau.hdf5'
            with h5py.File(indirec + infname, 'r') as f:
                Group = f['Group']
                subfindIDs = Group['SubfindID'][:]
                f.close()

            track_subfindIDs(subfindIDs)

        a = time.time()
        track_tracers(snap)
        b = time.time()
        print('TNG50-1 inspected branches track_tracers at snap %03d: %.3g [s]'%(snap, (b-a)))    

    # and find the unmatched tracers from min_snap + 1 until max_snap
    for snap in range(first_snap, last_snap):
        start = time.time()
        find_unmatched_tracers(snap)
        end = time.time()
        print('%s snap %03d find_unmatched_tracers: %.3g [s]'%(sim, snap, (end-start)))

        if snap == (last_snap - 1):
            # add bound flag for the tracer parents
            snaps = range(min_snap+1, max_snap+1)
            Pool = mp.Pool(8)
            Pool.map(create_bound_flags, snaps)
            Pool.close()
            Pool.join()
    """

    return


def track_tracers(snap, Config):
    """
    Match the bound cold gas cells of the subfindIDs at snap to the tracers.
    If snap < min_snap, then loads the previous tracers and checks if they still
    belong to the same subhalo. 
    Saves offsets and tracers catalogs, but does not find the unmatched tracers -- 
    i.e., does not search for tracers whose parents were bound cold gas cells but
    no longer are (group 3 tracers in the code below).
    No returns.
    """

    sim = Config.sim
    basePath = Config.basePath
    min_snap = Config.min_snap
    tracer_outdirec = Config.tracer_outdirec
    tracer_ptn = Config.tracer_ptn
    gas_ptn = Config.gas_ptn

    print('Working on %s snap %03d'%(sim, snap))
       
    # load the subfindIDs
    with h5py.File(tracer_outdirec + 'offsets_%03d.hdf5'%snap, 'r') as f:
        subfindIDs = f['group']['SubfindID'][:]
        f.close()
      
    # initialize the outputs
    offsets_subhalo, tracers_subhalo = initialize_outputs(subfindIDs.size)
    
    # if snap > min_snap: load previous tracers
    if (snap > min_snap):
        offsets_past = []
        tracers_past = []
        for i in range(1, 4):
            if (snap - i) < min_snap:
                break
            offsets, tracers = load_catalogs(snap - i, Config)
            offsets_past.append(offsets)
            tracers_past.append(tracers)

    else:
        offsets_past = None
        tracers_past = None
        
    # find the gas cells of interest
    ParticleIDs, Particle_indices, Temperatures, offsets, lengths = find_coldgascells(subfindIDs, snap, Config)
    
    # if there are no particles of interest, save
    if ParticleIDs.size == 0:
        # by definition, all subhalos have 0 offset
        for key in offsets_subhalo.keys():
            offsets_subhalo[key][:] = 0
        # reshape the tracers_subhalo arrays
        end = offsets_subhalo['SubhaloOffset'][-1] + offsets_subhalo['SubhaloLength'][-1]
        for key in tracers_subhalo.keys():
            tracers_subhalo[key] = tracers_subhalo[key][:end]
        save_catalogs(offsets_subhalo, tracers_subhalo, snap, Config)
        return

    # load all tracers in the simulation
    tracers = il.snapshot.loadSubset(basePath, snap, tracer_ptn)
    
    # match the ParticleIDs to the tracer ParentIDs
    indices1, indices2 = ru.match3(ParticleIDs, tracers['ParentID'])
    
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

        sub_indices = ((indices1 >= start) & (indices1 < stop))
        Ntracers = sub_indices[sub_indices].size
        offsets_subhalo['SubhaloLength'][subfind_i]            = Ntracers
        offsets_subhalo['SubhaloLengthColdGas'][subfind_i]     = Ntracers
        offsets_subhalo['SubhaloLengthColdGas_new'][subfind_i] = Ntracers
        
        tracer_IDs_sub = tracer_IDs[sub_indices]
        tracer_indices_sub = tracer_indices[sub_indices]

        # if snap > SnapNum: cross match with previous snapshot(s)
        if snap > min_snap:
            ### check which cold gas cell tracers from previous snap are still here ###
            i = 1
            IDs_past = np.array([])
            while ((snap - i) >= min_snap) & (i < 4):
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
                    IDs_past = indices_past = np.zeros(0, dtype=int) - 1
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
            indices_group3 = np.zeros(IDs_group3.size, dtype=int) - 1
            
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
        tracers_subhalo['ParentPartType'][start:start+length_cgas] = np.zeros(part_indices.size, dtype=int) + gas_ptn
        tracers_subhalo['ParentGasTemp'][start:start+length_cgas] = temperatures[sub_indices]

        # for now, make the parent indices and parent part type -1
        # then later load the other baryonic particles in the sim and match
        # their particle IDs with all unmatched tracers
        tracers_subhalo['ParentIndices'][start+length_cgas:start+length]  = np.zeros((length - length_cgas), dtype=int) - 1
        tracers_subhalo['ParentPartType'][start+length_cgas:start+length] = np.zeros((length - length_cgas), dtype=int) - 1
        tracers_subhalo['ParentGasTemp'][start+length_cgas:start+length]  = np.zeros((length - length_cgas), dtype=float) - 1
    
    # finish loop over subfindIDs    

    # reshape the tracers_subhalo arrays
    end = offsets_subhalo['SubhaloOffset'][-1] + offsets_subhalo['SubhaloLength'][-1]
    for key in tracers_subhalo.keys():
        tracers_subhalo[key] = tracers_subhalo[key][:end]
        
    save_catalogs(offsets_subhalo, tracers_subhalo, snap, Config)
            
    return

def find_unmatched_tracers(snap, Config):
    """
    For all unmatched tracers at snap, find their parents.
    """

    basePath = Config.basePath
    tracer_ptn = Config.tracer_ptn
    bary_ptns = Config.bary_ptns
    gas_ptn = Config.gas_ptn
    
    # load the offsets, tracers at snap
    a = time.time()
    offsets_subhalo, tracers_subhalo = load_catalogs(snap, Config)
    b = time.time()

    # find the unmatched tracers
    unmatched_indices = tracers_subhalo['ParentPartType'] == -1
    unmatched_TracerIDs = tracers_subhalo['TracerIDs'][unmatched_indices]
    c = time.time()
    
    if unmatched_TracerIDs.size == 0:
        print('Warning, no unmatched tracers. Returning.')
        return

    # load all tracers at snap
    tracers = il.snapshot.loadSubset(basePath, snap, tracer_ptn)

    # match unmatched tracerIDs to all tracers at snap to save indices and ParentIDs
    # NB: unmatched_tracerIDs is not necessarily unique
    tracers_indices, matched_unmatched_indices = ru.match3(tracers['TracerID'], unmatched_TracerIDs)
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
        if Particles['count'] == 0:
            continue
        
        ParticleIDs = Particles['ParticleIDs']

        # match ParentIDs with unmatched_ParentIDs
        # reminder that unmatched_ParentIDs are not unique
        Particle_indices, Parent_indices = ru.match3(ParticleIDs, unmatched_ParentIDs)

        if Particle_indices is None:
            print('Warning, no matched parents for part type %d. Continuing to the next bary_ptn'%ptn)
            continue

        parent_indices = Particle_indices
        parent_ptn = np.ones(parent_indices.size, dtype=int) * ptn

        # calculate the temperature for the gas cells
        if ptn == gas_ptn:
            for key in fields:
                Particles[key] = Particles[key][parent_indices]
            Particles['count'] = parent_indices.size
            Particles = ru.calc_temp_dict(Particles)
            temps = Particles['Temperature']
        else:
            temps = np.zeros(parent_indices.size, dtype=float) - 1.

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
    save_catalogs(offsets_subhalo, tracers_subhalo, snap, Config)
        
    return


def find_coldgascells(subfindIDs, snap, Config):
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
    
    sim = Config.sim
    basePath = Config.basePath
    gas_ptn = Config.gas_ptn
    tlim = Config.tlim
    
    a = time.time()
    gas_cells = il.snapshot.loadSubset(basePath, snap, gas_ptn, fields=gas_fields)
    b = time.time()

    gas_cells = ru.calc_temp_dict(gas_cells)

    c = time.time()

    dtype = int
    offsets = np.zeros(subfindIDs.size, dtype=dtype)
    lengths = np.zeros(subfindIDs.size, dtype=dtype)

    dtype = gas_cells['ParticleIDs'].dtype
    ParticleIDs = np.zeros(big_array_length, dtype=dtype) - 1
    Particle_indices = np.zeros(big_array_length, dtype=int) - 1

    for i, subfindID in enumerate(subfindIDs):

        if i > 0:
            offsets[i] = offsets[i-1] + lengths[i-1]
            
        if subfindID == -1:
            continue
            
        r = il.snapshot.getSnapOffsets(basePath, snap, subfindID, 'Subhalo')
        start = r['offsetType'][gas_ptn]
        end = r['lenType'][gas_ptn]

        temps = gas_cells['Temperature'][start:start+end]
        indices = temps <= tlim

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
    
    if ParticleIDs.size == 0:
        print('Warning, no cold gas cells.')
    elif np.min(ParticleIDs) <= 0:
        print('Warning, ParticleIDs has negative values.')
    
    return ParticleIDs, Particle_indices, Temperatures, offsets, lengths


def create_bound_flags(snap, Config):
    """
    Post-process the catalogs to add a still-bound flag to each tracer.
    'Still-Bound' means that the tracer's parent is still gravitationally 
    bound via Subfind to the relevant subhalo. 
    True == 1 means yes; False == 0 means no; -1 means not checked.
    Every tracer particle whose parent has been found should be checked.
    NB: tracer parents with a False flag may still be bound, but they are bound
    to a subhalo different from where it came. For example, many stripped gas cells
    from satellite galaxies will no longer be bound to the satellite (flag == False),
    but the cells will probably be bound to the central galaxies.
    Saves the StillBound_flag to the tracers catalog. No returns.
    """
    
    basePath = Config.basePath
    bary_ptns = Config.bary_ptns
    
    # load the catalogs at snap
    offsets_subhalo, tracers_subhalo = load_catalogs(snap, Config)

    # initalize the ouput
    still_bound = np.zeros(tracers_subhalo['ParentPartType'].size, dtype=int) - 1

    # begin loop over subfindIDs
    for subfind_i, subfindID in enumerate(offsets_subhalo['SubfindID']):

        if subfindID == -1:
            continue

        # first, all group 1 and 2 tracers are still bound by definition
        start_groups12 = offsets_subhalo['SubhaloOffset'][subfind_i]
        end_groups12 = start_groups12 + offsets_subhalo['SubhaloLengthColdGas'][subfind_i]
        still_bound[start_groups12:end_groups12] = 1

        # now let's check the group the group 3 tracers
        start_group3 = start_groups12 + offsets_subhalo['SubhaloLengthColdGas'][subfind_i]
        end_group3 = start_groups12 + offsets_subhalo['SubhaloLength'][subfind_i]

        # load the relevant parent indices and parent part type
        parent_indices = tracers_subhalo['ParentIndices'][start_group3:end_group3]
        parent_parttype = tracers_subhalo['ParentPartType'][start_group3:end_group3]

        # load the particle indices for the subahlo
        r = il.snapshot.getSnapOffsets(basePath, snap, subfindID, 'Subhalo')

        # loop over the baryon part types 
        for i, ptn in enumerate(bary_ptns):
            # find the starting and ending indices for the subhalo for the given part type
            subhalo_start = r['offsetType'][ptn]
            subhalo_end = subhalo_start + r['lenType'][ptn]

            # find the relevant parent indices for the given part type
            parent_indices_indices = parent_parttype == ptn

            # find the bound indices
            bound_indices = ((parent_indices[parent_indices_indices] >= subhalo_start) &
                             (parent_indices[parent_indices_indices] < subhalo_end))

            # save the bound and unbound flags
            save_indices = np.arange(start_group3, end_group3)[np.where(parent_indices_indices)[0]]
            save_bound_indices = save_indices[bound_indices]
            save_unbound_indices = save_indices[~bound_indices]
            still_bound[save_bound_indices] = 1
            still_bound[save_unbound_indices] = 0
        # finish loop over the baryon part types
    # finish loop over subfindIDs

    # check that all particles have a bound / unbound flag
    if still_bound[still_bound == -1].size > 0:
        print('Warning, not all particles were checked!')
        orphan_indices = tracers_subhalo['ParentPartType'] == -1
        print(still_bound[still_bound == -1].size, orphan_indices[orphan_indices].size)

    # save the catalogs
    tracers_subhalo['StillBound_flag'] = still_bound
    save_catalogs(offsets_subhalo, tracers_subhalo, snap, Config)
    
    return


def track_subfindIDs(Config, z0_flag=True):
    """
    Given the list of subhalo subfindIDs at either z0 (default) or 
    at min_snap, use either the MPB (default) or MDB to find the
    corresponding subfindIDs at the other snapshots between min_snap and 99.
    Be careful at subhalos that don't exist in the trees or skip snaps.
    """

    # initialize result
    subfindIDs = Config.SubfindIDs
    snaps    = Config.SnapNums
    n_snaps  = snaps.size
    result   = np.zeros((subfindIDs.size, n_snaps), dtype=int) - 1
    sim = Config.sim
    treeName = Config.treeName
    min_snap = Config.min_snap
    max_snap = Config.max_snap

    fields   = ['SubfindID', 'SnapNum']
    treeName = 'SubLink_gal'
    
    # begin loop over subfindIDs
    for i, subfindID in enumerate(subfindIDs):
        
        if (z0_flag):
            tree = ru.loadMainTreeBranch(sim, max_snap, subfindID, treeName=treeName,
                                         fields=fields, min_snap=min_snap, max_snap=max_snap)
        else:
            tree = ru.loadMainTreeBranch(sim, min_snap, subfindID, treeName=treeName,
                                         fields=fields, min_snap=min_snap, max_snap=max_snap)
            
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
        with h5py.File(Config.tracer_outdirec + outfname, 'a') as outf:
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
    offsets_subhalo['SubhaloOffset']            = np.zeros(Nsubhalos, dtype=int) - 1
    offsets_subhalo['SubhaloLength']            = offsets_subhalo['SubhaloOffset'].copy()
    offsets_subhalo['SubhaloLengthColdGas']     = offsets_subhalo['SubhaloOffset'].copy()
    offsets_subhalo['SubhaloLengthColdGas_new'] = offsets_subhalo['SubhaloOffset'].copy()

    tracers_subhalo = {}
    # rewrite into a 4xbig_array_length array rather than a dictionary for a speed increase
    tracers_subhalo['TracerIDs']      = np.zeros(big_array_length, dtype=int) - 1
    tracers_subhalo['TracerIndices']  = tracers_subhalo['TracerIDs'].copy()
    tracers_subhalo['ParentIndices']  = tracers_subhalo['TracerIDs'].copy()
    tracers_subhalo['ParentPartType'] = tracers_subhalo['TracerIDs'].copy()
    tracers_subhalo['ParentGasTemp']  = np.zeros(big_array_length, dtype=float) - 1
    
    return offsets_subhalo, tracers_subhalo


def load_catalogs(snap, Config):
    """
    Load the offsets and tracers catalogs at snap.
    Returns dicitonaries of the offsets, tracers catalogs.
    """
    result = {}
    fnames = ['offsets', 'tracers']
    for i, fname in enumerate(fnames):
        result[fname] = {}
        with h5py.File(Config.tracer_outdirec + '%s_%03d.hdf5'%(fname, snap), 'r') as f:
            group = f['group']
            for key in group.keys():
                result[fname][key] = group[key][:]
        f.close()
        
    return result[fnames[0]], result[fnames[1]]    


def save_catalogs(offsets, tracers, snap, Config):
    """
    Save the offsets and tracers catalogs at snap.
    """
    dicts  = [offsets, tracers]
    fnames = ['offsets', 'tracers']
    for d_i, d in enumerate(dicts):
        fname    = fnames[d_i]
        tracer_outfname = '%s_%03d.hdf5'%(fname, snap)

        with h5py.File(Config.tracer_outdirec + tracer_outfname, 'a') as outf:
            group = outf.require_group('group')
            for dset_key in d.keys():
                dset = d[dset_key]
                dataset = group.require_dataset(dset_key, shape=dset.shape, dtype=dset.dtype)
                dataset[:] = dset

            outf.close()
            
    return

