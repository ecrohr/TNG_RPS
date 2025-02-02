#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 09:29:53 2021

@author: rohr
"""

### import modules
import illustris_python as il
import numpy as np
import h5py
import rohr_utils as ru 
import os
import multiprocessing as mp

def run_clean_zooniverseGRP(Config):
    """ Clean the Zooniverse sample based on various selection criteria. """
    
    outdirec = Config.outdirec
    GRPfname = Config.GRPfname
    
    if Config.run_cleanSGRP:
        dic        = load_dict(GRPfname, Config)
        keys_dic   = clean_subfindGRP(dic, Config)
        
        # for each set of keys, save the resulting GRP dictionaries
        for out_key in Config.taudict_keys:
            keys = keys_dic[out_key]

            result = {}
            for key in keys:
                group = dic[key]

                new_key = '%08d'%(group['SubfindID'][0])
                result[new_key] = group

            fname = Config.return_fnames(subsample=out_key)[0]
            # if the file already exists, delete before making a new one.
            if os.path.exists(outdirec + fname):
                print('File %s already exists. Deleting before saving new version.'%(outdirec + fname))
                os.system('rm %s '%(outdirec + fname))
            with h5py.File(outdirec + fname, 'a') as outf:
                for group_key in result.keys():
                    group = outf.require_group(group_key)
                    for dset_key in result[group_key].keys():
                        dset = result[group_key][dset_key]
                        dataset = group.require_dataset(dset_key, shape=dset.shape, dtype=dset.dtype)
                        dataset[:] = dset
                        
                outf.close()
                                
        if Config.zooniverse_flag:
            # now split the inspected branches into jellyfish, if there's a jellyfish classificaiton
            # at snap >= snap_first, and into nonjellyf, if there are no jelly classiifications at snap >= snap_first
            # this means that some of the branches with a jellyfish classification may become nonjellyf branches!
            split_inspected_branches(Config)
            # reorganize each of the three sets of branches [inspected, jellyfish, nonjellyf] into tau dictionaries
 
    
    if Config.run_createtau:
        if not Config.zooniverse_flag:
            out_keys = Config.taudict_keys
        else:
            out_keys = Config.zooniverse_keys
        # and run for each of the out_keys
        for out_key in out_keys:
            if Config.min_snap == Config.max_snap & Config.min_snap == 99:
                if out_key != Config.all_key:
                    continue
            print(out_key)
            create_taudict(Config, out_key=out_key)
            split_tau_gasz0(Config, out_key=out_key)

    return


def clean_subfindGRP(dic, Config):
    """
    Wrapper function for cleaning branches based on either
    satellites or centrals of interest.
    """
    
    if Config.centrals_flag:
        return clean_subfindGRP_centrals(dic, Config)
    else:
        return clean_subfindGRP_satellites(dic, Config)


def clean_subfindGRP_satellites(dic, Config):
    """
    For each branch, load the various flags and sort them.
    Based on the flags defined in Config, decided on which criteria
    to clean the satellite subfindGRP branches.
    """

    keys = np.array(list(dic.keys()))
    
    print('We are going through %s branches of interest.'%keys.size)

    # let's clean the branches, noting the keys of the objects excised at each step

    subfind_flags = load_subfindsnapshot_flags(Config)
    central_z0_flag = Config.central_z0_flag
    backsplash_prev_flag = Config.backsplash_prev_flag
    preprocessed_flag = Config.preprocessed_flag
    
    nonz0_key = Config.nonz0_key
    beforesnapfirst_key = Config.beforesnapfirst_key
    backsplash_prev_key = backsplash_prev_flag
    preprocessed_key = preprocessed_flag
    clean_key = Config.clean_key

    # note that all means all branches for analysis, not necessarily all branches originally
    # in general, all branches includes preprocessed, clean, and backsplash_prev branches
    all_key = Config.all_key

    # initalize empty lists to hold the various keys
    clean_keys           = []
    nonz0_keys           = []
    backsplash_prev_keys = []
    preprocessed_keys    = []
    centralz0_keys = []
    all_keys = []
    if Config.zooniverse_flag:
        beforesnapfirst_keys = []
        snap_first = Config.zooniverse_snapfirst
        
    # begin loop over the subhalos in the GRP dictionary
    for key in keys:
        group = dic[key]

        # ignore the snaps where the subahlo was not identified
        indices = group['SubfindID'] != -1

        SnapNum            = group['SnapNum'][indices]
        SubfindID          = group['SubfindID'][indices]
        if Config.zooniverse_flag:
            ins_flags = group['ins_flags'][indices]

        # the MDB must reach z=0 (snap 99)
        if SnapNum.max() < 99:
            nonz0_keys.append(key)
            continue

        # there must be at least one inspection at snap >= snap_first for Zooniverse objects
        if Config.zooniverse_flag:
            ins_flag = max(ins_flags[SnapNum >= snap_first])
            if (ins_flag != 1):
                beforesnapfirst_keys.append(key)
                continue
            
        # the subhalo exists at z=0, so use subfind flags to append to the appropriate list
        SubfindID_z0  = SubfindID[0]
        
        # confirm that the subhalo is a z=0 satellite
        if (subfind_flags[central_z0_flag][SubfindID_z0] == 1):
            centralz0_keys.append(key)
            continue

        # the subhalo meets criteria to be considered in further analysis
        all_keys.append(key)

        # for the zooniverse project, conisder backsplash_prev clean and separate from pre-processed:
        if Config.zooniverse_flag:
            if subfind_flags[preprocessed_flag][SubfindID_z0]:
                preprocessed_keys.append(key)
            else:
                clean_keys.append(key)

                # clean keys could also include backsplash_prev
                if subfind_flags[backsplash_prev_flag][SubfindID_z0]:
                    backsplash_prev_keys.append(key)

        else:
            # in general, consider the following groups all separately
            # consider backsplash galaxies together, regardless if they 
            # are backsplashes of their z=0 or a pre-processing host
            if subfind_flags[backsplash_prev_flag][SubfindID_z0]:
                backsplash_prev_keys.append(key)
                continue

            if subfind_flags[preprocessed_flag][SubfindID_z0]:
                preprocessed_keys.append(key)
                continue

            # not a backsplash or a pre-processed galaxy, so consider clean
            clean_keys.append(key)
    
    # end loop over the branches

    print('satellite branches not reaching z=0: %d'%(len(nonz0_keys)))
    if Config.zooniverse_flag:
        print('not inspected since %d: %d'%(snap_first, len(beforesnapfirst_keys)))
    print('central at z=0: %d'%len(centralz0_keys))
    print('backsplash_prev: %d; preprocessed: %d'%(len(backsplash_prev_keys), len(preprocessed_keys)))
    print('clean: %d'%(len(clean_keys)))
    print('all keys (clean + preprocessed + backsplash_prev): %d'%(len(all_keys)))
    # save the keys as a dictionary and return to main function
    result  = {}
    
    if Config.zooniverse_flag:
        result_keys  = [nonz0_key,
                        beforesnapfirst_key,
                        backsplash_prev_key,
                        preprocessed_key,
                        clean_key,
                        all_key]
        
        result_dsets = [nonz0_keys,
                        beforesnapfirst_keys,
                        backsplash_prev_keys,
                        preprocessed_keys,
                        clean_keys,
                        all_keys]
    else:
        result_keys  = [nonz0_key,
                        backsplash_prev_key,
                        preprocessed_key,
                        clean_key,
                        all_key]
        
        result_dsets = [nonz0_keys,
                        backsplash_prev_keys,
                        preprocessed_keys,
                        clean_keys,
                        all_keys]

    for result_i, result_key in enumerate(result_keys):
        result[result_key] = result_dsets[result_i]

    return result


def clean_subfindGRP_centrals(dic, Config):
    """
    For each branch, load the various flags and sort them.
    Based on the flags defined in Config, decided on which criteria
    to clean the central subfindGRP branches.
    """

    keys = np.array(list(dic.keys()))
    
    print('We are going through %s branches of interest.'%keys.size)

    # let's clean the branches, noting the keys of the objects excised at each step

    subfind_flags = load_subfindsnapshot_flags(Config)
    central_z0_flag = Config.central_z0_flag
    backsplash_z0_flag = Config.backsplash_z0_flag
    
    all_key = Config.all_key
    clean_key = Config.clean_key

    # initalize empty lists to hold the various keys
    all_keys = []
    clean_keys = []
    backsplash_z0_keys = []
        
    # begin loop over the subhalos in the GRP dictionary
    for key in keys:
        group = dic[key]

        # ignore the snaps where the subahlo was not identified
        indices = group['SubfindID'] != -1

        SnapNum            = group['SnapNum'][indices]
        SubfindID          = group['SubfindID'][indices]

        # the MDB must reach z=0 (snap 99)
        if SnapNum.max() < 99:
            raise ValueError('branch %s does not reach z=0'%key)
            
        # the subhalo exists at z=0, so use subfind flags to append to the appropriate list
        SubfindID_z0  = SubfindID[0]
        
        # confirm that the subhalo is a z=0 central
        if (subfind_flags[central_z0_flag][SubfindID_z0] == 0):
            raise ValueError('branch %s has %s == 0'%(key, central_z0_flag))
            
        all_keys.append(key)

        # backsplash at z=0? if not then considered clean
        if subfind_flags[backsplash_z0_flag][SubfindID_z0]:
            backsplash_z0_keys.append(key)
            continue

        clean_keys.append(key)

    # end loop over the branches

    print('all (backsplash + non backsplash) %d'%(len(all_keys)))
    print('backsplash at z=0: %d'%len(backsplash_z0_keys))
    print('clean (i.e., not backsplash at z=0): %d'%(len(clean_keys)))

    # save the keys as a dictionary and return to main function
    result  = {}
    
    result_keys  = [all_key,
                    clean_key,
                    backsplash_z0_flag]
    
    result_dsets = [all_keys,
                    clean_keys,
                    backsplash_z0_keys]

    for result_i, result_key in enumerate(result_keys):
        result[result_key] = result_dsets[result_i]

    return result


def load_subfindsnapshot_flags(Config):
    """ Helpfer function to lead the subfind flags. """
    
    from Create_SubfindSnapshot_Flags import return_outdirec_outfname
    direc, fname = return_outdirec_outfname(Config, snapshotflags=False)
    
    result = {}
    
    with h5py.File(direc + fname, 'r') as f:
        group = f['group']
        for dset_key in group.keys():
            result[dset_key] = group[dset_key][:]
                
        f.close()
        
    return result

# split the inspected branches into jellyfish and nonjellyf
def split_inspected_branches(Config):
    """
    split the zooniverse branches into all (inspected), jellyfish,
    and non-jellyfish branches based on their classification at z <= 0.5
    """
    outdirec = Config.outdirec
    sim = Config.sim
    zooniverse_keys = Config.zooniverse_keys
    zooniverse_snapfirst = Config.zooniverse_snapfirst
    ins_key = Config.ins_key
    
    fnames = []
    for zooniverse_key in zooniverse_keys:
        fname = outdirec + 'zooniverse_%s_%s_clean.hdf5'%(sim, zooniverse_key)
        fnames.append(fname)

        # manually rename the zooniverse clean file to the old naming convention
        if zooniverse_key == ins_key:
            if os.path.exists(fname):
                print('File %s exists (but shouldn"t). Manually deleting and recreating from %s.'%(fname, return_outfname(Config, out_key='clean')))
                os.system('rm %s'%fname)
                os.system('ln -s %s %s'%(return_outfname(Config, out_key='clean'), fname))
            else:
                print('File %s doesn"t exist. Manually recreating from %s.'%(fname, return_outfname(Config, out_key='clean')))
                os.system('ln -s %s %s'%(return_outfname(Config, out_key='clean'), fname))
        else:
            # if the jellyfish and nonjellyf files already exist, then delete them
            if os.path.exists(fname):
                os.system('rm %s'%fname)
    
    insf = h5py.File(fnames[0], 'a')
    jelf = h5py.File(fnames[1], 'a')
    nonf = h5py.File(fnames[2], 'a')

    jelbeforesnapfirst_keys = []
    
    for group_key in insf.keys():
        group = insf[group_key]
        SnapNum = group['SnapNum'][:]
        SnapNum_indices = SnapNum >= zooniverse_snapfirst
        jel_flag = np.max(group['jel_flags'][SnapNum_indices])
        if (jel_flag == 1):
        #if (jel_flag):
            insf.copy(insf[group_key], jelf)
        else:
            insf.copy(insf[group_key], nonf)
            # check if the galaxy was a jellyfish before snap_first
            if (np.max(group['jel_flags'][~SnapNum_indices]) == 1):
                jelbeforesnapfirst_keys.append(group_key) 

    print('Of the %d inspected clean branches in %s'%(len(insf.keys()), sim))
    print('%d are jellyfish and %d are nonjellyf.'%(len(jelf.keys()), len(nonf.keys())))
    print('%d were jellyfish before snap %d but are included in nonjellyf.'%(len(jelbeforesnapfirst_keys),
                                                                             zooniverse_snapfirst))

    insf.close()
    jelf.close()
    nonf.close()

    return


def create_taudict_onlyz0(Config, out_key=None):
    """
    reorganize the standard GRP file into a large 2D array
    """

    GRPfname = Config.return_fnames(out_key)[0]
    result = load_dict(GRPfname, Config)
    result_keys = list(result.keys())
    result_keys.sort()

    # if there are no subhalos of interest in this file, don't create a tau file
    if len(result_keys) == 0:
        print('There are no subhalos of interest in this file %s.'%GRPfname)
        return

    # pick the datasets we want to output at given times for all subhalos
    if Config.centrals_flag:
        grp_keys = ['SnapNum', 'CosmicTime', 'HostGroup_M_Crit200',
                    'HostGroup_R_Crit200', 'HostSubhalo_Mstar_Rgal', 'SubhaloMass',
                    'Subhalo_Mstar_Rgal',
                    'SubhaloColdGasMass', 'SubhaloHotGasMass', 'SubhaloGasMass']
    else:
        grp_keys = ['SnapNum', 'CosmicTime', 'HostCentricDistance_norm', 'HostGroup_M_Crit200',
                    'HostGroup_R_Crit200', 'HostSubhalo_Mstar_Rgal', 'SubhaloMass',
                    'Subhalo_Mstar_Rgal',
                    'SubhaloColdGasMass', 'SubhaloHotGasMass', 'SubhaloGasMass',
                    'Nperipass', 'min_Dperi_norm', 'min_Dperi_phys', 'Napopass',
                    'min_HostCentricDistance_norm', 'min_HostCentricDistance_phys',
                    'SubhaloStellarPhotometricsU',
                    'SubhaloStellarPhotometricsB',
                    'SubhaloStellarPhotometricsV',
                    'SubhaloStellarPhotometricsK',
                    'SubhaloStellarPhotometricsg',
                    'SubhaloStellarPhotometricsr',
                    'SubhaloStellarPhotometricsi',
                    'SubhaloStellarPhotometricsz']
    if Config.TNGCluster_flag:
        grp_keys.append('SubhaloDMLowResMass')    

    tauresult = {}
    # begin loop over subhalos
    for group_index, group_key in enumerate(result_keys):
        group = result[group_key]
        # if just starting, then initialize the dictionary 
        if group_index == 0:
            tauresult['SubfindID'] = np.zeros(len(result_keys), dtype=int)
            tauresult['HostSubhaloGrNr'] = np.zeros(len(result_keys), dtype=int)
            for grp_key in grp_keys:
                tauresult_key = grp_key + '_z0'
                tauresult[tauresult_key] = np.zeros(len(result_keys),
                                                    dtype=group[grp_key].dtype) - 1
                    
        tauresult['SubfindID'][group_index] = group['SubfindID'][0]
        tauresult['HostSubhaloGrNr'][group_index] = group['HostSubhaloGrNr'][0]
        # finish initializing the the result

        # assign the values at z=0, which are always the 0th element in the array
        for grp_key in grp_keys:
            tauresult_key = grp_key + '_z0'
            tauresult[tauresult_key][group_index] = group[grp_key][0]
    # finish loop over the branches
      
    # save the tau dictionary
    outdirec = Config.outdirec
    fname = Config.return_fnames(out_key)[1]
    if os.path.exists(outdirec + fname):
        print('File %s already exists. Deleting and recreating.'%(outdirec +fname))
        os.system('rm %s'%(outdirec + fname))
    with h5py.File(outdirec + fname, 'w') as outf:
        group = outf.require_group('Group')        
        for dset_key in tauresult.keys():  
            dset = tauresult[dset_key]
            dataset = group.require_dataset(dset_key, shape=dset.shape, dtype=dset.dtype)
            dataset[:] = dset

        outf.close()        
    
    return 


def create_taudict(Config, out_key=None):
    """
    reorganize the evolutionary tracks into 1D arrays with scalars at specific times.
    this only applies to subhalos that reach z=0, as z=0 is one of the specific times of interest.
    other times could include infall into z=0 host, the time when a galaxy has lost 100% of its gas, etc.
    note that there are additional datasets if the tracers and/or the quenching analysis has already been run.
    """
    
    def return_tauresult_key(grp_key, tau_key, tau_val):
        """ helper function to determine the key name """
        return (grp_key + '_' + tau_key + '%d'%tau_val)
    
    if Config.min_snap == Config.max_snap and Config.min_snap == 99:
        create_taudict_onlyz0(Config, out_key)
        return
    
    zooniverse_flag = Config.zooniverse_flag
    tracers_flag = Config.tracers_flag
    quench_flag = not Config.TNGCluster_flag # quenching catalogs not available for TNG-Cluster

    GRPfname = Config.return_fnames(out_key)[0]
        
    result = load_dict(GRPfname, Config)
    result_keys = list(result.keys())
    result_keys.sort()
    
    # if there are no subhalos of interest in this file, don't create a tau file
    if len(result_keys) == 0:
        print('There are no subhalos of interest in this file %s.'%GRPfname)
        return

    tauresult = {}
    tauvals_dict = {}
    tau_keys = []
    tautypes = ['tau_medpeak', 'tau_infall']
    gastypes = ['ColdGas', 'HotGas', 'Gas', 'Mass']
    for tautype in tautypes:
        for gastype in gastypes:
            key = tautype + '_' + gastype
            tau_keys.append(key)
            tauvals_dict[key] = np.array([0., 100.])

    # for backwards compatibility with Rohr+23 studying RPS in TNG50 jellyfish
    if tracers_flag and zooniverse_flag:
        tau_infall_key = tau_keys[3]
        tau_medpeak_key = tau_keys[0]
        tau_RPS_est_infall_key = 'tau_RPS_est'
        tau_RPS_tot_infall_key = 'tau_RPS_tot'
        tau_RPS_sRPS_key = 'tau_RPS_sRPS'
        tau_keys = [tau_infall_key, tau_medpeak_key, tau_RPS_est_infall_key, tau_RPS_tot_infall_key,
                    tau_RPS_sRPS_key]
                    
        tauvals_dict[tau_RPS_est_infall_key] = np.array([0., 90., 100.])
        tauvals_dict[tau_RPS_tot_infall_key] = np.array([0., 90., 100.])
        tauvals_dict[tau_RPS_sRPS_key] = np.array([0., 100.])
        
    print(tau_keys)
            
    # pick the datasets we want to output at given times for all subhalos
    if Config.centrals_flag:
        grp_keys = ['SnapNum', 'CosmicTime', 'HostGroup_M_Crit200',
                    'HostGroup_R_Crit200', 'HostSubhalo_Mstar_Rgal', 'SubhaloMass',
                    'Subhalo_Mstar_Rgal',
                    'SubhaloColdGasMass', 'SubhaloHotGasMass', 'SubhaloGasMass']
    else:
        grp_keys = ['SnapNum', 'CosmicTime', 'HostCentricDistance_norm', 'HostGroup_M_Crit200',
                    'HostGroup_R_Crit200', 'HostSubhalo_Mstar_Rgal', 'SubhaloMass',
                    'Subhalo_Mstar_Rgal',
                    'SubhaloColdGasMass', 'SubhaloHotGasMass', 'SubhaloGasMass',
                    'Nperipass', 'min_Dperi_norm', 'min_Dperi_phys', 'Napopass',
                    'min_HostCentricDistance_norm', 'min_HostCentricDistance_phys',
                    'SubhaloStellarPhotometricsU',
                    'SubhaloStellarPhotometricsB',
                    'SubhaloStellarPhotometricsV',
                    'SubhaloStellarPhotometricsK',
                    'SubhaloStellarPhotometricsg',
                    'SubhaloStellarPhotometricsr',
                    'SubhaloStellarPhotometricsi',
                    'SubhaloStellarPhotometricsz']
    if Config.TNGCluster_flag:
        grp_keys.append('SubhaloDMLowResMass')    
        
    # begin loop over subhalos
    for group_index, group_key in enumerate(result_keys):

        group = result[group_key]

        subfind_indices = np.where(group['SubfindID'] != -1)[0]
    
        # if just starting, then initialize the dictionary 
        if group_index == 0:
            tauresult['SubfindID'] = np.zeros(len(result_keys), dtype=int)
            tauresult['HostSubhaloGrNr'] = np.zeros(len(result_keys), dtype=int)
            for grp_key in grp_keys:
                for tau_key in tau_keys:
                    tau_vals = tauvals_dict[tau_key]
                    for tau_val in tau_vals:
                        tauresult_key = return_tauresult_key(grp_key, tau_key, tau_val)
                        tauresult[tauresult_key] = np.zeros(len(result_keys), 
                                                            dtype=group[grp_key].dtype) - 1
                tauresult_key = grp_key + '_z0'
                tauresult[tauresult_key] = np.zeros(len(result_keys),
                                                    dtype=group[grp_key].dtype) - 1

                if quench_flag:
                    tauresult_key = grp_key + '_quench'
                    tauresult[tauresult_key] = np.zeros(len(result_keys),
                                                        dtype=group[grp_key].dtype) - 1

            # also calculate tau at the quenching time
            if quench_flag:
                for tau_key in tau_keys:
                    tauresult_key = tau_key + '_quench'
                    tauresult[tauresult_key] = np.zeros(len(result_keys),
                                                        dtype=group[grp_key].dtype) - 1
                    
        tauresult['SubfindID'][group_index] = group['SubfindID'][0]
        tauresult['HostSubhaloGrNr'][group_index] = group['HostSubhaloGrNr'][0]
        # finish initializing the the result
        
        # for each of the definitions of tau, let's tabulate important properties at tau_X 
        for tau_key in tau_keys:
            tau = group[tau_key][subfind_indices]
            tau_vals = tauvals_dict[tau_key]
            for tau_val in tau_vals:
                if tau.max() >= tau_val:
                    tau_index = subfind_indices[np.where((tau - tau_val) >= 0)[0].max()]
                    for grp_key in grp_keys:
                        tauresult_key = return_tauresult_key(grp_key, tau_key, tau_val)
                        tauresult[tauresult_key][group_index] = group[grp_key][tau_index]
                        
        # and at z=0 -- this is always the first element in the arrays 
        for grp_key in grp_keys:
            tauresult_key = grp_key + '_z0'
            tauresult[tauresult_key][group_index] = group[grp_key][0]

        if quench_flag:
            # and at the quenching time, if this exists
            quench_snap = group['quenching_snap'][0]
            if quench_snap >= 0:
                SnapNum = group['SnapNum']
                quench_index = np.where(quench_snap == SnapNum)[0]
                # check that the quenching snap is after min(SnapNum)
                if quench_index.size > 0:
                    for grp_key in grp_keys:
                        tauresult_key = grp_key + '_quench'
                        tauresult[tauresult_key][group_index] = group[grp_key][quench_index]

                    # also calculate the tau values at quenching
                    for tau_key in tau_keys:
                        tauresult_key = tau_key + '_quench'
                        tauresult[tauresult_key][group_index] = group[tau_key][quench_index]

    # finish loop over the branches

    # hard code the characteristic cold gas loss timescales
    for tau_key in tau_keys:
        tau_vals = tauvals_dict[tau_key]
        tau_lo = tau_vals.min()
        tau_hi = tau_vals.max()
        key_lo = 'CosmicTime_%s%d'%(tau_key, tau_lo)
        key_hi = 'CosmicTime_%s%d'%(tau_key, tau_hi)
        lo = tauresult[key_lo]
        hi = tauresult[key_hi]
        indices = hi > 0
        
        tstrip = np.zeros(lo.size, dtype=lo.dtype) - 1
        tstrip[indices] = hi[indices] - lo[indices]
        
        tstrip_key = 'Tstrip_%s_tau%d-tau%d'%(tau_key, tau_lo, tau_hi)
        tauresult[tstrip_key] = tstrip

        # now hard code the quenching time: time of last quenching - tau_*_lo
        if quench_flag:
            key_hi = 'CosmicTime_quench'
            hi = tauresult[key_hi]
            indices = (lo > 0) & (hi > 0)
            
            tquench = np.zeros(lo.size, dtype=lo.dtype) -1
            tquench[indices] = hi[indices] - lo[indices]
            
            tquench_key = 'Tquench_%s'%tau_key
            tauresult[tquench_key] = tquench

    # and mu(z=0) = M_star^sat (z=0) / M_200c^host (z=0)
    tauresult['muz0'] = (tauresult['Subhalo_Mstar_Rgal_z0']
                         / tauresult['HostGroup_M_Crit200_z0'])
    
    # save the tau dictionary
    outdirec = Config.outdirec
    fname = Config.return_fnames(out_key)[1]
    if os.path.exists(outdirec + fname):
        print('File %s already exists. Deleting and recreating.'%(outdirec +fname))
        os.system('rm %s'%(outdirec + fname))
    with h5py.File(outdirec + fname, 'w') as outf:
        group = outf.require_group('Group')        
        for dset_key in tauresult.keys():  
            dset = tauresult[dset_key]
            dataset = group.require_dataset(dset_key, shape=dset.shape, dtype=dset.dtype)
            dataset[:] = dset

        outf.close()        
    
    return


def split_tau_gasz0(Config, split_key='SubhaloGasMass_z0', out_key=None):
    """
    Split the tau catalog into two samples: those with cold gas 
    at z=0, and those without 
    """
    
    outdirec = Config.outdirec
    fname = Config.return_fnames(out_key)[1]
    print(outdirec + fname)
    
    if not os.path.isfile(outdirec + fname):
        print('Warning file %s does not exist. Returning'%(outdirec + fname))
        return

    tau_dict = h5py.File(outdirec + fname, 'r')
    group = tau_dict['Group']

    # manually check either whether the there is no gas at z=0 or if tau100 has already been reached.
    if Config.min_snap == Config.max_snap and Config.min_snap == 99:
        if Config.TNGCluster_flag:
            mask = group[split_key][:] < 1.0e9
        else: 
            mask = group[split_key][:] <= 0.
    else:
        mask = np.logical_or(group[split_key][:] == 0., np.logical_and(group['CosmicTime_tau_infall_Gas100'][:] <= 13.7,
                                                                    group['CosmicTime_tau_infall_Gas100'][:] > 0.))
        
        if Config.TNGCluster_flag:
            mask = np.logical_or(group[split_key][:] < 1.0e9, np.logical_and(group['CosmicTime_tau_infall_Gas100'][:] <= 13.7,
                                                                            group['CosmicTime_tau_infall_Gas100'][:] > 0.))
            
        if Config.zooniverse_flag:
            mask = np.logical_or(group['SubhaloColdGasMass_z0'][:] == 0., group['CosmicTime_tau_RPS_tot100'][:] <= 13.7)
    
    result_gas = {}
    result_nogas = {}

    for group_key in group.keys():
        result_nogas[group_key] = group[group_key][mask]
        result_gas[group_key] = group[group_key][np.logical_not(mask)]

    fname_gas = fname[:-5] + '_gasz0.hdf5'
    fname_nogas = fname[:-5] + '_nogasz0.hdf5'
    
    # also make a symbolic link for all to maintain conventions
    fname_tot = fname[:-5] + '_tot.hdf5'
    # for symbolic links, they require full path names not relative
    outdirec_symlink = '/u/reric/Output/%s_subfindGRP/'%Config.sim
    if os.path.isfile(outdirec_symlink + fname_tot):
        os.system('rm %s'%(outdirec_symlink + fname_tot))
    os.symlink(outdirec_symlink + fname, outdirec_symlink + fname_tot)

    tau_fnames = [fname_gas, fname_nogas]
    results = [result_gas, result_nogas]

    for i, tau_fname in enumerate(tau_fnames):
        result = results[i]
        if (os.path.exists(outdirec + tau_fname)):
            os.system('rm %s'%(outdirec + tau_fname))
        with h5py.File(outdirec + tau_fname, 'w') as outf:
            group = outf.require_group('Group')
            for dset_key in result.keys():  
                dset = result[dset_key]
                dataset = group.require_dataset(dset_key, shape=dset.shape, dtype=dset.dtype)
                dataset[:] = dset

            outf.close() 

    return
    
    
def load_dict(fname, Config):
    """
    imports the hdf5 catalog and returns the dictionary.
    """
       
    result = {}
        
    with h5py.File(Config.outdirec + fname, 'a') as f:
        for group_key in f.keys():
            result[group_key] = {}
            for dset_key in f[group_key].keys():
                result[group_key][dset_key] = f[group_key][dset_key][:]
        f.close()
        
    return result


def return_outfname(Config, out_key=None, tau=False):
    """
    return the output filename.
    """
    if not out_key:
        if not tau:
            outfname = Config.GRPfname
        else:
            ftype = 'tau'
            if Config.zooniverse_flag:
                outfname = 'zooniverse_%s_%s_%s.hdf5'%(Config.sim, Config.zooniverse_key, ftype)
            elif Config.centrals_flag:
                outfname = 'central_subfind_%s_%s.hdf5'%(Config.sim, ftype)
            elif Config.allsubhalos_flag:
                outfname = 'all_subfind_%s_%s.hdf5'%(Config.sim, ftype)
            else:
                outfname = 'subfind_%s_%s.hdf5'%(Config.sim, ftype)
    else:
        if tau:
            ftype = 'tau'
            if Config.zooniverse_flag:
                outfname = 'zooniverse_%s_%s_clean_%s.hdf5'%(Config.sim, out_key, ftype)
                return outfname
        else:
            ftype = 'branches'
        if Config.zooniverse_flag:
            outfname = 'zooniverse_%s_%s_%s.hdf5'%(Config.sim, out_key, ftype)
        elif Config.centrals_flag:
            outfname = 'central_subfind_%s_%s_%s.hdf5'%(Config.sim, ftype, out_key)
        elif Config.allsubhalos_flag:
            outfname = 'all_subfind_%s_%s_%s.hdf5'%(Config.sim, ftype, out_key)
        else:
            outfname = 'subfind_%s_%s_%s.hdf5'%(Config.sim, ftype, out_key)
            
    return outfname

