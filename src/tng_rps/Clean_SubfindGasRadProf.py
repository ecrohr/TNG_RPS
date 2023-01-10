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
from importlib import reload

global sim, basePath, ins_key, jel_key, non_key
global snap_first, Nsnaps_PreProcessed, M200c0_lolim
global nonz0_key, beforesnapfirst_key, backsplash_key
global preprocessed_key, clean_key, out_keys
global outfname, outdirec, zooniverse

ins_key = 'inspected'
jel_key = 'jellyfish'
non_key = 'nonjellyf'
taudict_keys = [ins_key, jel_key, non_key]

snap_first = 67
Nsnaps_PreProcessed = 5
M200c0_lolim = 1.0e11

nonz0_key           = 'nonz0_keys'
beforesnapfirst_key = 'beforesnapfirst_keys'
backsplash_key      = 'backsplash_keys'
preprocessed_key    = 'preprocessed_keys'
clean_key           = 'clean_keys'

out_keys = [nonz0_key, beforesnapfirst_key, backsplash_key,
            preprocessed_key, clean_key]


def clean_zooniverseGRP(savekeys=False):

    
    dic        = load_dict(ins_key)
    keys_dic   = run_clean_zooniverseGRP(dic)
    final_keys = keys_dic[clean_key]
    
    # save each set of dic keys
    if (savekeys):
        for out_key in out_keys:
            keys     = keys_dic[out_key]
            fname = 'zooniverse_%s_%s_%s.txt'%(sim, ins_key, out_key)
            print('Writing file %s'%(outdirec + fname))
            with open(outdirec + fname, 'w') as f:
                for key in keys:
                    f.write('%s\n'%key)
                f.close()

    # save the cleaned branches using the subfindID at snap 99
    result = {}
    for key in final_keys:
        group = dic[key]

        new_key = '%08d'%(group['SubfindID'][0])
        result[new_key] = group

    fname = return_outfname(clean=True)
    with h5py.File(outdirec + fname, 'a') as outf:
        for group_key in result.keys():
            group = outf.require_group(group_key)
            for dset_key in result[group_key].keys():
                dset = result[group_key][dset_key]
                dataset = group.require_dataset(dset_key, shape=dset.shape, dtype=dset.dtype)
                dataset[:] = dset
                
        outf.close()
    

    # now split the inspected branches into jellyfish, if there's a jellyfish classificaiton
    # at snap >= snap_first, and into nonjellyf, if there are no jelly classiifications at snap >= snap_first
    # this means that some of the branches with a jellyfish classification may become nonjellyf branches!
    split_inspected_branches()
    # reorganize each of the three sets of branches [inspected, jellyfish, nonjellyf] into tau dictionaries
 

    keys = ['inspected', 'jellyfish', 'nonjellyf']
    for key in keys:
        _ = return_taudict(key)

    return


def run_clean_zooniverseGRP(dic):

    keys = np.array(list(dic.keys()))

    print('There are %d total %s zooniverse branhces in %s.'%(len(keys), ins_key, sim))

    # let's clean the zooniverse jellyfish branches, noting the keys of the objects
    # excised at each step

    subfindsnapshot_flags = load_subfindsnapshot_flags()

    # initalize empty lists to hold the various keys
    clean_keys           = []
    beforesnapfirst_keys = []
    nonz0_keys           = []
    backsplash_keys      = []
    preprocessed_keys    = []

    for key in keys:
        group = dic[key]

        # ignore the snaps where the subahlo was not identified
        indices = group['SubfindID'] != -1

        SnapNum            = group['SnapNum'][indices]
        SubfindID          = group['SubfindID'][indices]
        jel_flags          = group['jel_flags'][indices]
        ins_flags          = group['ins_flags'][indices]
        central_flags      = group['central_flags'][indices]
        memberlifof_flags  = group['memberlifof_flags'][indices]
        preprocessed_flags = group['preprocessed_flags'][indices]

        # 1. the MDB must reach z=0 (snap 99)
        if max(SnapNum) < 99:
            nonz0_keys.append(key)
            continue

        # 2. must have at least one inspection at snap >= snap_first
        ins_flag = max(ins_flags[SnapNum >= snap_first])
        if not (ins_flag):
            beforesnapfirst_keys.append(key)
            continue
            
        # because this MDB reaches z=0, load the relevant flags from SubfindSnapshot cat
        SubfindID_z0  = SubfindID[0]
        subfind_flags = subfindsnapshot_flags['%08d'%SubfindID_z0]
        in_tree       = subfind_flags['in_tree']
        intree_indices = in_tree == 1
                
        central    = subfind_flags['central'][intree_indices]
        host_m200c = subfind_flags['host_m200c'][intree_indices] >= M200c0_lolim
        in_z0_host = subfind_flags['in_z0_host'][intree_indices]
        
        # 3. no backsplash galaxies -- must not be a central at z=0 (snap 99)
        if (central[0]):
            backsplash_keys.append(key)
            continue
            
        # 4. no pre-processed galaxies -- galaxy must not be a satellite of
        #    a group of mass M200c > Mlolim other than its z=0 host for more than
        #    NSnaps_PreProcessed consecutive snaps.
        #    only consider times after tau_0 (time of max cold gas mass)
        #    note that we can only check for branches that have a defined tau_0
        #    and that there are at least Nsnaps_check snaps after tau_0 
        tau_medpeak = group['tau_medpeak']
        if np.max(tau_medpeak) > 0:
            tau0_index = np.max(np.argwhere(tau_medpeak >= 0))
            central    = central[:tau0_index]
            in_z0_host = in_z0_host[:tau0_index]
            host_m200c = host_m200c[:tau0_index]
            if len(in_z0_host) > Nsnaps_PreProcessed:
                preprocessed_indices = ~central & ~in_z0_host & host_m200c
                preprocessed_check = [True] * Nsnaps_PreProcessed
                if (ru.is_slice_in_list(preprocessed_check, list(preprocessed_indices))):
                    preprocessed_keys.append(key)
                    continue
                    
        # galaxy has passed every test -- add to the cleaned list
        clean_keys.append(key)
        
    # end loop over the branches
        
    Ntotal = (len(clean_keys) + len(beforesnapfirst_keys) + len(nonz0_keys) +
              len(backsplash_keys) + len(preprocessed_keys))
    if Ntotal != len(keys):
        print('Mismatch in total number of keys!')

    print('%s total %s branches: %d; not reaching z=0: %d'%(sim, ins_key, len(keys), len(nonz0_keys)))
    print('not inspected since %d: %d'%(snap_first, len(beforesnapfirst_keys)))
    print('backsplash: %d; preprocessed: %d'%(len(backsplash_keys), len(preprocessed_keys)))
    print('clean: %d'%(len(clean_keys)))

    result       = {}

    result_keys  = [nonz0_key,
                    beforesnapfirst_key,
                    backsplash_key,
                    preprocessed_key,
                    clean_key]
    
    result_dsets = [nonz0_keys,
                    beforesnapfirst_keys, 
                    backsplash_keys,
                    preprocessed_keys,
                    clean_keys]

    for result_i, result_key in enumerate(result_keys):
        result[result_key] = result_dsets[result_i]

    return result


def load_subfindsnapshot_flags():
    
    direc = '../Output/%s_subfindflags/'%sim
    fname = 'subfindflags_%s_zooniverse.hdf5'%(sim)
    result = {}
    
    with h5py.File(direc + fname, 'r') as f:
        for group_key in f.keys():
            result[group_key] = {}
            for dset_key in f[group_key]:
                result[group_key][dset_key] = f[group_key][dset_key][:]
                
        f.close()
        
    return result

# split the inspected branches into jellyfish and nonjellyf
def split_inspected_branches():
    
    keys = ['inspected', 'jellyfish', 'nonjellyf']
    fnames = []
    for key in keys:
        fname = outdirec + 'zooniverse_%s_%s_branches_clean.hdf5'%(sim, key)
        fnames.append(fname)
        
        # if the jellyfish and nonjellyf files exist, delete them 
        if key != keys[0]:
            if (os.path.exists(fname)):
                os.system('rm %s'%fname)
    
    insf = h5py.File(fnames[0], 'a')
    jelf = h5py.File(fnames[1], 'a')
    nonf = h5py.File(fnames[2], 'a')

    jelbeforesnapfirst_keys = []
    
    for group_key in insf.keys():
        group = insf[group_key]
        SnapNum = group['SnapNum'][:]
        jel_flag = np.max(group['jel_flags'][SnapNum >= snap_first])
        if (jel_flag):
            insf.copy(insf[group_key], jelf)
        else:
            insf.copy(insf[group_key], nonf)
            # check if the galaxy was a jellyfish before snap_first
            if (min(SnapNum) < snap_first):
                if (np.max(group['jel_flags'][SnapNum < snap_first])):
                    jelbeforesnapfirst_keys.append(group_key)

    print('Of the %d inspected clean branches in %s'%(len(insf.keys()), sim))
    print('%d are jellyfish and %d are nonjellyf.'%(len(jelf.keys()), len(nonf.keys())))
    print('%d were jellyfish before snap %d but are included in nonjellyf.'%(len(jelbeforesnapfirst_keys),
                                                                             snap_first))

    insf.close()
    jelf.close()
    nonf.close()

    return


# reorganize the evolutionary tracks into 1D arrays with scalars at specific times
# we want an array of scalar quantities at important times for various plots.
# namely, we care about times tau0, tau10, and tau90 defined by infall and peak MCgas, at z=0, and quenching time

def return_taudict(key):
            
    def return_tauresult_key(grp_key, tau_key, tau_val):
        return (grp_key + '_' + tau_key + '%d'%tau_val)

    result = load_dict(key, clean=True)
    result_keys = list(result.keys())
    result_keys.sort()

    tauresult = {}
    tau_infall_key = 'tau_infall'
    tau_medpeak_key = 'tau_medpeak'
    tau_RPS_est_infall_key = 'tau_RPS_est'
    tau_RPS_tot_infall_key = 'tau_RPS_tot'
    tau_RPS_sRPS_key = 'tau_RPS_sRPS'
    tau_keys = [tau_infall_key, tau_medpeak_key, tau_RPS_est_infall_key, tau_RPS_tot_infall_key,
                tau_RPS_sRPS_key]

    tauvals_dict = {}
    tauvals_dict[tau_infall_key] = np.array([0., 90.])
    tauvals_dict[tau_medpeak_key] = np.array([0., 10., 90.])
    tauvals_dict[tau_RPS_est_infall_key] = np.array([0., 90., 100.])
    tauvals_dict[tau_RPS_tot_infall_key] = np.array([0., 90., 100.])
    tauvals_dict[tau_RPS_sRPS_key] = np.array([0., 100.])
            
    grp_keys = ['SnapNum', 'CosmicTime', 'HostCentricDistance_norm', 'HostGroup_M_Crit200',
                'HostGroup_R_Crit200', 'HostSubhalo_Mstar_Rgal', 'SubhaloMass',
                'Subhalo_Mstar_Rgal',
                'SubhaloColdGasMass', 'SubhaloHotGasMass', 'SubhaloGasMass',
                'Nperipass', 'min_Dperi_norm', 'min_Dperi_phys', 'Napopass',
                'min_HostCentricDistance_norm', 'min_HostCentricDistance_phys']
    
    for group_index, group_key in enumerate(result_keys):
        group = result[group_key]
        subfind_indices = np.where(group['SubfindID'] != -1)[0]
    
        # if just starting, then initialize the dictionary 
        if group_index == 0:
            tauresult['SubfindID'] = np.zeros(len(result_keys), dtype=int)
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

                tauresult_key = grp_key + '_quench'
                tauresult[tauresult_key] = np.zeros(len(result_keys),
                                                    dtype=group[grp_key].dtype) - 1

            # also calculate tau at the quenching time
            for tau_key in tau_keys:
                tauresult_key = tau_key + '_quench'
                tauresult[tauresult_key] = np.zeros(len(result_keys),
                                                    dtype=group[tau_key].dtype) - 1
                    
        tauresult['SubfindID'][group_index] = int(float(group_key))
    
        # for each of the definitions of tau, let's tabulate important properties at tau_X 
        for tau_key in tau_keys:
            tau = group[tau_key][subfind_indices]
            tau_vals = tauvals_dict[tau_key]
            for tau_val in tau_vals:
                if tau.max() >= tau_val:
                    if tau_val == tau_vals[0]:
                        tau_index = subfind_indices[np.where((tau_val - tau) >= 0)[0].min()]
                    elif (tau_val == tau_vals[-1]):
                        tau_index = subfind_indices[np.where((tau - tau_val) >= 0)[0].max()]
                    else:
                        tau_index = subfind_indices[np.where((tau - tau_val) >= 0)[0].max()]
                    for grp_key in grp_keys:
                        tauresult_key = return_tauresult_key(grp_key, tau_key, tau_val)
                        tauresult[tauresult_key][group_index] = group[grp_key][tau_index]
                        
        # and at z=0 -- this is always the first element in the arrays 
        for grp_key in grp_keys:
            tauresult_key = grp_key + '_z0'
            tauresult[tauresult_key][group_index] = group[grp_key][0]

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
        if tau_key == tau_medpeak_key:
            tau_lo = tau_vals[1]
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
    fname = 'zooniverse_%s_%s_clean_tau.hdf5'%(sim, key)
    with h5py.File(outdirec + fname, 'a') as outf:
        group = outf.require_group('Group')        
        for dset_key in tauresult.keys():  
            dset = tauresult[dset_key]
            dataset = group.require_dataset(dset_key, shape=dset.shape, dtype=dset.dtype)
            dataset[:] = dset

        outf.close()        
    
    return tauresult


def combine_taudicts():
    """
    Combine the TNG50-1 and TNG100-1 tau dictionaries
    """

    # load both tau files
    sims = ['TNG50-1', 'TNG100-1']
    keys = ['jellyfish', 'nonjellyf', 'inspected']

    for key in keys:

        print(key)

        fname = 'zooniverse_%s_%s_clean_tau.hdf5'%(sims[0], key)
        f0 = h5py.File(outdirec + fname, 'r')
        group0 = f0['Group']

        fname = 'zooniverse_%s_%s_clean_tau.hdf5'%(sims[1], key)
        f1 = h5py.File(outdirec + fname, 'r')
        group1 = f1['Group']


        sim = sims[0] + '+' + sims[1]
        outfname = 'zooniverse_%s_%s_clean_tau.hdf5'%(sim, key)
        with h5py.File(outdirec + outfname, 'a') as outf:
            group = outf.require_group('Group')
            for dset_key in group0.keys():
                dset = np.concatenate([group0[dset_key], group1[dset_key]])
                datset = group.require_dataset(dset_key,  shape=dset.shape, dtype=dset.dtype)
                datset[:] = dset
            outf.close()

        f0.close()
        f1.close()

    return

def split_tau_gasz0(sim='TNG50-1', key=jel_key):
    """
    Split the tau catalog into two samples: those with cold gas 
    at z=0, and those without 
    """

    fname = 'zooniverse_%s_%s_clean_tau.hdf5'%(sim, key)
    tau_dict = h5py.File(outdirec + fname, 'r')
    group = tau_dict['Group']
    split_key = 'SubhaloColdGasMass_z0'
    split_key_alt = 'CosmicTime_tau_RPS_tot100'
    split_key_alt_check = group['CosmicTime_z0'][0]

    mask = (group[split_key][:] == 0) or (group[split_key_alt[:] < split_key_alt_check)

    result_gas = {}
    result_nogas = {}

    for group_key in group.keys():
        result_nogas[group_key] = group[group_key][mask]
        result_gas[group_key] = group[group_key][~mask]

    fname_gas = 'zooniverse_%s_%s_clean_tau_gasz0.hdf5'%(sim, key)
    fname_nogas = 'zooniverse_%s_%s_clean_tau_nogasz0.hdf5'%(sim, key)

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

def load_dict(key, clean=False):
    """
    imports the hdf5 catalog and returns the dictionary.
    """
    
    # key == [inspected, jellyfish, nonjellyf] -- otherwise file doesn't exist
    fname = return_outfname(sim=sim, key=key, zooniverse=zooniverse, clean=clean)
    
    result = {}
        
    with h5py.File(outdirec + fname, 'a') as f:
        for group_key in f.keys():
            result[group_key] = {}
            for dset_key in f[group_key].keys():
                result[group_key][dset_key] = f[group_key][dset_key][:]
        f.close()
        
    return result


def return_outfname(sim='TNG50-1', key='inspected', zooniverse=True, clean=False):
    """
    return the output filename.
    """
    if not zooniverse:
        outfname = 'subfind_%s_branches.hdf5'%sim
        return outfname
    else:
        outfname = 'zooniverse_%s_%s_branches'%(sim, key)
        if clean:
            outfname += '_clean.hdf5'
        else:
            outfname += '.hdf5'
        return outfname

zooniverse = False
for sim in ['TNG50-1']:
    outdirec = '../Output/%s_subfindGRP/'%sim
    outfname = return_outfname(sim=sim, key=ins_key, zooniverse=zooniverse, clean=False)
    
    #clean_zooniverseGRP(savekeys=True)

for key in taudict_keys:
    #_ = return_taudict(key)
    split_tau_gasz0(key=key)

#combine_taudicts()
