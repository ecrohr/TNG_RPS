#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 09:29:53 2021

@author: rohr
"""

### import modules
import illustris_python as il
import numpy as np
from scipy.signal import argrelextrema
import h5py
import rohr_utils as ru 
import os
import glob
import multiprocessing as mp
from functools import partial
from tenet.util import sphMap
from scipy.ndimage import gaussian_filter

scalar_keys = ['SubhaloColdGasMass', 'SubhaloGasMass', 'SubhaloHotGasMass']
threed_keys = ['radii', 'vol_shells',
               'SubhaloColdGasMassShells', 'SubhaloColdGasDensityShells',
               'SubhaloHotGasMassShells', 'SubhaloHotGasDensityShells',
               'SubhaloGasMassShells', 'SubhaloDensityShells']

# hardcode the snapshots of interest
zooniverse_snapshots_TNG50 = [99, 98, 97, 96, 95, 94, 93, 92, 91, 90,
                              89, 88, 87, 86, 85, 84, 83, 82, 81, 80,
                              79, 78, 77, 76, 75, 74, 73, 72, 71, 70,
                              69, 68, 67, 59, 50, 40, 33]

zooniverse_snapshots_TNG100 = [99, 91, 84, 78, 72, 67, 59, 50, 40, 33]

def run_subfindGRP(Config):
    """
    Run the Create_SubfindGRP module.
    """

    add_memberflags(Config)
    add_times(Config)
    
    outdirec = Config.outdirec
    outfname = Config.outfname

    if Config.run_SGRP:
        dics = []
        
        f = h5py.File(outdirec + outfname, 'a')
        for group_key in f.keys():
            dic = {}
            dic[group_key] = {}
            group = f[group_key]
            for dset_key in group.keys():
                dic[group_key][dset_key] = group[dset_key][:]
            dics.append(dic)

        Pool = mp.Pool(Config.Nmpcores)
        
        print(len(dics))

        if Config.mp_flag:
            result_list = Pool.map(partial(create_subfindGRP, Config=Config), dics)
        else:
            result_list = []
            for dic in dics:
                result_list.append(create_subfindGRP(dic, Config))

        Pool.close()
        Pool.join()

        result = {}
        for i, d in enumerate(result_list):
            key = list(d.keys())[0]
            result[key] = d[key]

        new_keys = threed_keys + scalar_keys

        for group_key in result.keys():
            group = f.require_group(group_key)
            for dset_key in new_keys:
                dset = result[group_key][dset_key]
                dataset = group.require_dataset(dset_key, shape=dset.shape, dtype=dset.dtype)
                dataset[:] = dset
                
        f.close()
    
    if Config.run_SGRP_PP:
        run_postprocessing(Config)
    
    return
    
def run_postprocessing(Config):
    """
    run all post processing functions
    """
    
    # standard for all branches 
    if Config.min_snap != Config.max_snap:
        add_gastau(Config)
    
    # for satellites only
    if not Config.centrals_flag:
        add_dmin(Config)
        add_Nperipass(Config)
        # quenching times require the appropriate catalogs
        if not Config.TNGCluster_flag:
            add_quenchtimes(Config)
        else:
            # LBE Ram Pressure only for TNGCluster:
            add_LBEramPressure(Config)

    # only for TNG-Cluster centrals
    elif Config.TNGCluster_flag:
        add_Lxmaps(Config)

    # if the tracers have already been calculated
    if Config.tracers_flag:
        add_tracers(Config)
        add_tracers_postprocessing(Config)
        add_coldgasmasstracerstau(Config)

    # for zooniverse only
    if Config.zooniverse_flag:
        add_zooniverseflags(Config)
        
    return
    

def create_subfindGRP(dic, Config):
    """
    given the subhalo dictionary, calculate and add the gas radial profile
    """
        
    # create a directory for the given subhalo,
    # named after its subfindID at a given SnapNum
    dict_list = []

    gal_key   = list(dic.keys())[0]
    gal       = dic[gal_key]
    SubfindID = gal['SubfindID'].copy()
    SnapNum   = gal['SnapNum'].copy()

    # if TNG-Cluster satellties, only calculate between z=0 and infall or else it takes too long
    if Config.TNGCluster_flag and not Config.centrals_flag:
        M200c0_lolim_PP = Config.M200c0_lolim_PP
        Nsnaps_PP = Config.Nsnaps_PP
        group = gal
        subhalo_indices = np.where(group['SubfindID'][:] != -1)[0]
        # find infall time as the first time subhalo was a satellite for Nsnaps_PP consecutive snapshots
        satellite_indices = ((group['central_flags'][subhalo_indices] == 0) &
                             (group['HostGroup_M_Crit200'][subhalo_indices] >= M200c0_lolim_PP))

        satellite_check = [True] * Nsnaps_PP
        satellite_indices_bool = ru.where_is_slice_in_list(satellite_check, satellite_indices)

        if any(satellite_indices_bool):
            # from this first time that the subhalo was a satellite, find the index of
            # the first conescutive snapshot.
            infall_tau_index = np.where(satellite_indices_bool)[0].max()
            SnapNum[infall_tau_index+2:] = -1
            SubfindID[infall_tau_index+2:] = -1
        else:
            SnapNum[1:] = -1
            SubfindID[1:] = -1
    
    for snapnum_index, snapnum in enumerate(SnapNum):
        subfindID = SubfindID[snapnum_index]
    
        dict_list.append(return_subfindGRP(snapnum, subfindID, Config))
    
    radii_dict = {}
    for d in dict_list:
        radii_dict.update(d)

    # initialize and fill result dicitonary
    # note that threed_keys are vectors, and scalar_keys are scalars
    shape           = (len(SnapNum), len(radii_dict['%d'%SnapNum[0]]['radii']))
    result          = {}
    result[gal_key] = gal
    for key in threed_keys:
        result[gal_key][key] = np.zeros(shape, dtype=float)
    for key in scalar_keys:
        result[gal_key][key] = np.zeros(shape[0])

    for row, snap_key in enumerate(radii_dict.keys()):
        for key in threed_keys:
            result[gal_key][key][row,:] = radii_dict[snap_key][key]
        for key in scalar_keys:
            result[gal_key][key][row]   = radii_dict[snap_key][key]

    return result


def return_subfindGRP(snapnum, subfindID, Config):
    """
    for the given snapnum and subfindID, calculate the gas properties
    """
    
    print('return_subfindGRP(): Working on %s snap %03d subfindID %08d'%(Config.sim, snapnum, subfindID))
    
    sim = Config.sim
    basePath = Config.basePath
    h = Config.h
    tlim = Config.tlim
    star_ptn = Config.star_ptn
    gas_ptn = Config.gas_ptn
    centrals_flag = Config.centrals_flag
    gas_lolim = Config.gas_lolim
    
    ### define radial bins and bincenters ###
    if centrals_flag:
        rmin_norm = 10.**(-2) # r / Rvir
        rmax_norm = 3.0 # r / Rvir
        radii_binwidth = 0.1 # r / Rvir, linear

        radii_bins_norm, radii_bincents_norm = ru.returnlogbins([rmin_norm, rmax_norm], radii_binwidth)
        radii_bins_norm = np.insert(radii_bins_norm, 0, 0.)
        radii_bincents_norm = np.insert(radii_bincents_norm, 0, radii_bins_norm[1]/2.)
        #radii_bins_norm = np.arange(rmin_norm, rmax_norm + radii_binwidth*1.0e-3, radii_binwidth)
        #radii_bincents_norm = (radii_bins_norm[1:] + radii_bins_norm[:-1]) / 2.

    else:
        radii_binwidth = 0.1 # r / rgal, log spacing
        rmin_norm = 10.**(-1.) # [r/rgal]
        rmax_norm = 10.**(2.)  # [r/rgal]
        radii_bins_norm, radii_bincents_norm = ru.returnlogbins([rmin_norm, rmax_norm], radii_binwidth)

        # prepend 0 to the radial bins to capture the center sphere
        radii_bins_norm = np.insert(radii_bins_norm, 0, 0.)
        radii_bincents_norm = np.insert(radii_bincents_norm, 0, radii_bins_norm[1]/2.)

    nbins = radii_bincents_norm.size
      
    # initialize result
    result            = {}
    group_key         = '%d'%snapnum
    result[group_key] = {}
    for threed_key in threed_keys:
        result[group_key][threed_key] = np.zeros(nbins, dtype=float) - 1.
    for scalar_key in scalar_keys:
        result[group_key][scalar_key] = -1.
  
    # check if the subhalo is identified at this snap
    if subfindID == -1:
        return result 

    # load general simulation parameters
    header  = il.groupcat.loadHeader(basePath, snapnum)
    a       = header['Time'] # scale factor
    boxsize = header['BoxSize'] * a / h
        
    subhalofields = ['SubhaloHalfmassRadType', 'SubhaloPos', 'SubhaloGrNr']
    gasfields     = ['Coordinates', 'Masses', 'InternalEnergy',
                     'ElectronAbundance', 'StarFormationRate']
            
    subhalo      = ru.loadSingleFields(basePath, snapnum, subhaloID=subfindID, fields=subhalofields)
    subhalopos   = subhalo['SubhaloPos'] * a / h
    subhalo_rgal = 2. * subhalo['SubhaloHalfmassRadType'][star_ptn] * a / h
                     
    if centrals_flag:
        R200c = ru.loadSingleFields(basePath, snapnum, haloID=subhalo['SubhaloGrNr'], fields=['Group_R_Crit200']) * a / h
                            
    # load gas particles for relevant halo
    gasparts = il.snapshot.loadSubhalo(basePath, snapnum, subfindID, gas_ptn, fields=gasfields)
    
    # if the satellite has no gas, write 0 for all dsets
    if gasparts['count'] == 0:
        for threed_key in threed_keys:
            result[group_key][threed_key][:] = 0
        for scalar_key in scalar_keys:
            result[group_key][scalar_key] = 0
        return result
    
    gas_coordinates        = gasparts['Coordinates'] * a / h
    gas_masses             = gasparts['Masses'] * 1.0e10 / h
    gas_internalenergies   = gasparts['InternalEnergy']
    gas_electronabundances = gasparts['ElectronAbundance']
    gas_starformationrates = gasparts['StarFormationRate']
    
    if Config.centrals_flag:
        radii_bins     = radii_bins_norm * R200c # pkpc
        radii_bincents = radii_bincents_norm * R200c # pkpc
        vol_shells = (4./3.) * np.pi * ((radii_bins[1:])**3 - radii_bins[:-1]**3)
                  
    else:
        radii_bins     = radii_bins_norm * subhalo_rgal # pkpc
        radii_bincents = radii_bincents_norm * subhalo_rgal # pkpc
    
        # set the volume of the shells; len(volume) = len(bincents) = len(bins) - 1
        vol_shells = (4./3.) * np.pi * ((radii_bins[2:])**3 - (radii_bins[1:-1])**3)
        vol_shells = np.insert(vol_shells, 0, (4./3.) * np.pi * (radii_bins[1])**3)

    # save the total amount of gas
    subhalo_gasmass = np.sum(gas_masses)
    
    # calculate the distance between each gas cell and subhalo center  
    gas_radii = ru.mag(gas_coordinates, subhalopos, boxsize)
    
    # calculate the temperature of each gas cell
    gas_temperatures = ru.calc_temp(gas_internalenergies, gas_electronabundances, gas_starformationrates)
    
    # separate the gas into cold component
    coldgas_masses = gas_masses[gas_temperatures < tlim] 
    coldgas_radii  = gas_radii[gas_temperatures < tlim]
    
    hotgas_masses = gas_masses[gas_temperatures >= tlim]
    hotgas_radii = gas_radii[gas_temperatures >= tlim]
    
    # calculate and save the total cold and hot gas masses
    subhalo_coldgasmass = np.sum(coldgas_masses)
    subhalo_hotgasmass  = np.sum(hotgas_masses)    
    # sort the gas masses by their radius
    coldgas_masses = coldgas_masses[np.argsort(coldgas_radii)]
    coldgas_radii  = coldgas_radii[np.argsort(coldgas_radii)]
    
    hotgas_masses = hotgas_masses[np.argsort(hotgas_radii)]
    hotgas_radii  = hotgas_radii[np.argsort(hotgas_radii)]

    gas_masses = gas_masses[np.argsort(gas_radii)]
    gas_radii = gas_radii[np.argsort(gas_radii)]
    
    # calculate the radial profile via histogram                      
    coldgas_mass_shells = np.histogram(coldgas_radii, bins=radii_bins, weights=coldgas_masses)[0]
    hotgas_mass_shells = np.histogram(hotgas_radii, bins=radii_bins, weights=hotgas_masses)[0]
    gas_mass_shells = np.histogram(gas_radii, bins=radii_bins, weights=gas_masses)[0]

    coldgas_densities_shells = coldgas_mass_shells / vol_shells
    hotgas_densities_shells = hotgas_mass_shells / vol_shells
    gas_densities_shells = gas_mass_shells / vol_shells

    dsets = [radii_bincents, vol_shells,
             coldgas_mass_shells, coldgas_densities_shells,
             hotgas_mass_shells, hotgas_densities_shells,
             gas_mass_shells, gas_densities_shells]
    
    scalars = [subhalo_coldgasmass, subhalo_gasmass, subhalo_hotgasmass]
             
    for threed_index, threed_key in enumerate(threed_keys):
        result[group_key][threed_key] = dsets[threed_index]
            
    for scalar_index, scalar_key in enumerate(scalar_keys):
        result[group_key][scalar_key] = scalars[scalar_index]

    return result


### post processing functions ###
def add_memberflags(Config):
    """
    add membership flags -- central flag, pre-processed flag, member of final FoF flag
    this should be updated to use the subfindsnapshot flags rather than recalculating them here
    """

    f = h5py.File(Config.outdirec + Config.outfname, 'a')

    keys = ['central_flags', 'preprocessed_flags', 'memberlifof_flags']

    for group_key in f:
        group = f[group_key]
        SubfindID = group['SubfindID'][:]
        SubhaloGroupFirstSub = group['SubhaloGroupFirstSub'][:]
        SubhaloGrNr = group['SubhaloGrNr'][:]
        HostSubhaloGrNr = group['HostSubhaloGrNr'][:]
        HostSubfindID = group['HostSubfindID'][:]

        flags = np.zeros(len(SubfindID), dtype=int)
        
        # flag when the subhalo was a central galaxy
        # i.e., it's the primrary subhalo in the group
        centralflags = flags.copy()
        central_indices = np.where((SubfindID == SubhaloGroupFirstSub) &
                                   (SubfindID != -1))[0]
        centralflags[central_indices] = 1

        # flag when the subhalo was preprocessed
        # i.e., it's a satellite of a FoF that's NOT it's last identified host
        preprocflags = flags.copy()
        preproc_indices = np.where((SubhaloGrNr != HostSubhaloGrNr) &
                                   (SubfindID != SubhaloGroupFirstSub) &
                                   (SubfindID != -1))[0]
        preprocflags[preproc_indices] = 1

        # flag when the subhalo was in its last identified FoF
        # note that min(inLIFoF_indices) is the infall time
        inLIFoFflags = flags.copy()
        inLIFoF_indices = np.where((SubhaloGrNr == HostSubhaloGrNr) &
                                   (SubfindID != -1))[0]
        inLIFoFflags[inLIFoF_indices] = 1

        dsets = [centralflags, preprocflags, inLIFoFflags]
        for dset_index, dset_key in enumerate(keys):
            dset = dsets[dset_index]
            dataset = group.require_dataset(dset_key, shape=dset.shape, dtype=dset.dtype)
            dataset[:] = dset

    f.close()

    return


def add_zooniverseflags(Config):
    """
    Load the zooniverse catalogs and add inspected + jellyfish flags to the branches.
    Saves the flags directly to the GRP catalogs. No returns.
    """

    # load the inpsected and jellyfish ID dictionaries
    insIDs_dict, Scores_dict = load_zooniverseIDs(Config)

    f = h5py.File(Config.outdirec + Config.outfname, 'a')

    keys = ['ins_flags', 'jel_flags', 'jel_flags_raw', 'ScoreRaw', 'ScoreAdjusted']

    if Config.sim == 'TNG50-1':
        zooniverse_snapshots = zooniverse_snapshots_TNG50
    elif Config.sim == 'TNG100-1':
        zooniverse_snapshots = zooniverse_snapshots_TNG100
    else:
        raise ValueError('Config.sim %s and zooniverse flag are incompatible.'%Config.sim)

    for group_key in f.keys():
        group     = f[group_key]
        SubfindID = group['SubfindID'][:]
        SnapNum   = group['SnapNum'][:]
        
        ins_flags = np.zeros(SnapNum.size, dtype=int) - 1
        jel_flags = ins_flags.copy()
        jel_flags_raw = ins_flags.copy()

        ScoreRaw = np.zeros(SnapNum.size, dtype=float) - 1
        ScoreAdjusted = ScoreRaw.copy()

        for index, snap in enumerate(SnapNum):
            snap_key = 'Snapshot_%03d'%snap
            subfindID = SubfindID[index]
            if subfindID == -1:
                continue
            if snap not in zooniverse_snapshots:
                continue
            insIDs = insIDs_dict[snap_key]
            if subfindID not in insIDs:
                continue
            ins_flags[index] = 1
            jel_flags_raw[index] = 0
            jel_flags[index] = 0
            subfind_index = np.where(subfindID == insIDs)[0]
            ScoreRaw[index] = Scores_dict[snap_key]['ScoreRaw'][subfind_index]
            if ScoreRaw[index] >= Config.jellyscore_min:
                jel_flags_raw[index] = 1
            ScoreAdjusted[index] = Scores_dict[snap_key]['ScoreAdjusted'][subfind_index]
            if ScoreAdjusted[index] >= Config.jellyscore_min:
                jel_flags[index] = 1

        # check that the key maximum inspected flag is 1 and jellyfish flag is 0
        assert np.max(ins_flags) == 1
        assert np.max(jel_flags) >= 0

        # finish loop over SnapNum for the given group (subhalo)
        dsets = [ins_flags, jel_flags, jel_flags_raw, ScoreRaw, ScoreAdjusted]
        for i, key in enumerate(keys):
            dset       = dsets[i]
            dataset    = group.require_dataset(key, shape=dset.shape, dtype=dset.dtype)
            dataset[:] = dset

    # finish loop over groups (subhalos)
    f.close()

    return

def load_zooniverseIDs(Config):
    """
    Load all zooniverse catalogs. Create a dictionary with each snapshot as the key,
    and the entries are the subfindIDs of all inspected galaxies at that snapshot.
    Creates a second dictionary for when the galaxy is a jellyfish.
    Returns the dictionaries.
    """
    
    indirec  = '../IllustrisTNG/%s/postprocessing/Zooniverse_CosmologicalJellyfish/'%Config.sim
    infname  = 'jellyfish.hdf5'
    jellyfish = h5py.File(indirec + infname, 'r')
    
    insIDs_dict = {}
    Scores_dict = {}

    if Config.sim == 'TNG50-1':
        zooniverse_snapshots = zooniverse_snapshots_TNG50
    elif Config.sim == 'TNG100-1':
        zooniverse_snapshots = zooniverse_snapshots_TNG100
    else:
        raise ValueError('Config.sim %s and zooniverse flag are incompatible.'%Config.sim)
    
    for snapshot in zooniverse_snapshots:
        key = 'Snapshot_%03d'%snapshot
        group = jellyfish[key]
        
        SubhaloIDs = group['SubhaloIDs'][:]
        ScoreAdjusted = group['ScoreAdjusted'][:]
        ScoreRaw = group['ScoreRaw'][:]
        
        insIDs_dict[key] = SubhaloIDs
        
        Scores_dict[key] = {}
        Scores_dict[key]['ScoreAdjusted'] = ScoreAdjusted
        Scores_dict[key]['ScoreRaw'] = ScoreRaw
        
    jellyfish.close()
    
    return insIDs_dict, Scores_dict

    
def add_times(Config):
    """
    Add redshift, cosmic time, and scale factor.
    """

    # tabulate redshift, scale factor, and calculate cosmic time
    # as functions of the snap number
    f = h5py.File(Config.outdirec+Config.outfname, 'a')

    keys = ['CosmicTime', 'Redshift', 'Time']
    dsets = [Config.CosmicTimes, Config.Redshifts, Config.Times]
    for group_key in f.keys():
        group = f[group_key]
        for dset_index, dset_key in enumerate(keys):
            dset = dsets[dset_index]
            dataset = group.require_dataset(dset_key, shape=dset.shape, dtype=dset.dtype)
            dataset[:] = dset

    f.close()

    return


def add_dmin(Config):
    """
    add the min distance to the host.
    """
    f = h5py.File(Config.outdirec+Config.outfname, 'a')
    keys = ['min_HostCentricDistance_norm', 'min_HostCentricDistance_phys']

    for group_key in f.keys():
        group = f[group_key]

        HostCentricDistance_phys = group['HostCentricDistance_phys'][:]        
        HostCentricDistance_norm = group['HostCentricDistance_norm'][:]

        dmin_phys = np.zeros(HostCentricDistance_phys.size, dtype=HostCentricDistance_phys.dtype)
        dmin_norm = np.zeros(HostCentricDistance_norm.size, dtype=HostCentricDistance_norm.dtype)

        for i, _ in enumerate(HostCentricDistance_phys):
            indices = group['SubfindID'][i:] != -1
            if len(indices[indices]) == 0:
                continue
            
            dmin_phys[i] = np.min(HostCentricDistance_phys[i:][indices])
            dmin_norm[i] = np.min(HostCentricDistance_norm[i:][indices])

        dsets = [dmin_norm, dmin_phys]
        for dset_index, dset_key in enumerate(keys):
            dset = dsets[dset_index]
            dataset = group.require_dataset(dset_key, shape=dset.shape, dtype=dset.dtype)
            dataset[:] = dset

            
    f.close()

    return
        
    
def add_Nperipass(Config, mindist_phys=1000.0, mindist_norm=2.0):
    """
    add flags for pericenter passages.
    """
    
    f = h5py.File(Config.outdirec+Config.outfname, 'a')
    keys = ['Nperipass', 'min_Dperi_norm', 'min_Dperi_phys', 'Napopass']
    
    for group_key in f.keys():
        group = f[group_key]

        SubfindID = group['SubfindID']
        
        HostCentricDistance_phys = group['HostCentricDistance_phys'][:]    
        min_indices_phys = argrelextrema(HostCentricDistance_phys, np.less)[0]
    
        HostCentricDistance_norm = group['HostCentricDistance_norm'][:]
    
        indices = np.where((HostCentricDistance_phys[min_indices_phys] < mindist_phys) 
                           & (HostCentricDistance_norm[min_indices_phys] < mindist_norm)
                           & (SubfindID != -1))[0]
    
        peri_indices = min_indices_phys[indices]
        Dperi_phys = HostCentricDistance_phys[peri_indices]
        Dperi_norm = HostCentricDistance_norm[peri_indices]
    
        Nperipass = np.zeros(HostCentricDistance_phys.size, dtype=int)
        Dperi_norm_min = np.zeros(Nperipass.size, dtype=HostCentricDistance_norm.dtype)
        Dperi_phys_min = Dperi_norm_min.copy()
        Napopass = Nperipass.copy()
        for i, peri_index in enumerate(peri_indices):
            Nperipass[:peri_index+1] += 1

            # add the minimum pericentric distance the galaxy has had
            if i == 0:
                Dperi_norm_min[:peri_index+1] = np.min(Dperi_norm[i:])
                Dperi_phys_min[:peri_index+1] = np.min(Dperi_phys[i:])
                
                apo_start_index = 0

            else:
                Dperi_norm_min[peri_indices[i-1]+1:peri_index+1] = np.min(Dperi_norm[i:])
                Dperi_phys_min[peri_indices[i-1]+1:peri_index+1] = np.min(Dperi_phys[i:])
                
                apo_start_index = peri_indices[i-1]
                          
            max_index_phys = np.argmax(HostCentricDistance_phys[apo_start_index:peri_index])
            Napopass[:apo_start_index+max_index_phys+1] += 1                    
                
        dsets = [Nperipass, Dperi_norm_min, Dperi_phys_min, Napopass]

        for dset_index, dset_key in enumerate(keys):
            dset = dsets[dset_index]
            dataset = group.require_dataset(dset_key, shape=dset.shape, dtype=dset.dtype)
            dataset[:] = dset

        
    f.close()

    return 


def add_LBEramPressure(Config):
    """
    For TNGCluster satellites, compute the the LBE ram pressure acting on them.
    """
    
    f = h5py.File(Config.outdirec + Config.outfname, 'a')
    keys = np.array(list(f.keys()))

    LBE_direc = '/vera/ptmp/gc/mayromlou/public/LBE/TNG-Cluster1000/snap99/'

    dset_key = 'LBEramPressure'

    # initialize important arrays
    LBE_RP = np.zeros(keys.size, dtype=float) - 1.0
    HostZoomBoxNumbers = np.zeros(keys.size, dtype=int) - 1
    SubhaloZoomBoxIndices = HostZoomBoxNumbers.copy()

    # find the primary zoom targets and their HaloIDs
    halos = il.groupcat.loadHalos(Config.basePath, 99)
    PrimaryZoomTargets = halos['GroupPrimaryZoomTarget'] == 1
    HaloIDs = np.where(PrimaryZoomTargets)[0]

    # for now, only care about z=0
    time_index = 0

    # loop over each group, computing the subhalo index into the original zoom box, 
    # and the original zoom box number
    for subhalo_i, key in enumerate(keys):
        group = f[key]
        subfindID = int(group['SubfindID'][time_index])
        groupfirstsub = int(group['SubhaloGroupFirstSub'][time_index])
        subhalogrnr = int(group['SubhaloGrNr'][time_index])

        SubhaloZoomBoxIndices[subhalo_i] = subfindID - groupfirstsub
        HostZoomBoxNumbers[subhalo_i] = np.where(subhalogrnr == HaloIDs)[0][0]

    assert SubhaloZoomBoxIndices.min() > 0, "Not all SubhaloZoomBoxIndices properly assigned."
    assert HostZoomBoxNumbers.min() == 0, "Not all HostZoomBoxNumbers properly assigned."

    # now loop over the boxes and store the LBE_RP value
    subhalo_count_tot = 0

    for box_number, haloID in enumerate(HaloIDs):
        LBE_fname = 'LBE_TNG-Cluster1000_snap099_%d.hdf5'%box_number

        LBE_file = h5py.File(LBE_direc + LBE_fname, 'r')
        Subhalo = LBE_file['Subhalo']

        zoom_indices = HostZoomBoxNumbers == box_number
        subhalo_indices = SubhaloZoomBoxIndices[zoom_indices]

        Nsubhalos = zoom_indices[zoom_indices].size

        LBE_RP[subhalo_count_tot:subhalo_count_tot+Nsubhalos] = Subhalo['LBEramPressure'][subhalo_indices]

        subhalo_count_tot += Nsubhalos

        LBE_file.close()

    assert LBE_RP.min() >= 0, "Not all LBE_RP values assigned properly."

    # loop back over the original groups and save the data
    for key_i, key in enumerate(keys):
        group = f[key]
        dset = np.zeros(group['SnapNum'].size, dtype=LBE_RP.dtype) - 1
        dset[time_index] = LBE_RP[key_i]
        dataset = group.require_dataset(dset_key, shape=dset.shape, dtype=dset.dtype)
        dataset[:] = dset

    f.close()

    return 


def add_Lxmaps(Config):
    """
    For TNGCluster centrals, add the soft x-ray maps in 3 projections.
    The x-ray luminosities are computed using the APEC lookup tables, 
    following the methods of Truong+21, Nelson+23. In addition to the 
    three projections of the maps, we also save smoothed versions of the maps.
    """

    f = h5py.File(Config.outdirec+Config.outfname, 'a')
    f_keys = np.array(list(f.keys()))
    
    # load basic simulation parameters
    basePath = Config.basePath
    z0_index = 0
    snapNum = Config.SnapNums[z0_index]
    gas_ptn = Config.gas_ptn

    flag_4k = False

    Lx_key = 'xray_lum_0.5-2.0kev'

    gas_fields = ['Masses', 'Coordinates', 'Density']

    pixel_size = 5. # kpc
    smoothing_scale = 300. / pixel_size # keep constant at 300 kpc

    # load the soft x-ray luminosities
    Lx_direc = '/vera/ptmp/gc/dnelson/sims.TNG/L680n8192TNG/data.files/cache/'
    Lx_fname = 'cached_gas_xray_lum_0.5-2.0kev_99.hdf5'
    with h5py.File(Lx_direc + Lx_fname, 'r') as Lx_f:
        Lx_all = Lx_f[Lx_key][:]

    for key in f_keys:

        print('add_Lxmap: Working on %s.'%key)
        group = f[key]

        haloID = group['HostSubhaloGrNr'][z0_index]

        # load all gas cells,
        total_gas_cells = il.snapshot.loadOriginalZoom(basePath, snapNum, haloID, gas_ptn, fields=gas_fields)
        total_gas_cells[Lx_key] = np.zeros(total_gas_cells['count'], dtype=float) - 1.

        ### find the relevant indices to load the Lx dataset
        # load fuzz length, compute offset, call loadSubset                                                                     
        subset = il.snapshot.getSnapOffsets(basePath, snapNum, haloID, "Group")

        # identify original halo ID and corresponding index
        halo = il.snapshot.loadSingle(basePath, snapNum, haloID=haloID)
        assert 'GroupOrigHaloID' in halo, 'Error: loadOriginalZoom() only for the TNG-Cluster simulation.'
        orig_index = np.where(subset['HaloIDs'] == halo['GroupOrigHaloID'])[0][0]

        # (1) load all FoF particles/cells
        length_FoF = subset['GroupsTotalLengthByType'][orig_index, gas_ptn]
        start_FoF = subset['GroupsSnapOffsetByType'][orig_index, gas_ptn]
        total_gas_cells[Lx_key][:length_FoF] = Lx_all[start_FoF:start_FoF+length_FoF]

        # (2) load all non-FoF particles/cells
        length_fuzz = subset['OuterFuzzTotalLengthByType'][orig_index, gas_ptn]
        start_fuzz = subset['OuterFuzzSnapOffsetByType'][orig_index, gas_ptn]
        total_gas_cells[Lx_key][length_FoF:] = Lx_all[start_fuzz:start_fuzz+length_fuzz]

        assert total_gas_cells[Lx_key].min() >= 0, 'Error: not all Lx array indices were set.'

        subset_GroupFirstSub = il.snapshot.getSnapOffsets(basePath, snapNum, halo['GroupFirstSub'], "Subhalo")

        # find subset only belonging to the BCG
        onlyBCG_gas_cells = {}
        for key in total_gas_cells.keys():
            if key == 'count':
                onlyBCG_gas_cells[key] = subset_GroupFirstSub['lenType'][gas_ptn]
            else:
                onlyBCG_gas_cells[key] = total_gas_cells[key][:subset_GroupFirstSub['lenType'][gas_ptn]]

        # find the subset belonging to all subhalos and other FoFs, but not the BCG or fuzz of the BCG
        # find the last gas cell belonging to the last subhalo
        halo_lastsubhalo = halo['GroupFirstSub'] + halo['GroupNsubs']
        subset_lastsubhalo = il.snapshot.getSnapOffsets(basePath, snapNum, halo_lastsubhalo, "Subhalo")

        last_gascell_index = subset_lastsubhalo['offsetType'][gas_ptn] + subset_lastsubhalo['lenType'][gas_ptn]
        length_boundcells = last_gascell_index - start_FoF
        length_boundcells_noBCG = length_boundcells - subset_GroupFirstSub['lenType'][gas_ptn]

        # to include other FoFs that may be in the FoV, we need the end of the given FoF's gas cells
        length_FoF_gas = halo['GroupLenType'][gas_ptn]
        length_gascells = length_boundcells_noBCG + (length_FoF - length_FoF_gas)

        noBCG_nofuzz_gas_cells = {}
        for key in total_gas_cells.keys():
            if key == 'count':
                noBCG_nofuzz_gas_cells[key] = length_gascells
            else:
                noBCG_nofuzz_gas_cells[key] = np.concatenate((total_gas_cells[key][subset_GroupFirstSub['lenType'][gas_ptn]:length_boundcells],
                                                            total_gas_cells[key][length_FoF_gas:length_FoF]))    

        # find the subset belonging to the outer fuzz 
        onlyFuzz_gas_cells = {}
        for key in total_gas_cells.keys():
            if key == 'count':
                onlyFuzz_gas_cells[key] = length_fuzz
            else:
                onlyFuzz_gas_cells[key] = total_gas_cells[key][length_FoF:]

        R200c = group['HostGroup_R_Crit200'][z0_index]
        boxSizeImg = [3.*R200c, 3.*R200c]

        axes_list = [[0,1],
                     [0,2],
                     [1,2]]

        labels_list = ['xy',
                       'xz',
                       'yz']

        for axes_i, axes in enumerate(axes_list):

            if axes_i != 0:
                continue

            label = labels_list[axes_i]

            Lx_map_key = Lx_key + '_' + label

            if flag_4k:
                nPixels = 4096
                pixel_size = boxSizeImg[0] / nPixels
                smoothing_scale = 300. / pixel_size
                Lx_map_key += '_4k'

            Lx_map_smooth_key = Lx_map_key + '_smooth'

            Lx_map = return_Lx_map(Config, total_gas_cells, halo, axes, in4k=flag_4k)
            Lx_map_smooth = gaussian_filter(Lx_map, smoothing_scale, mode='constant')
                        
            Lx_map_key_onlyBCG = Lx_map_key + '_onlyBCG'
            Lx_map_key_noBCG_nofuzz =  Lx_map_key + '_noBCG_nofuzz'
            Lx_map_key_onlyFuzz = Lx_map_key + '_onlyFuzz'

            Lx_map_onlyBCG = return_Lx_map(Config, onlyBCG_gas_cells, halo, axes, in4k=flag_4k)
            Lx_map_noBCG_nofuzz = return_Lx_map(Config, noBCG_nofuzz_gas_cells, halo, axes, in4k=flag_4k)
            Lx_map_onlyFuzz = return_Lx_map(Config, onlyFuzz_gas_cells, halo, axes, in4k=flag_4k)

            dsets = [Lx_map, Lx_map_smooth, Lx_map_onlyBCG, Lx_map_noBCG_nofuzz, Lx_map_onlyFuzz]
            dset_keys = [Lx_map_key, Lx_map_smooth_key, Lx_map_key_onlyBCG, Lx_map_key_noBCG_nofuzz, Lx_map_key_onlyFuzz]
            for dset_index, dset_key in enumerate(dset_keys):
                dset = dsets[dset_index]
                dataset = group.require_dataset(dset_key, shape=dset.shape, dtype=dset.dtype)
                dataset[:] = dset
            

    # finish loop over keys
    f.close()

    return 

def return_Lx_map(Config, dic, halo, axes, in4k=True):
    """
    given the dictionary of masses, coordinates, densities, and Lx_soft,
    and the halo, which is also a dictionary contatining at least R200c and halo_pos,
    create the map and return.
    """
    a = Config.Times[0]
    boxsize = Config.BoxSizes[0]
    h = Config.h

    gas_hsml_fact = 1.5

    Lx_key = 'xray_lum_0.5-2.0kev'

    Masses = dic['Masses'] * 1.0e10 / h
    Coordinates = dic['Coordinates'] * a / h
    Densities = dic['Density'] * 1.0e10 / h / (a / h)**3
    Sizes = (Masses / (Densities * 4./3. * np.pi))**(1./3.) * gas_hsml_fact
    Lxsoft = dic[Lx_key]

    R200c = halo['Group_R_Crit200'] * a / h
    halo_pos = halo['GroupPos'] * a / h

    pos = Coordinates[:,axes]
    hsml = Sizes
    mass = Lxsoft
    quant = None
    boxSizeImg = [3.*R200c, 3.*R200c] # kpc
    boxSizeSim = [boxsize, boxsize, boxsize]
    boxCen = halo_pos[axes]    
    ndims = 3

    nPixels = 4096
    if not in4k:
        pixel_size = 5.
        nPixels = int(boxSizeImg[0] / pixel_size)
    
    Lx_map_4k = sphMap.sphMap(pos, hsml, mass, quant, [0,1], boxSizeImg, boxSizeSim, boxCen, [nPixels, nPixels], ndims, colDens=True)

    return Lx_map_4k



def add_quenchtimes(Config):
    """
    Use the quenching and stellar assembly times catalog to add the
    quenching time (if the subhalo is quenched) for each subhalo. 
    """

    f = h5py.File(Config.outdirec + Config.outfname, 'a')
    keys = np.array(list(f.keys()))
    
    # load the quenching catalog
    quench_direc = '../IllustrisTNG/%s/postprocessing/QuenchingStellarAssemblyTimes/'%Config.sim
    with h5py.File(quench_direc + 'quenching_099.hdf5', 'r') as quench_cat:
        flag = quench_cat['Subhalo']['flag'][:]
        quenching_snap = quench_cat['Subhalo']['quenching_snap'][:]

        quench_cat.close()

    for i, key in enumerate(keys):
        quench_snap = np.array([-1], dtype=int)
        group = f[key]
        indices = np.where(group['SubfindID'][:] != -1)[0]
        
        SnapNum = group['SnapNum'][indices]
        if np.max(SnapNum) == 99:
            subfindID = group['SubfindID'][0]
            if flag[subfindID] == 1:
                quench_snap = np.array([quenching_snap[subfindID]], dtype=int)

        # save the quench snap
        dataset = group.require_dataset('quenching_snap', shape=quench_snap.shape,
                                       dtype=quench_snap.dtype)
        dataset[:] = quench_snap

    # finish loop over keys
    f.close()

    return

 
def add_gastau(Config):
    """
    add the peak cold gas mass time and the respective tau
    """
    
    def return_tau(peak_index, dset):
        """
        Calculate and return tau given the peak_index and the dset
        """
        tau = np.zeros(dset.size, dtype=dset.dtype) - 1.

        peak = dset[peak_index]
        tau[peak_index] = 0.

        # if a subhalo is its own LI host, then peak_index occurs at the first snapshot,
        # and its cold gas mass may be zero -- if so, then return a list of -1.
        # or if M_coldgas(infall) = 0, then return 0 at peak_index and -1 elsewhere.
        if peak == 0:
            return tau

        # if the peak is at the latest (first) snapshot, then don't calculate tau
        if peak_index == 0:
            return tau
        
        tau[:peak_index+1] = (peak - dset[:peak_index+1]) / peak * 100.
            
        return tau

    f = h5py.File(Config.outdirec+Config.outfname, 'a')
    
    N_RM = 3 # the number of snapshots to average over for running median
             # should be an odd number
    Nsnaps_PP = Config.Nsnaps_PP
    M200c0_lolim_PP = Config.M200c0_lolim_PP

    keys = ['tau_medpeak', 'tau_infall']
    gastypes = ['ColdGas', 'HotGas', 'Gas', '']

    f_keys = list(f.keys())

    for group_index, group_key in enumerate(f_keys):
        group = f[group_key]

        result = np.zeros(len(group['SnapNum'][:]), dtype=float) - 1.
        
        # find the indices that the subbhalo was identified at
        subhalo_indices = np.where(group['SubfindID'][:] != -1)[0]
        
        # first infall time
        infall_tau = result.copy()
        infall_index = np.max(np.argwhere(group['memberlifof_flags'][:] == 1))
        infall_tau_index = np.where(group['SnapNum'][subhalo_indices] == group['SnapNum'][infall_index])[0][0]

        # for TNG-Cluster galaxies, allow for infall time to be into another host
        if Config.TNGCluster_flag:
            # find infall time as the first time subhalo was a satellite for Nsnaps_PP consecutive snapshots
            satellite_indices = ((group['central_flags'][subhalo_indices] == 0) &
                                 (group['HostGroup_M_Crit200'][subhalo_indices] >= M200c0_lolim_PP))

            satellite_check = [True] * Nsnaps_PP
            satellite_indices_bool = ru.where_is_slice_in_list(satellite_check, satellite_indices)

            if any(satellite_indices_bool):
                # from this first time that the subhalo was a satellite, find the index of
                # the first conescutive snapshot.
                infall_tau_index = np.where(satellite_indices_bool)[0].max()
            else:
                infall_tau_index = 0

        for gastype in gastypes:
        
            gas_dset = group['Subhalo%sMass'%gastype][subhalo_indices]
            
            # infall time
            infall_tau = result.copy()
            if infall_tau_index >= gas_dset.size:
                print(infall_tau_index, gas_dset.size, gastype, group_index, group_key)
            infall_tau[subhalo_indices] = return_tau(infall_tau_index, gas_dset)

            # running median maximum of the cold gas mass
            # ensure that there are enough snaps to calc the running median
            # galaxies that do not reach z=0 will be ignored later anyways
            medpeak_tau = result.copy()
            if len(gas_dset) >=  N_RM:
                med_SGM = ru.RunningMedian(gas_dset, N_RM)
                # choose the median x value corresponding to the max, as this is the peak
                medpeak_SGM_index = int(np.median(np.argwhere(med_SGM == max(med_SGM)).T))
                #medpeak_SCGM_index = np.min(np.argwhere(med_SCGM == max(med_SCGM))) + int((N_RM - 1) / 2)
                medpeak_tau[subhalo_indices] = return_tau(medpeak_SGM_index, gas_dset)

            dsets = [medpeak_tau, infall_tau]
            if gastype == '':
                gastype = 'Mass'
            outkeys = [key +'_'+ gastype for key in keys]
            for dset_index, dset_key in enumerate(outkeys):
                dset = dsets[dset_index]
                dataset = group.require_dataset(dset_key, shape=dset.shape, dtype=dset.dtype)
                dataset[:] = dset
            
    # finish loop over branches
    
    f.close()
        
    return


def add_tracers(Config):
    """
    Add tracer particle post-processing datasets to the GRP catalog.
    """
    
    max_snap = Config.max_snap
    min_snap = Config.min_snap
    snaps = Config.SnapNums
    CosmicTimes = Config.CosmicTimes
    h = Config.h
    basePath = Config.basePath
    tracer_outdirec = Config.tracer_outdirec
    tracer_ptn = Config.tracer_ptn
    gas_ptn = Config.gas_ptn
    tlim = Config.tlim
    star_ptn = Config.star_ptn
    bh_ptn = Config.bh_ptn

    f = h5py.File(Config.outdirec + Config.outfname, 'a')
    keys = np.array(list(f.keys()))
    NsubfindIDs = keys.size

    group = f[keys[0]]

    header      = ru.loadHeader(basePath, max_snap)
    tracer_mass = header['MassTable'][tracer_ptn] * 1.0e10 / h
    
    # initialize results
    SubhaloColdGasTracer_Mass      = np.zeros((NsubfindIDs, snaps.size), dtype=float) - -1.
    SubhaloColdGasTracer_new       = SubhaloColdGasTracer_Mass.copy()
    SubhaloColdGasTracer_out       = SubhaloColdGasTracer_Mass.copy()
    SubhaloColdGasTracer_StripTot  = SubhaloColdGasTracer_Mass.copy()
    SubhaloColdGasTracer_StripCold = SubhaloColdGasTracer_Mass.copy()
    SubhaloColdGasTracer_Heat      = SubhaloColdGasTracer_Mass.copy()
    SubhaloColdGasTracer_Star      = SubhaloColdGasTracer_Mass.copy()
    SubhaloColdGasTracer_BH        = SubhaloColdGasTracer_Mass.copy()

    # tabulate all GRP subfindIDs at each snap
    all_subfindIDs = np.zeros((snaps.size, keys.size), dtype=int) - 1
    for key_i, key in enumerate(keys):
        all_subfindIDs[:,key_i] = f[key]['SubfindID'][:]
    
    for snap_i, snap in enumerate(snaps):
        
        print(snap_i, snap)
        
        offsets = h5py.File(tracer_outdirec + 'offsets_%03d.hdf5'%snap, 'r')
        tracers = h5py.File(tracer_outdirec + 'tracers_%03d.hdf5'%snap, 'r')

        offsets_group = offsets['group']
        tracers_group = tracers['group']

        # find the overlapping indices between the tracers and the subfind_GRP catalogs
        tracers_indices, GRP_indices = ru.match3(offsets_group['SubfindID'][:], all_subfindIDs[snap_i])
        
        if tracers_indices.size == 0:
            continue
        
        SubhaloColdGasTracer_Mass[GRP_indices,snap_i] = offsets_group['SubhaloLengthColdGas'][:][tracers_indices] * tracer_mass
        SubhaloColdGasTracer_new[GRP_indices,snap_i] = offsets_group['SubhaloLengthColdGas_new'][:][tracers_indices] * tracer_mass
        SubhaloColdGasTracer_out[GRP_indices,snap_i] = ((offsets_group['SubhaloLength'][:][tracers_indices] - offsets_group['SubhaloLengthColdGas'][:][tracers_indices])
                                                      * tracer_mass)

        # for every snap except min_snap (the last one), calculate the tracer derivatives
        if snap_i != min_snap:

            # loop over each subhalo at this snapshot to split the out sample into the various components
            for subfind_i, subfindID in enumerate(offsets_group['SubfindID'][:][tracers_indices]):

                # if the subhalo is not identified at this snap, continue
                if subfindID == -1:
                    continue

                tracer_index = tracers_indices[subfind_i]
                GRP_index = GRP_indices[subfind_i]
                                                  
                start = offsets_group['SubhaloOffset'][tracer_index] + offsets_group['SubhaloLengthColdGas'][tracer_index]
                end   = offsets_group['SubhaloOffset'][tracer_index] + offsets_group['SubhaloLength'][tracer_index]

                ParentPartType = tracers_group['ParentPartType'][start:end]

                # first, gas particles: could be stripping + outflows or heating
                gas_indices = ParentPartType == gas_ptn
                still_bound = tracers_group['StillBound_flag'][start:end][gas_indices]
                strip_indices = still_bound == 0
                
                ParentGasTemp = tracers_group['ParentGasTemp'][start:end][gas_indices]
                cold_indices = ParentGasTemp <= tlim
                
                # total stripping + outflows = number of unbound gas cells
                # cold stripping + outflows = number of unbound cold gas cells
                # gas heating = number of bound gas cells that are heated == number of still bound gas cells
                # include a check that these three groups make up all gas parents
                
                Ntot = np.where(strip_indices)[0].size
                Ncold = np.where(strip_indices & cold_indices)[0].size
                Nheat = np.where(~cold_indices & ~strip_indices)[0].size
                Nheat_check = np.where(~strip_indices)[0].size
                if Nheat != Nheat_check:
                    print('Warning for bound heated gas cells for %s %s subfindID %s'%(Config.sim, snap, subfindID))
                
                SubhaloColdGasTracer_StripTot[GRP_index,snap_i] = Ntot * tracer_mass
                SubhaloColdGasTracer_StripCold[GRP_index,snap_i] = Ncold * tracer_mass
                SubhaloColdGasTracer_Heat[GRP_index,snap_i] = Nheat * tracer_mass
                
                # second, star paticles: treating winds + stars identically here
                star_indices = ParentPartType == star_ptn
                SubhaloColdGasTracer_Star[GRP_index,snap_i] = star_indices[star_indices].size * tracer_mass

                # lastly, black holes
                bh_indices = ParentPartType == bh_ptn
                SubhaloColdGasTracer_BH[GRP_index,snap_i] = bh_indices[bh_indices].size * tracer_mass

            # finish loop over subhalos at the given snapshot

        offsets.close()
        tracers.close()

    # finish loop over snapshots
    
    # for each subhalo, calculate the time between snapshots that the subhalo exists in the merger trees
    # then divide the tracer particles by this time such that the units are Msun / yr
    dsets = [SubhaloColdGasTracer_StripTot,
             SubhaloColdGasTracer_StripCold, 
             SubhaloColdGasTracer_Heat,
             SubhaloColdGasTracer_Star,
             SubhaloColdGasTracer_BH]

    for key_i, key in enumerate(keys):
        indices = np.where(f[keys[key_i]]['SubfindID'][:] != -1)[0]
        times = CosmicTimes[indices]
        time_diffs = (times[:-1] - times[1:]) * 1.0e9

        for dset in dsets:
            dset[key_i,indices[:-1]] /= time_diffs

    # save new datasets
    dset_keys = ['SubhaloColdGasTracer_Mass',
                 'SubhaloColdGasTracer_new',
                 'SubhaloColdGasTracer_out',
                 'SubhaloColdGasTracer_StripTot',
                 'SubhaloColdGasTracer_StripCold',
                 'SubhaloColdGasTracer_Heat',
                 'SubhaloColdGasTracer_Star',
                 'SubhaloColdGasTracer_BH']

    dsets     = [SubhaloColdGasTracer_Mass,
                 SubhaloColdGasTracer_new,
                 SubhaloColdGasTracer_out,
                 SubhaloColdGasTracer_StripTot,
                 SubhaloColdGasTracer_StripCold,
                 SubhaloColdGasTracer_Heat,
                 SubhaloColdGasTracer_Star,
                 SubhaloColdGasTracer_BH]

    for key_i, key in enumerate(keys):
        group = f[key]
        for dset_key_i, dset_key in enumerate(dset_keys):
            dset = dsets[dset_key_i][key_i]
            dataset = group.require_dataset(dset_key, shape=dset.shape, dtype=dset.dtype)
            dataset[:] = dset
            
    f.close()

    return


def add_tracers_postprocessing(Config):
    """
    post process the tracer quantities
    """
    
    f = h5py.File(Config.outdirec+Config.outfname, 'a')
    f_keys = np.array(list(f.keys()))
    
    result = np.zeros(Config.SnapNums.size, dtype=float) - -1.

    keys = ['RPS_int_tot',
            'SFR_int_tot',
            'sRPS',
            'sSFR']

    tracer_key = 'SubhaloColdGasTracer_Mass'
    RPS_key = 'SubhaloColdGasTracer_StripTot'
    SFR_key = 'SubhaloSFR'
    SCGM_key = 'SubhaloColdGasMass'


    for group_i, group_key in enumerate(f_keys):
        group = f[group_key]

        RPS_int_tot = result.copy()
        SFR_int_tot = result.copy()
        sRPS = result.copy()
        sSFR = result.copy()
        
        subhalo_indices = np.where(group['SubfindID'][:] != -1)[0]
                
        SCGM_flag = False
        
        if subhalo_indices.size > 1:
                    
            infall_index = np.where(group['memberlifof_flags'][subhalo_indices] == 1)[0].max() + 1

            if subhalo_indices.size > 1:
                SCGM = group[SCGM_key][subhalo_indices]
                if 0 in SCGM:
                    start_index = np.where(SCGM[:infall_index] == 0)[0]
                    if start_index.size > 0:
                        start_index = start_index.max()
                        SCGM_flag = True
                        subhalo_indices = subhalo_indices[start_index:]
                        SCGM = SCGM[start_index:]
                
                CosmicTimes = group['CosmicTime'][subhalo_indices]
                RPS = group[RPS_key][subhalo_indices]
                SFR = group[SFR_key][subhalo_indices]
                SCGM = group[SCGM_key][subhalo_indices]
                time_diffs = (CosmicTimes[:-1] - CosmicTimes[1:]) * 1.0e9

                save_indices = subhalo_indices[:-1]
                RPS_int_tot[save_indices] = np.cumsum((RPS[:-1] * time_diffs)[::-1])[::-1]
                SFR_int_tot[save_indices] = np.cumsum((SFR[:-1] * time_diffs)[::-1])[::-1]

                # calculate the specific RPS + outflows, SFR rate over all time
                calc_indices = (RPS >= 0) & (SCGM > 0)
                if calc_indices[calc_indices].size > 0:
                    save_indices = subhalo_indices[calc_indices]
                    sRPS[save_indices] = RPS[calc_indices] / SCGM[calc_indices]

                calc_indices = (SFR >= 0) & (SCGM > 0)
                if calc_indices[calc_indices].size > 0:
                    save_indices = subhalo_indices[calc_indices]
                    sSFR[save_indices] = SFR[calc_indices] / SCGM[calc_indices]
                    
            if (SCGM_flag):
                subhalo_indices = np.where(group['SubfindID'][:] != -1)[0]
                RPS_int_tot[subhalo_indices[:start_index+1]] = RPS_int_tot[subhalo_indices[start_index]]
                SFR_int_tot[subhalo_indices[:start_index+1]] = SFR_int_tot[subhalo_indices[start_index]]
                sRPS[subhalo_indices[:start_index+1]] = 0
                sSFR[subhalo_indices[:start_index+1]] = 0
            

        dsets = [RPS_int_tot, SFR_int_tot, sRPS, sSFR]
        for dset_index, dset_key in enumerate(keys):
            dset = dsets[dset_index]
            dataset = group.require_dataset(dset_key, shape=dset.shape, dtype=dset.dtype)
            dataset[:] = dset
            
    f.close()
    
    return


def add_coldgasmasstracerstau(Config):
    """
    add tau clock definitions based on the tracer quantities.
    must be called after adding time and tracer datasets.
    There are two tau definitions using the tracers:
    (i) tau_RPS_tot_infall
    starting one snapshot before infall,
    integrate RPS + outflows until z_x. tau_x is when the integral
    reaches x% of the integral until z=0.
    (ii) tau_RPS_est_infall
    calcualte the mass loading factor median(RPS + outflows / SFR)
    when the galaxy was a central, and use this to estimate the 
    outflows when the galaxy is a satellite. Then the difference
    between the estimated outflows and the measured RPS + outflows
    is the RPS. integrate this from infall until z=0 and use the
    same tau_x clock as definition (i).
    (iii) tau_RPS_sRPS
    find the peak in the (RPS + outflows) / M_ColdGas after infall,
    and then define tau_0 as when RPS + outflows = the average 
    (RPS + outflows) before infall, and tau_100 as the first of:
    a) (RPS + outflows) / M_ColdGas < 1 / t_H, t_H = Hubble Time
    b) M_ColdGas reaches 0;
    c) z=0
    """
    
    f = h5py.File(Config.outdirec+Config.outfname, 'a')
    f_keys = np.array(list(f.keys()))

    result = np.zeros(Config.SnapNums.size, dtype=float) - -1.

    keys = ['tau_RPS_tot',
            'tau_RPS_est',
            'tau_RPS_sRPS']

    # define the number of points to use for smoothing via running median
    N_RM = 7
    
    tracer_key = 'SubhaloColdGasTracer_Mass'
    RPS_key = 'SubhaloColdGasTracer_StripTot'
    SFR_key = 'SubhaloSFR'
    SCGM_key = 'SubhaloColdGasMass'
    
    RPS_int_tot_key = 'RPS_int_tot'
    SFR_int_tot_key = 'SFR_int_tot'
    sRPS_key = 'sRPS'
    sSFR_key = 'sSFR'

    for group_index, group_key in enumerate(f_keys):
        group = f[group_key]

        # initalize results
        tau_RPS_tot = result.copy()
        tau_RPS_est = result.copy()
        tau_RPS_sRPS = result.copy()

        # start with the last snapshot that the galaxy was a central 
        # this is the same as one snapshot before infall 
        subhalo_indices = np.where(group['SubfindID'][:] != -1)[0]

        infall_index = np.where(group['memberlifof_flags'][subhalo_indices] == 1)[0].max() + 1
        
        calc_indices = subhalo_indices[:infall_index]
        
        ### tau_RPS_tot
        # check that there is a well defined infall time and some snaps afterwards 
        if (infall_index < subhalo_indices.size) & (calc_indices.size > 1):
            # now let's start the clock at the infall
            RPS_int_tot = group[RPS_int_tot_key][calc_indices]
            tau_RPS_tot[calc_indices] = RPS_int_tot / RPS_int_tot[0] * 100.
            tau_RPS_tot[subhalo_indices[infall_index]] = 0

        ### tau_RPS_est
        # check that there are at least N_RM snaps before infall to calc eta
        # note that the there is no calculated RPS + outflows for the last snpshot
        if subhalo_indices[infall_index:-1].size > N_RM:
            SFR = group[SFR_key][subhalo_indices][infall_index:-1]
            out = group[RPS_key][subhalo_indices][infall_index:-1]
            calc_indices = (SFR > 0) & (out >= 0)

            if calc_indices[calc_indices].size > 0:
                eta = np.median(out[calc_indices] / SFR[calc_indices])
                out_est = eta * group[SFR_key][subhalo_indices]
                RPS = group[RPS_key][subhalo_indices]
                                
                RPS_RM = ru.RunningMedian(RPS, N_RM)
                out_est_RM = ru.RunningMedian(out_est, N_RM)

                RPS_RM_peakindex = (RPS_RM[:infall_index] - out_est_RM[:infall_index]).argmax()
                diff = RPS_RM[RPS_RM_peakindex:] - out_est_RM[RPS_RM_peakindex:]
                tau0 = np.where(diff < 0)[0].min()
                tau0_index = subhalo_indices[RPS_RM_peakindex:][tau0]
                
                # ensure tau0 is before z=0
                if tau0 > 0:
                    SFR_int_tot = group[SFR_int_tot_key][subhalo_indices]
                    SFR_int_tot -= SFR_int_tot[tau0_index]
                    SFR_int_tot[tau0_index:] = 0.
                    RPS_est_cumsum = group[RPS_int_tot_key][subhalo_indices]
                    RPS_est_cumsum -= RPS_est_cumsum[tau0_index]
                    RPS_est_cumsum[tau0_index:] = 0
                    RPS_est_cumsum[:tau0_index] -= eta * SFR_int_tot[:tau0_index]
                    RPS_est_cumsum[RPS_est_cumsum < 0] = 0

                    tau_RPS_est[subhalo_indices[:tau0_index]] = (RPS_est_cumsum[:tau0_index] / RPS_est_cumsum[:tau0_index][0]) * 100.
                    tau_RPS_est[subhalo_indices[tau0_index]] = 0.

                    tau0_index_RPS_est = tau0_index

        ### tau_sRPS
        # check that there are at least N_RM snaps before infall to calc avg specific RPS + outflows
        if subhalo_indices[infall_index:-1].size > N_RM:
            sRPS = group[sRPS_key][subhalo_indices]
            calc_indices = np.where(sRPS[infall_index:-1] >= 0)[0]
            
            if calc_indices.size > 0:
                avg_sRPS = np.median(sRPS[infall_index:-1][calc_indices])
                
                calc_indices = sRPS >= 0
                sRPS_RM = ru.RunningMedian(sRPS[calc_indices], N_RM)
                
                sRPS_RM_peakindex = sRPS_RM[:infall_index].argmax()

                diff = sRPS_RM[sRPS_RM_peakindex:] - avg_sRPS
                
                
                if diff[diff <= 0].size > 0:
                
                    tau0 = np.where(diff <= 0)[0].min()
                    tau0_index = subhalo_indices[calc_indices][sRPS_RM_peakindex:][tau0]

                    SCGM = group[SCGM_key][subhalo_indices]

                    if 0 in SCGM[calc_indices][:sRPS_RM_peakindex]:
                        tau100_index = subhalo_indices[np.where(SCGM == 0)[0].max()]
                    else:
                        tau100_index = subhalo_indices[0]

                    tau_RPS_sRPS[:tau100_index+1] = 100.
                    tau_RPS_sRPS[tau100_index+1:tau0_index] = 50.
                    tau_RPS_sRPS[tau0_index] = 0.

                    tau0_index_RPS_sRPS = tau0_index
            
        # save the output
        dsets = [tau_RPS_tot,
                 tau_RPS_est,
                 tau_RPS_sRPS]
        
        for dset_index, dset_key in enumerate(keys):
            dset = dsets[dset_index]
            dataset = group.require_dataset(dset_key, shape=dset.shape, dtype=dset.dtype)
            dataset[:] = dset        
    
    # finish loop over branches
    f.close()

    return
                     
                    
