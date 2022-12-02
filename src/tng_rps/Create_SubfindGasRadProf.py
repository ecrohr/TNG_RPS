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
from importlib import reload

global sim, basePath, fname, direc
global tlim, radii_binwidth, key
global scalar_keys, dset_keys
global rmin_norm, rmax_norm, radii_bins_norm, radii_bincents_norm, nbins
global tracer_ptn, star_ptn, gas_ptn, bh_ptn, bary_ptns
global jellyscore_min

tlim = 10.**(4.5) # K; cutoff between cold and hot CGM gas
radii_binwidth = 0.1
key = 'inspected'

scalar_keys = ['SubhaloColdGasMass', 'SubhaloGasMass', 'SubhaloHotGasMass']
dset_keys = ['radii', 'mass_shells', 'vol_shells', 'densities_shells']

### define radial bins and bincenters ###
rmin_norm = 10.**(-1.) # [r/rgal]
rmax_norm = 10.**(2.)  # [r/rgal]
    
radii_bins_norm, radii_bincents_norm = ru.returnlogbins([rmin_norm, rmax_norm], radii_binwidth)
            
# prepend 0 to the radial bins to capture the center sphere
radii_bins_norm     = np.insert(radii_bins_norm, 0, 0.)
radii_bincents_norm = np.insert(radii_bincents_norm, 0, radii_bins_norm[1]/2.)

nbins = len(radii_bins_norm)

tracer_ptn = il.util.partTypeNum('tracer')
star_ptn   = il.util.partTypeNum('star')
gas_ptn    = il.util.partTypeNum('gas')
bh_ptn     = il.util.partTypeNum('bh')

bary_ptns   = [gas_ptn,
               star_ptn,
               bh_ptn]

jellyscore_min = 16

def run_subfindGRP():

    dics = []
    
    f = h5py.File(direc + fname, 'a')
    for group_key in f.keys():
        dic = {}
        dic[group_key] = {}
        group = f[group_key]
        for dset_key in group.keys():
            dic[group_key][dset_key] = group[dset_key][:]
        dics.append(dic)

    Pool = mp.Pool(8) # should be 8 when running interactively; mp.cpu_count() for SLURM

    result_list = Pool.map(create_subfindGRP, dics)

    Pool.close()
    Pool.join()

    result = {}
    for i, d in enumerate(result_list):
        key = list(d.keys())[0]
        result[key] = d[key]  

    new_keys = ['radii', 'mass_shells', 'vol_shells', 'densities_shells',
                'SubhaloColdGasMass', 'SubhaloGasMass', 'SubhaloHotGasMass']

    for group_key in result.keys():
        group = f.require_group(group_key)        
        for dset_key in new_keys:  
            dset = result[group_key][dset_key]
            dataset = group.require_dataset(dset_key, shape=dset.shape, dtype=dset.dtype)
            dataset[:] = dset

    f.close()
    
    # post process the gas radial profiles
    add_memberflags()
    add_zooniverseflags()
    add_times()
    add_dmin()
    add_Nperipass()
    add_coldgasmasstau()
    add_quenchtimes()
    #add_tracers()
    #add_coldgasmasstracerstau()
    
    return


def create_subfindGRP(dic):
    
    # create a directory for the given jellyfish,
    # named after its subfindID at last Zooniverse inspection
    dict_list = []

    gal_key   = list(dic.keys())[0]
    gal       = dic[gal_key]
    SnapNum   = gal['SnapNum']
    SubfindID = gal['SubfindID']

    for snapnum_index, snapnum in enumerate(SnapNum):
        subfindID = SubfindID[snapnum_index]
    
        dict_list.append(return_subfindGRP(snapnum, subfindID))
    
    radii_dict = {}
    for d in dict_list:
        radii_dict.update(d)

    # initialize and fill result dicitonary
    # note that result_keys are vectors, and scalar_keys are scalars
    result_keys     = ['radii', 'mass_shells', 'vol_shells', 'densities_shells']
    shape           = (len(SnapNum), len(radii_dict['%d'%snapnum]['radii']))
    result          = {}
    result[gal_key] = gal
    for key in result_keys:
        result[gal_key][key] = np.zeros(shape, dtype=float)
    for key in scalar_keys:
        result[gal_key][key] = np.zeros(len(SnapNum))

    for row, snap_key in enumerate(radii_dict.keys()):
        for key in result_keys:
            result[gal_key][key][row,:] = radii_dict[snap_key][key]
        for key in scalar_keys:
            result[gal_key][key][row]   = radii_dict[snap_key][key]

    return result


def return_subfindGRP(snapnum, subfindID):
    
    print('Working on %s snap %s subfindID %d'%(sim, snapnum, subfindID))
    
    # initialize result
    result            = {}
    group_key         = '%d'%snapnum
    result[group_key] = {}

    # check if the subhalo is identified at this snap
    if subfindID == -1:
        for dset_key in dset_keys:
            result[group_key][dset_key] = np.ones(nbins-1, dtype=float) * -1.     
        for scalar_key in scalar_keys:
            result[group_key][scalar_key] = -1.
        return result 

    # load general simulation parameters
    header  = il.groupcat.loadHeader(basePath, snapnum)
    a       = header['Time'] # scale factor
    h       = header['HubbleParam'] # = 0.6774
    boxsize = header['BoxSize'] * a / h
        
    subhalofields = ['SubhaloHalfmassRadType', 'SubhaloPos']
    gasfields     = ['Coordinates', 'Masses', 'InternalEnergy',
                     'ElectronAbundance', 'StarFormationRate']
            
    subhalo      = ru.loadSingleFields(basePath, snapnum, subhaloID=subfindID, fields=subhalofields)
    subhalopos   = subhalo['SubhaloPos'] * a / h
    subhalo_rgal = 2. * subhalo['SubhaloHalfmassRadType'][star_ptn] * a / h 
                            
    # load gas particles for relevant halo
    gasparts = il.snapshot.loadSubhalo(basePath, snapnum, subfindID, gas_ptn, fields=gasfields)
    
    # if the satellite has no gas, write zeros for all dsets
    if gasparts['count'] == 0:
        for dset_index, dset_key in enumerate(dset_keys):
            result[group_key][dset_key] = np.zeros(nbins-1, float)
        for scalar_index, scalar_key in enumerate(scalar_keys):
            result[group_key][scalar_key] = 0.
        return result
    
    gas_coordinates        = gasparts['Coordinates'] * a / h
    gas_masses             = gasparts['Masses'] * 1.0e10 / h
    gas_internalenergies   = gasparts['InternalEnergy']
    gas_electronabundances = gasparts['ElectronAbundance']
    gas_starformationrates = gasparts['StarFormationRate']

    # save the total amount of gas
    subhalo_gasmass = np.sum(gas_masses)
    
    # calculate the distance between each gas cell and subhalo center  
    gas_radii = ru.mag(gas_coordinates, subhalopos, boxsize)
    
    # calculate the temperature of each gas cell
    gas_temperatures = ru.calc_temp(gas_internalenergies, gas_electronabundances, gas_starformationrates)
    
    # separate the gas into cold component
    coldgas_masses = gas_masses[gas_temperatures < tlim] 
    coldgas_radii  = gas_radii[gas_temperatures < tlim]    
    
    # calculate and save the total cold and hot gas masses
    subhalo_coldgasmass = np.sum(coldgas_masses)
    subhalo_hotgasmass  = subhalo_gasmass - subhalo_coldgasmass
    
    # sort the gas masses by their radius
    coldgas_masses = coldgas_masses[np.argsort(coldgas_radii)]
    coldgas_radii  = coldgas_radii[np.argsort(coldgas_radii)]

    # calculate the radial profile via histogram 
    radii_bins     = radii_bins_norm * subhalo_rgal # pkpc
    radii_bincents = radii_bincents_norm * subhalo_rgal # pkpc
    
    # set the volume of the shells; len(volume) = len(bincents) = len(bins) - 1
    vol_shells = (4./3.) * np.pi * ((radii_bins[2:])**3 - (radii_bins[1:-1])**3)
    vol_shells = np.insert(vol_shells, 0, (4./3.) * np.pi * (radii_bins[1])**3)
    
    mass_shells = np.histogram(coldgas_radii, bins=radii_bins, weights=coldgas_masses)[0]
            
    densities_shells = mass_shells / vol_shells
    
    dsets = [radii_bincents, mass_shells, vol_shells, densities_shells]
    for dset_index, dset_key in enumerate(dset_keys):
        result[group_key][dset_key] = dsets[dset_index]
        
    scalars = [subhalo_coldgasmass, subhalo_gasmass, subhalo_hotgasmass]
    
    for scalar_index, scalar_key in enumerate(scalar_keys):
        result[group_key][scalar_key] = scalars[scalar_index]

    return result


### post processing functions ###
# add membership flags -- central flag, pre-processed flag, member of final FoF flag
def add_memberflags():

    f = h5py.File(direc+fname, 'a')

    keys = ['central_flags', 'preprocessed_flags', 'memberlifof_flags']

    for group_key in f:
        group = f[group_key]
        SubfindID = group['SubfindID'][:]
        SubGroupFirstSub = group['SubGroupFirstSub'][:]
        SubhaloGrNr = group['SubhaloGrNr'][:]
        HostSubhaloGrNr = group['HostSubhaloGrNr'][:]
        HostSubfindID = group['HostSubfindID'][:]

        flags = np.zeros(len(SubfindID), dtype=int)
        
        # flag when the subhalo was a central galaxy
        # i.e., it's the primrary subhalo in the group
        centralflags = flags.copy()
        central_indices = np.where((SubfindID == SubGroupFirstSub) &
                                   (SubfindID != -1))[0]
        centralflags[central_indices] = 1

        # flag when the subhalo was preprocessed
        # i.e., it's a satellite of a FoF that's NOT it's last identified host
        preprocflags = flags.copy()
        preproc_indices = np.where((SubhaloGrNr != HostSubhaloGrNr) &
                                   (SubfindID != SubGroupFirstSub) &
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


def add_zooniverseflags():
    """
    Load the zooniverse catalogs and add inspected + jellyfish flags to the branches.
    Saves the flags directly to the GRP catalogs. No returns.
    """

    # load the inpsected and jellyfish ID dictionaries
    insIDs_dict, jelIDs_dict = load_zooniverseIDs()

    f = h5py.File(direc + fname, 'a')

    keys = ['ins_flags', 'jel_flags']

    for group_key in f.keys():
        group     = f[group_key]
        SnapNum   = group['SnapNum'][:]
        SubfindID = group['SubfindID'][:]
        ins_flags = np.zeros(len(SnapNum), dtype=int)
        jel_flags = np.zeros(len(SnapNum), dtype=int)

        for index, snap in enumerate(SnapNum):
            snap_key = '%03d'%snap
            try:
                if SubfindID[index] in insIDs_dict[snap_key]:
                    ins_flags[index] = 1
                    if SubfindID[index] in jelIDs_dict[snap_key]:
                        jel_flags[index] = 1
            except KeyError: # no zooniverse classifications as this snap
                continue

        # finish loop over SnapNum
        dsets = [ins_flags, jel_flags]
        for i, key in enumerate(keys):
            dset       = dsets[i]
            dataset    = group.require_dataset(key, shape=dset.shape, dtype=dset.dtype)
            dataset[:] = dset

    # finish loop over groups
    f.close()

    return

def load_zooniverseIDs():
    """
    Load all zooniverse catalogs. Create a dictionary with each snapshot as the key,
    and the entries are the subfindIDs of all inspected galaxies at that snapshot.
    Creates a second dictionary for when the galaxy is a jellyfish.
    Returns the dictionaries.
    """
    
    # load in the filenames for each snapshot, starting at the last snap
    indirec  = '../IllustrisTNG/%s/postprocessing/Zooniverse_CosmologicalJellyfish/flags/'%sim
    infname  = 'cosmic_jellyfish_flags_*.hdf5'
    infnames = glob.glob(indirec + infname)
    infnames.sort(reverse=True)

    # create dictionaries with snapnum as the key and lists of subfindIDs as the entires
    insIDs_dict = {}
    jelIDs_dict = {}
    for filename in infnames:
        snap_key = filename[-8:-5]
        f        = h5py.File(filename, 'r')
        done     = f['done'][0]
        Score    = f['Score'][0]
        
        insIDs_dict[snap_key] = np.where(done == 1)[0]
        jelIDs_dict[snap_key] = np.where(Score >= jellyscore_min)[0]

        f.close()

    return insIDs_dict, jelIDs_dict
    

def add_times():
    """
    Add redshift, cosmic time, and scale factor.
    """

    # tabulate redshift, scale factor, and calculate cosmic time
    # as functions of the snap number
    zs, cosmictimes = ru.timesfromsnap(basePath, range(100))
    cosmictimes /= 1.0e9 # convert to [Gyr]
    scales = 1. / (1. + zs)

    f = h5py.File(direc+fname, 'a')

    keys = ['CosmicTime', 'Redshift', 'Time']
    dsets = [cosmictimes, zs, scales]
    for group_key in f.keys():
        group = f[group_key]
        indices = group['SnapNum']
        for dset_index, dset_key in enumerate(keys):
            dset = dsets[dset_index][indices]
            dataset = group.require_dataset(dset_key, shape=dset.shape, dtype=dset.dtype)
            dataset[:] = dset

    f.close()

    return


def add_dmin():
    """
    add the min distance to the host.
    """
    f = h5py.File(direc+fname, 'a')
    keys = ['min_HostCentricDistance_norm', 'min_HostCentricDistance_phys']

    for group_key in f.keys():
        group = f[group_key]

        HostCentricDistance_phys = group['HostCentricDistance_phys'][:]        
        HostCentricDistance_norm = group['HostCentricDistance_norm'][:]

        dmin_phys = np.zeros(len(HostCentricDistance_phys), dtype=HostCentricDistance_phys.dtype)
        dmin_norm = np.zeros(len(HostCentricDistance_norm), dtype=HostCentricDistance_norm.dtype)

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
        
    

def add_Nperipass(mindist_phys=1000.0, mindist_norm=2.0):
    """
    add flags for pericenter passages.
    """
    
    f = h5py.File(direc+fname, 'a')
    keys = ['Nperipass', 'min_Dperi_norm', 'min_Dperi_phys']
    
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
    
        Nperipass = np.zeros(len(HostCentricDistance_phys), dtype=int)
        Dperi_norm_min = np.zeros(len(Nperipass), dtype=HostCentricDistance_norm.dtype)
        Dperi_phys_min = np.zeros(len(Nperipass), dtype=HostCentricDistance_phys.dtype)
        for i, peri_index in enumerate(peri_indices):
            Nperipass[:peri_index+1] += 1

            # add the minimum pericentric distance the galaxy has had
            if i == 0:
                Dperi_norm_min[:peri_index+1] = np.min(Dperi_norm[i:])
                Dperi_phys_min[:peri_index+1] = np.min(Dperi_phys[i:])

            else:
                Dperi_norm_min[peri_indices[i-1]+1:peri_index+1] = np.min(Dperi_norm[i:])
                Dperi_phys_min[peri_indices[i-1]+1:peri_index+1] = np.min(Dperi_phys[i:])
                

        dsets = [Nperipass, Dperi_norm_min, Dperi_phys_min]

        for dset_index, dset_key in enumerate(keys):
            dset = dsets[dset_index]
            dataset = group.require_dataset(dset_key, shape=dset.shape, dtype=dset.dtype)
            dataset[:] = dset

        
    f.close()

    return 


def add_quenchtimes():
    """
    Use the quenching and stellar assembly times catalog to add the
    quenching time (if the subhalo is quenched) for each subhalo. 
    """

    f = h5py.File(direc + fname, 'a')
    keys = np.array(list(f.keys()))
    
    # load the quenching catalog
    quench_direc = '../IllustrisTNG/%s/postprocessing/QuenchingStellarAssemblyTimes/'%sim
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


 
def add_coldgasmasstau():
    """
    add the peak cold gas mass time and the respective tau
    """
    
    def return_tau(peak_index, dset):
        """
        Calculate and return tau given the peak_index and the dset
        """
        tau = np.ones(len(dset), dtype=float) * -1.

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

    f = h5py.File(direc+fname, 'a')
    
    N_RM = 3 # the number of snapshots to average over for running median
             # should be an odd number

    keys = ['tau_rawpeak', 'tau_medpeak', 'tau_infall']

    f_keys = list(f.keys())

    for group_index, group_key in enumerate(f_keys):
        group = f[group_key]

        result = np.ones(len(group['SnapNum'][:]), dtype=float) * -1.
        
        # find the indices that the subbhalo was identified at
        SCGM_indices = np.where(group['SubfindID'][:] != -1)[0]        
        SubhaloColdGasMass = group['SubhaloColdGasMass'][SCGM_indices]

        # absolute maximum of the cold gas mass
        rawpeak_tau = result.copy()
        rawpeak_SCGM_index = np.argmax(SubhaloColdGasMass)
        rawpeak_tau[SCGM_indices] = return_tau(rawpeak_SCGM_index, SubhaloColdGasMass)

        # running median maximum of the cold gas mass
        # ensure that there are enough snaps to calc the running median
        # galaxies that do not reach z=0 will be ignored later anyways
        medpeak_tau = result.copy()
        if len(SubhaloColdGasMass) >=  N_RM:
            med_SCGM = ru.RunningMedian(SubhaloColdGasMass, N_RM)
            # choose the median x value corresponding to the max, as this is the peak
            medpeak_SCGM_index = int(np.median(np.argwhere(med_SCGM == max(med_SCGM)).T))
            #medpeak_SCGM_index = np.min(np.argwhere(med_SCGM == max(med_SCGM))) + int((N_RM - 1) / 2)
            medpeak_tau[SCGM_indices] = return_tau(medpeak_SCGM_index, SubhaloColdGasMass)

        # first infall time
        infall_tau = result.copy()
        infall_index = np.max(np.argwhere(group['memberlifof_flags'][:] == 1))
        infall_tau_index = np.where(group['SnapNum'][SCGM_indices] == group['SnapNum'][infall_index])[0][0]
        infall_tau[SCGM_indices] = return_tau(infall_tau_index, SubhaloColdGasMass)

        dsets = [rawpeak_tau, medpeak_tau, infall_tau]
        for dset_index, dset_key in enumerate(keys):
            dset = dsets[dset_index]
            dataset = group.require_dataset(dset_key, shape=dset.shape, dtype=dset.dtype)
            dataset[:] = dset
        
    # finish loop over branches
    
    f.close()
        
    return


def add_quenchtimes():
    """
    Use the quenching and stellar assembly times catalog to add the
    quenching time (if the subhalo is quenched) for each subhalo. 
    """

    f = h5py.File(direc + fname, 'a')
    keys = np.array(list(f.keys()))
    
    # load the quenching catalog
    quench_direc = '../IllustrisTNG/%s/postprocessing/QuenchingStellarAssemblyTimes/'%sim
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
    

def add_tracers():
    """
    Add tracer particle post-processing datasets to the GRP catalog.
    """

    off_direc = '../Output/%s_tracers_zooniverse/'%sim

    f = h5py.File(direc + fname, 'a')
    keys = np.array(list(f.keys()))

    group = f[keys[0]]
    NsubfindIDs = len(keys)
    max_snap = np.max(group['SnapNum'])
    min_snap = np.min(group['SnapNum'])
    snaps = np.arange(max_snap, min_snap-1, -1)
    CosmicTimes = group['CosmicTime'][:]

    header      = ru.loadHeader(basePath, max_snap)
    h           = header['HubbleParam']
    tracer_mass = header['MassTable'][tracer_ptn] * 1.0e10 / h
    
    # initialize results
    SubhaloColdGasTracer_Mass      = np.ones((NsubfindIDs, len(snaps))) * -1.
    SubhaloColdGasTracer_new       = np.ones((NsubfindIDs, len(snaps))) * -1.
    SubhaloColdGasTracer_out       = np.ones((NsubfindIDs, len(snaps))) * -1.
    SubhaloColdGasTracer_StripTot  = np.ones((NsubfindIDs, len(snaps))) * -1.
    SubhaloColdGasTracer_StripCold = np.ones((NsubfindIDs, len(snaps))) * -1.
    SubhaloColdGasTracer_Heat      = np.ones((NsubfindIDs, len(snaps))) * -1.
    SubhaloColdGasTracer_Star      = np.ones((NsubfindIDs, len(snaps))) * -1.
    SubhaloColdGasTracer_BH        = np.ones((NsubfindIDs, len(snaps))) * -1.

    # tabulate all GRP subfindIDs at each snap
    all_subfindIDs = np.zeros((snaps.size, keys.size), dtype=int) - 1
    for key_i, key in enumerate(keys):
        all_subfindIDs[:,key_i] = f[key]['SubfindID'][:]
    
    for snap_i, snap in enumerate(snaps):
        
        print(snap_i, snap)
        
        offsets = h5py.File(off_direc + 'offsets_%03d.hdf5'%snap, 'r')
        tracers = h5py.File(off_direc + 'tracers_%03d.hdf5'%snap, 'r')

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
        if snap_i != (len(snaps) - 1):

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
                    print('Warning for bound heated gas cells for %s %s subfindID %s'%(sim, snap, subfindID))
                
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


def add_tracers_postprocessing():
    """
    post process the tracer quantities
    """
    
    f = h5py.File(direc+fname, 'a')
    f_keys = np.array(list(f.keys()))
    
    result = np.ones(f[f_keys[0]]['SnapNum'].size, dtype=float) * -1.

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
            
            SCGM = group[SCGM_key][subhalo_indices]
            if 0 in SCGM:
                SCGM_flag = True
                start_index = np.where(SCGM == 0)[0].max()                
                subhalo_indices = subhalo_indices[start_index:]
                SCGM = SCGM[start_index:]
                
            if subhalo_indices.size > 1:

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


def add_coldgasmasstracerstau():
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
    
    f = h5py.File(direc+fname, 'a')
    f_keys = np.array(list(f.keys()))

    result = np.ones(f[f_keys[0]]['SnapNum'].size, dtype=float) * -1.

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
              

sims = ['TNG50-1']
for sim in sims:
    basePath = ru.ret_basePath(sim)
    #direc = '../Output/zooniverse/'
    fname = 'zooniverse_%s_%s_branches.hdf5'%(sim, key)

    direc = '../Output/%s_subfindGRP/'%sim
    #fname = 'subfind_%s_branches.hdf5'%sim

    #run_subfindGRP()
    add_tracers()
    add_tracers_postprocessing()
    add_coldgasmasstracerstau()



