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
import multiprocessing as mp
from importlib import reload

global sim, basePath, fname
global tlim, radii_binwidth, key
global scalar_keys
global rmin_norm, rmax_norm, radii_bins_norm, radii_bincents_norm, nbins

tlim = 10.**(4.5) # K; cutoff between cold and hot CGM gas
radii_binwidth = 0.1
key = 'inspected'

scalar_keys = ['SubhaloColdGasMass', 'SubhaloGasMass', 'SubhaloHotGasMass']

### define radial bins and bincenters ###
rmin_norm = 10.**(-1.) # [r/rgal]
rmax_norm = 10.**(2.)  # [r/rgal]
    
radii_bins_norm, radii_bincents_norm = ru.returnlogbins([rmin_norm, rmax_norm], radii_binwidth)
            
# prepend 0 to the radial bins to capture the center sphere
radii_bins_norm     = np.insert(radii_bins_norm, 0, 0.)
radii_bincents_norm = np.insert(radii_bincents_norm, 0, radii_bins_norm[1]/2.)

nbins = len(radii_bins_norm)

def run_satelliteGRP():
    """
    dics = []
    f = h5py.File('../Output/'+fname, 'a')
    for group_key in f.keys():
        dic = {}
        dic[group_key] = {}
        group = f[group_key]
        for dset_key in group.keys():
            dic[group_key][dset_key] = group[dset_key][:]
        dics.append(dic)

    Pool = mp.Pool(mp.cpu_count()) # should be 8 when running interactively; mp.cpu_count() for SLURM

    result_list = Pool.map(create_satelliteGRP, dics)

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
    """
    
    # post process the gas radial profiles
    #add_memberflags()
    #add_times()
    #add_dmin()
    #add_Nperipass()
    add_coldgasmasstau()

    return


def create_satelliteGRP(dic):
    
    # create a directory for the given jellyfish,
    # named after its subfindID at last Zooniverse inspection
    dict_list = []

    gal_key   = list(dic.keys())[0]
    gal       = dic[gal_key]
    SnapNum   = gal['SnapNum']
    SubfindID = gal['SubfindID']

    for snapnum_index, snapnum in enumerate(SnapNum):
        subfindID = SubfindID[snapnum_index]
    
        dict_list.append(return_satelliteGRP(snapnum, subfindID))
    
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


def return_satelliteGRP(snapnum, subfindID):
    
    print('Working on %s snap %s subfindID %d'%(sim, snapnum, subfindID))
    
    result            = {}
    group_key         = '%d'%snapnum
    result[group_key] = {}

    # load general simulation parameters
    header  = il.groupcat.loadHeader(basePath, snapnum)
    a       = header['Time'] # scale factor
    h       = header['HubbleParam'] # = 0.6774
    boxsize = header['BoxSize'] * a / h
    
    gaspartnum  = il.util.partTypeNum('gas')
    starpartnum = il.util.partTypeNum('star')
    
    subhalofields = ['SubhaloHalfmassRadType', 'SubhaloPos']
    gasfields     = ['Coordinates', 'Masses', 'InternalEnergy',
                     'ElectronAbundance', 'StarFormationRate']
            
    subhalo      = ru.loadSingleFields(basePath, snapnum, subhaloID=subfindID, fields=subhalofields)
    subhalopos   = subhalo['SubhaloPos'] * a / h
    subhalo_rgal = 2. * subhalo['SubhaloHalfmassRadType'][starpartnum] * a / h 
        
    dset_keys = ['radii', 'mass_shells', 'vol_shells', 'densities_shells']
                    
    # load gas particles for relevant halo
    gasparts = il.snapshot.loadSubhalo(basePath, snapnum, subfindID, gaspartnum, fields=gasfields)
    
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

    f = h5py.File('../Output/'+fname, 'a')

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
        central_indices = np.where(SubfindID == SubGroupFirstSub)[0]
        centralflags[central_indices] = 1

        # flag when the subhalo was preprocessed
        # i.e., it's a satellite of a FoF that's NOT it's last identified host
        preprocflags = flags.copy()
        preproc_indices = np.where((SubhaloGrNr != HostSubhaloGrNr) &
                                   (SubfindID != SubGroupFirstSub))[0]
        preprocflags[preproc_indices] = 1

        # flag when the subhalo was in its last identified FoF
        # note that min(inLIFoF_indices) is the infall time
        inLIFoFflags = flags.copy()
        inLIFoF_indices = np.where(SubhaloGrNr == HostSubhaloGrNr)[0]
        inLIFoFflags[inLIFoF_indices] = 1

        dsets = [centralflags, preprocflags, inLIFoFflags]
        for dset_index, dset_key in enumerate(keys):
            dset = dsets[dset_index]
            dataset = group.require_dataset(dset_key, shape=dset.shape, dtype=dset.dtype)
            dataset[:] = dset

    f.close()

    return

# add redshift, cosmic time, and scale factor
def add_times():

    # tabulate redshift, scale factor, and calculate cosmic time
    # as functions of the snap number
    zs, cosmictimes = ru.timesfromsnap(basePath, range(100))
    cosmictimes /= 1.0e9 # convert to [Gyr]
    scales = 1. / (1. + zs)

    f = h5py.File('../Output/'+fname, 'a')

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


# add the min distance to the host
def add_dmin():

    f = h5py.File('../Output/'+fname, 'a')
    keys = ['min_HostCentricDistance_norm', 'min_HostCentricDistance_phys']

    for group_key in f.keys():
        group = f[group_key]

        HostCentricDistance_phys = group['HostCentricDistance_phys'][:]        
        HostCentricDistance_norm = group['HostCentricDistance_norm'][:]

        dmin_phys = np.zeros(len(HostCentricDistance_phys), dtype=HostCentricDistance_phys.dtype)
        dmin_norm = np.zeros(len(HostCentricDistance_norm), dtype=HostCentricDistance_norm.dtype)

        for i, _ in enumerate(HostCentricDistance_phys):
            dmin_phys[i] = np.min(HostCentricDistance_phys[i:])
            dmin_norm[i] = np.min(HostCentricDistance_norm[i:])

        dsets = [dmin_norm, dmin_phys]
        for dset_index, dset_key in enumerate(keys):
            dset = dsets[dset_index]
            dataset = group.require_dataset(dset_key, shape=dset.shape, dtype=dset.dtype)
            dataset[:] = dset

            
    f.close()

    return
        
    

# add flags for pericenter passages
def add_Nperipass(mindist_phys=1000.0, mindist_norm=2.0):

    f = h5py.File('../Output/'+fname, 'a')
    keys = ['Nperipass', 'min_Dperi_norm', 'min_Dperi_phys']
    
    for group_key in f.keys():
        group = f[group_key]

        HostCentricDistance_phys = group['HostCentricDistance_phys'][:]    
        min_indices_phys = argrelextrema(HostCentricDistance_phys, np.less)[0]
    
        HostCentricDistance_norm = group['HostCentricDistance_norm'][:]
    
        indices = np.where((HostCentricDistance_phys[min_indices_phys] < mindist_phys) 
                           & (HostCentricDistance_norm[min_indices_phys] < mindist_norm))[0]
    
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


# add the peak cold gas mass time and the respective tau 
def add_coldgasmasstau():

    # must be called after defining SubhaloColdGasMass
    def return_tau(peak_index):
        peak = SubhaloColdGasMass[peak_index]
        # if a subhalo is its own LI host, then peak_index occurs at the first snapshot,
        # and its cold gas mass may be zero -- if so, then return a list of -1.
        # or if M_coldgas(infall) = 0, then return 0 at peak_index and -1 elsewhere.
        if peak == 0:
            tau =  np.ones(len(SubhaloColdGasMass), dtype=float) * -1.
            tau[peak_index] = 0.
            return tau
        
        tau = np.zeros(len(SubhaloColdGasMass), dtype=float)
        tau[:peak_index+1] = (peak - SubhaloColdGasMass[:peak_index+1]) / peak * 100.
        tau[peak_index+1:] = -1

        return tau

    f = h5py.File('../Output/'+fname, 'a')
    
    N_RM = 3 # the number of snapshots to average over for running median
             # should be an odd number

    keys = ['tau_rawpeak', 'tau_medpeak', 'tau_infall']

    f_keys = list(f.keys())

    for group_index, group_key in enumerate(f_keys):
        group = f[group_key]
        SubhaloColdGasMass = group['SubhaloColdGasMass'][:]

        # absolute maximum of the cold gas mass
        rawpeak_SCGM_index = np.argmax(SubhaloColdGasMass)
        rawpeak_tau = return_tau(rawpeak_SCGM_index)

        # running median maximum of the cold gas mass
        # ensure that there are enough snaps to calc the running median
        # galaxies that do not reach z=0 will be ignored later anyways
        if len(SubhaloColdGasMass) < N_RM:
            strings = (sim, key, group_key)
            print('%s %s group_key %s does not have enough snaps to calc tau.'%strings)
            for dset_index, dset_key in enumerate(keys):
                dset = np.ones(len(SubhaloColdGasMass)) * -1.
                dataset = group.require_dataset(dset_key, shape=dset.shape, dtype=dset.dtype)
                dataset[:] = dset
            continue
            
        med_SCGM = ru.RunningMedian(SubhaloColdGasMass, N_RM)
        # if there are multiple maxima of the running median, choose the latest time
        medpeak_SCGM_index = np.min(np.argwhere(med_SCGM == max(med_SCGM))) + int((N_RM - 1) / 2)
        medpeak_tau = return_tau(medpeak_SCGM_index)

        # infall time
        infall_tau_index = np.max(np.argwhere(group['memberlifof_flags'][:] == 1))
        infall_tau = return_tau(infall_tau_index)

        dsets = [rawpeak_tau, medpeak_tau, infall_tau]
        for dset_index, dset_key in enumerate(keys):
            dset = dsets[dset_index]
            dataset = group.require_dataset(dset_key, shape=dset.shape, dtype=dset.dtype)
            dataset[:] = dset

    f.close()
        
    return
    

sims = ['TNG50-1', 'TNG100-1']
for sim in sims:
    basePath = ru.ret_basePath(sim)
    fname = 'zooniverse_%s_%s_branches.hdf5'%(sim, key)

    run_satelliteGRP()



