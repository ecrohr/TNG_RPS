#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 15:25:41 2021

@author: rohr
"""

import illustris_python as il
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import numpy.ma as ma
import matplotlib as mpl
import time
from scipy.stats import binned_statistic_2d
from scipy.ndimage import gaussian_filter
import h5py
import multiprocessing as mp
import os
import rohr_utils as ru

global sim, basePath, nbins, snap
global direc, result
global grp_dict

nbins = 100
sim = 'TNG50-1'
basePath = ru.ret_basePath(sim)
snap = 99
depth_frac = 1.

# load general simulation parameters
header = ru.loadHeader(basePath, snap)
z = header['Redshift']
a = header['Time'] # scale factor
h = header['HubbleParam'] # = 0.6774
boxsize = header['BoxSize'] * a / h
    
gaspart_num = il.util.partTypeNum('gas')

gas_fields = ['Coordinates', 'Masses', 'InternalEnergy',
              'ElectronAbundance', 'StarFormationRate', 'GFM_Metallicity']

subfindIDs = [19,     63872,  96793,  117260, 117265,
              143896, 167399, 184958, 184959, 208820,
              220616, 229946, 253878, 264888, 275549,
              289386, 289401, 294876, 333427, 394625]

outdirec = '../Output/zooniverse/'

    
def run_creategasmap(sim, subfindIDs, mp_flag=False):

    global grp_dict

    grp_dict = load_grpdict(sim, key='jellyfish')

    keys = ['%08d'%subfindID for subfindID in subfindIDs]

    # test that each key actually works
    for key in keys:
        _ = grp_dict[key]
    print('All keys work properly.')

    result_list = []
    if mp_flag:
        Pool = mp.Pool(8)
        result_list = Pool.map(create_gasmap, keys)

    else:
        for i, key in enumerate(keys):
            result_list.append(create_gasmap(key))

    result = {}
    for i, key in enumerate(keys):
        result[key] = result_list[i]

    fname = 'zooniverse_%s_jellyfish_gasmaps.hdf5'%(sim)
    
    with h5py.File(outdirec + fname, 'a') as f:
        for group_key in result.keys():
            group = f.require_group(group_key)        
            for dset_key in grp_dict[keys[0]].keys():
                if '_map' in dset_key:
                    dset = result[group_key][dset_key]
                else:
                    dset = np.array([result[group_key][dset_key][0]])
                dataset = group.require_dataset(dset_key, shape=dset.shape, dtype=dset.dtype)
                dataset[:] = dset                    
                    
        f.close()

    return


def create_gasmap(key):
    """
    Given the group key, create the temperature and metallcitity gas maps, 
    add these maps to the group dictionary, and return the dict.
    """
    # for each subindID / key, load all the dict info
    group = grp_dict[key]

    result_sub = calc_gasmap(group, subhalo=True)
    result_fof = calc_gasmap(group, subhalo=False)

    result = np.zeros(result_sub.shape, dtype=result_sub.dtype)

    # fiducial map is the FoF map, masked by the subhalo map
    indices = result_sub[0] > 0
    for i, _ in enumerate(result):
        result[i] = result_fof[i]
        result[i][indices] = result_sub[i][indices]

    group['Temperature_map']     = result[0]
    group['GFM_Metallicity_map'] = result[1]

    return group


def calc_gasmap(group, subhalo=True):
    """
    Given the group from the grp_dictionary, calculate the mass weighted histograms.
    Subhalo flag determines whether the gas cells come from the FoF or the subhalo host
    """

    maxcoord = 20. * group['Subhalo_Rgal'][0]
    center = group['SubhaloPos'][0]

    # load gas cells of either the subhalo or the host
    if subhalo:
        subfindID = group['SubfindID'][0]
        gas_cells = il.snapshot.loadSubhalo(basePath, snap, subfindID, gaspart_num, fields=gas_fields)
    else:
        fofID     = group['SubhaloGrNr'][0]
        gas_cells = il.snapshot.loadHalo(basePath, snap, fofID, gaspart_num, fields=gas_fields)

    # we only care about the gas cells that are within the image domain
    gas_coordinates = ru.shift(gas_cells['Coordinates'] * a / h, center, boxsize) # center at [0, 0, 0]

    gas_indices = ((abs(gas_coordinates[:,0]) <= maxcoord) &
                   (abs(gas_coordinates[:,1]) <= maxcoord) &
                   (abs(gas_coordinates[:,2]) <= maxcoord * depth_frac))

    for key in gas_cells.keys():
        if key == 'count':
            gas_cells[key] = len(gas_indices[gas_indices])
        else:
            gas_cells[key] = gas_cells[key][gas_indices]

    # convert units and use log values for the final histograms 
    gas_cells       = ru.calc_temp_dict(gas_cells)
    gas_masses      = gas_cells['Masses'] * 1.0e10 / h
    gas_coordinates = gas_coordinates[gas_indices]
    gas_cells['Temperature']     = np.log10(gas_cells['Temperature'])
    gas_cells['GFM_Metallicity'] = np.log10(gas_cells['GFM_Metallicity'] / 0.0127)

    # create mass histogram
    mass_result = binned_statistic_2d(gas_coordinates[:,0], gas_coordinates[:,1], gas_masses, 'sum',
                                      bins=nbins, range=[[-maxcoord, maxcoord], [-maxcoord, maxcoord]], expand_binnumbers=True)

    # mask pixels without any gas cells to avoid dividing by zero
    mass_vals = mass_result[0]
    mass_vals[mass_vals < 1.0e-3] = 1.

    # get the binnumbers for each gas cell
    ix = mass_result.binnumber[0]
    iy = mass_result.binnumber[1]
    ix -= 1
    iy -= 1

    # create weights for temp and metallicity histograms
    map_keys = ['Temperature', 'GFM_Metallicity']
    result = np.zeros([len(map_keys), nbins, nbins], dtype=gas_cells[map_keys[0]].dtype)

    for i, map_key in enumerate(map_keys):
        weights = gas_cells[map_key] * gas_masses / mass_vals[ix,iy]
        hist2d  = binned_statistic_2d(gas_coordinates[:,0], gas_coordinates[:,1], weights, 'sum',
                                      bins=nbins, range=[[-maxcoord, maxcoord], [-maxcoord, maxcoord]])

        gas_map = hist2d[0]

        result[i] = gas_map

    return result 


def load_grpdict(sim, key):
    """
    helper function to load the grp_dictionary 
    """
    # key == 'jellyfish' or 'control'; otherwise returns KeyError
    result = {}
    fname = 'zooniverse_%s_%s_branches_clean.hdf5'%(sim, key)
    with h5py.File(outdirec + fname, 'a') as f:
        for group_key in f.keys():
            result[group_key] = {}
            for dset_key in f[group_key].keys():
                result[group_key][dset_key] = f[group_key][dset_key][:]
        f.close()
        
    return result


run_creategasmap(sim, subfindIDs, mp_flag=False)
