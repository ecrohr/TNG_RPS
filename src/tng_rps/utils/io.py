"""
io.py written by Eric Rohr
This file contains useful functions related to the input and output 
of AREPO simulation outputs with the primary focus being on 
cosmological hydrodynamical simulations with AREPO, such as the 
IllustrisTNG simulations. 
"""

import illustris_python as il 
import h5py
import numpy as np
import os
from pathlib import Path
import six
from astropy import units as u 
from astropy import constants as const 
from stellar_array_helpers import expand_all_arrays
from createMCSTFiles import createMCSTFiles
from astropy.cosmology import FlatLambdaCDM
from .units import *

def loadSubboxSubset(basePath, snapNum, subboxNum, partType, fields=None, subset=None, mdi=None, sq=True, float32=False):
    """ 
    Load a subset of fields for all particles/cells of a given partType.
        If offset and length specified, load only that subset of the partType.
        If mdi is specified, must be a list of integers of the same length as fields,
        giving for each field the multi-dimensional index (on the second dimension) to load.
          For example, fields=['Coordinates', 'Masses'] and mdi=[1, None] returns a 1D array
          of y-Coordinates only, together with Masses.
        If sq is True, return a numpy array instead of a dict if len(fields)==1.
        If float32 is True, load any float64 datatype arrays directly as float32 (save memory). 
    """
    
    result = {}

    ptNum = il.util.partTypeNum(partType)
    gName = "PartType" + str(ptNum)

    # make sure fields is not a single element
    if isinstance(fields, six.string_types):
        fields = [fields]

    # load header from first chunk
    with h5py.File(subboxPath(basePath, snapNum, subboxNum), 'r') as f:

        header = dict(f['Header'].attrs.items())
        nPart = il.snapshot.getNumPart(header)

        # decide global read size, starting file chunk, and starting file chunk offset
        if subset:
            offsetsThisType = subset['offsetType'][ptNum] - subset['snapOffsets'][ptNum, :]

            fileNum = np.max(np.where(offsetsThisType >= 0))
            fileOff = offsetsThisType[fileNum]
            numToRead = subset['lenType'][ptNum]
        else:
            fileNum = 0
            fileOff = 0
            numToRead = nPart[ptNum]

        result['count'] = numToRead

        if not numToRead:
            # print('warning: no particles of requested type, empty return.')
            return result

        # find a chunk with this particle type
        i = 1
        while gName not in f:
            f = h5py.File(subboxPath(basePath, snapNum, subboxNum, i), 'r')
            i += 1

        # if fields not specified, load everything
        if not fields:
            fields = list(f[gName].keys())

        for i, field in enumerate(fields):
            # verify existence
            if field not in f[gName].keys():
                raise Exception("Particle type ["+str(ptNum)+"] does not have field ["+field+"]")

            # replace local length with global
            shape = list(f[gName][field].shape)
            shape[0] = numToRead

            # multi-dimensional index slice load
            if mdi is not None and mdi[i] is not None:
                if len(shape) != 2:
                    raise Exception("Read error: mdi requested on non-2D field ["+field+"]")
                shape = [shape[0]]

            # allocate within return dict
            dtype = f[gName][field].dtype
            if dtype == np.float64 and float32: dtype = np.float32
            result[field] = np.zeros(shape, dtype=dtype)

    # loop over chunks
    wOffset = 0
    origNumToRead = numToRead

    while numToRead:
        f = h5py.File(subboxPath(basePath, snapNum, subboxNum, fileNum), 'r')

        # no particles of requested type in this file chunk?
        if gName not in f:
            f.close()
            fileNum += 1
            fileOff  = 0
            continue

        # set local read length for this file chunk, truncate to be within the local size
        numTypeLocal = f['Header'].attrs['NumPart_ThisFile'][ptNum]

        numToReadLocal = numToRead

        if fileOff + numToReadLocal > numTypeLocal:
            numToReadLocal = numTypeLocal - fileOff

        #print('['+str(fileNum).rjust(3)+'] off='+str(fileOff)+' read ['+str(numToReadLocal)+\
        #      '] of ['+str(numTypeLocal)+'] remaining = '+str(numToRead-numToReadLocal))

        # loop over each requested field for this particle type
        for i, field in enumerate(fields):
            # read data local to the current file
            if mdi is None or mdi[i] is None:
                result[field][wOffset:wOffset+numToReadLocal] = f[gName][field][fileOff:fileOff+numToReadLocal]
            else:
                result[field][wOffset:wOffset+numToReadLocal] = f[gName][field][fileOff:fileOff+numToReadLocal, mdi[i]]

        wOffset   += numToReadLocal
        numToRead -= numToReadLocal
        fileNum   += 1
        fileOff    = 0  # start at beginning of all file chunks other than the first

        f.close()

    # verify we read the correct number
    if origNumToRead != wOffset:
        raise Exception("Read ["+str(wOffset)+"] particles, but was expecting ["+str(origNumToRead)+"]")

    # only a single field? then return the array instead of a single item dict
    if sq and len(fields) == 1:
        return result[fields[0]]

    return result


def subboxPath(basePath, snapNum, subboxNum, chunkNum=0):
    """ Return absolute path to a subbox snapshot HDF5 file. """
    str_subboxNum = 'subbox%01d'%subboxNum
    subboxPath = basePath + str_subboxNum
    snapPath = subboxPath + '/snapdir_' + str_subboxNum + '_' + str(snapNum).zfill(3) + '/'
    filePath = snapPath + 'snap_' + str_subboxNum + '_' + str(snapNum).zfill(3)
    filePath += '.' + str(chunkNum) + '.hdf5'
    return filePath

def loadHeader(basePath, snapNum):
    """ load Header information for given snapshot """
    snap = h5py.File(il.snapshot.snapPath(basePath, snapNum), 'r')
    Header = dict(snap['Header'].attrs.items())
    snap.close()
    return Header


def loadParameters(basePath, snapNum):
    """ load Parameters information for given snapshot """
    snap = h5py.File(il.snapshot.snapPath(basePath, snapNum), 'r')
    Parameters = dict(snap['Parameters'].attrs.items())
    snap.close()
    return Parameters


def loadConfig(basePath, snapNum):
    """ load Config information for given snapshot """
    snap = h5py.File(il.snapshot.snapPath(basePath, snapNum), 'r')
    Config = dict(snap['Config'].attrs.items())
    snap.close()
    return Config


def loadMainTreeBranch(basePath, snap, subfindID, fields=None, treeName='SubLink_gal',
                       min_snap=0, max_snap=99):
    """
    Return the entire main branch (progenitor + descendant) of a given subhalo.
    When snap == 99, then just returns the MPB.
    Has the option only to return the tree between min and max snaps. 
    if fields = None (default), then returns all fields.
    CURRENTLY ONLY WORKS FOR TNG SIMS WITH 100 SNAPSHOTS
    """
    
    # start by loading the MPB
    if not fields:
        subMPB = il.sublink.loadTree(basePath, snap, subfindID, treeName=treeName,
                                     onlyMPB=True)
        fields = []
        for key in subMPB.keys():
            if key != 'count':
                fields.append(key)
    else:
        subMPB = il.sublink.loadTree(basePath, snap, subfindID, treeName=treeName,
                                     fields=fields, onlyMPB=True)
        # make sure fields is not a single element
        if isinstance(fields, six.string_types):
            fields = [fields]

    # does the tree exist? 
    if not subMPB:
        return 
    
    # if snap == 99, then just return the MPB [ONLY FOR TNG SIMS]
    if snap == 99:
        tree = subMPB
        tree['count'] = len(tree['SnapNum'])
        return tree

    # load the MDB and combine 
    subMDB = il.sublink.loadTree(basePath, snap, subfindID, treeName=treeName,
                                 fields=fields, onlyMDB=True)

    # check if there's an issue with the MDB -- if the MDB reaches z=0
    # if so, then only use the MPB
    if (subMDB['count'] + snap) > (99 + 1):

        # find where the MDB stops
        stop  = -(max_snap - min_snap + 1)
        start = np.max(np.where((subMDB['SnapNum'][1:] - subMDB['SnapNum'][:-1]) >= 0)) + 1

        for key in fields:
            subMDB[key] = subMDB[key][start:stop]
        subMDB['count'] = len(subMDB[key])

    # for the clean MDB, combine the MPB and MDB trees
    tree = {}
    for key in subMDB:
        if key == 'count':
            tree[key] = subMDB[key] + subMPB[key] - 1
        else:
            tree[key] = np.concatenate([subMDB[key][:-1], subMPB[key]])


    indices = (tree['SnapNum'] >= min_snap) & (tree['SnapNum'] <= max_snap)
    for field in fields:
        tree[field] = tree[field][indices]
    tree['count'] = len(indices[indices])
    
    return tree

def convertSnapshotUnits(basePath, snapNum, dic):
    """ Convert all loaded snapshot properties to standard units """

    Header = loadHeader(basePath, snapNum)

    for key in list(dic.keys()):
        dset = dic[key]
        # check if the quantity already has code units attached
        if isinstance(dset, u.quantity.Quantity):
            # yes, let's convert to standard units
            # NB: BH_Mdot and related units are falsely coded (see arepo PR 419)
            #     so manually overwrite the unit scalings for these quantities
            if key in ['BH_Mdot', 'BH_MdotBondi', 'BH_MdotEddington'] and (dset.unit == (code_length * code_velocity / code_mass)):
                dic[key] = (dset.value * 10.22) * standard_massderivative
            # same for BH_Pressure (see arepo PR 420):
            elif (key == 'BH_Pressure') and (dset.unit == (code_velocity**2 * code_length * code_mass)):
                dic[key] = (dset.value * Header['Time']**(-4) * code_length**(-3) * code_mass * code_velocity**2).to(standard_pressure) 
            # manual edits to additional datasets where the scalings and units are incorrect
            elif key == 'StarFormationRate':
                dic[key] = (dset.value * u.Msun / u.yr).to(standard_massderivative)
            elif key in ['Metallicity', 'GFM_Metallicity', 'GFM_Metals', 'GFM_MetalsTagged', 'AmbientMetallicity']:
                dic[key] = dset.value / 0.0127 * standard_metallicity
            elif key == 'JeansNumber':
                dic[key] = dset.value * standard_JeansNumber
            elif key == 'CoolingTime':
                dic[key] = (dset.value * u.Gyr).to(standard_time)
            elif key in ['GrackleTemperature', 'Temperature', 'AmbientTemperature']:
                dic[key] = dset.value * standard_temperature
            elif key == 'GFM_CoolingRate':
                dic[key] = (dset.value * u.erg * u.cm**3 / u.s)
            elif key == 'Age':
                dic[key] = (dset.value * u.yr).to(standard_time)
            
            # begin standard unit conversions
            elif dset.unit == code_mass**0 == code_length**0 == code_velocity**0:
                dic[key] = dset.value * u.dimensionless_unscaled 
            elif dset.unit == code_mass:
                dic[key] = dset.to(standard_mass)
            elif dset.unit == code_length:
                dic[key] = dset.to(standard_length)
            elif dset.unit == code_velocity:
                dic[key] = dset.to(standard_velocity)
            elif dset.unit == code_mass * code_velocity / code_length:
                dic[key] = dset.to(standard_massderivative)
            elif dset.unit == code_mass / code_length**3:
                dic[key] = dset.to(standard_density)
            elif dset.unit == (code_velocity)**2:
                dic[key] = dset.to(standard_velocity**2)
            elif dset.unit == (code_mass * code_velocity**2 / code_length**3):
                dic[key] = dset.to(standard_pressure)
            elif dset.unit == (code_mass * code_velocity**2):
                dic[key] = dset.to(standard_energy)
            elif key == 'GFM_AGNRadiation':
                dic[key] = (dset.value * u.erg / u.s / u.cm**2) / (4. * np.pi)
            elif key == 'StellarArray':
                dic[key] = expand_all_arrays(dset) * u.dimensionless_unscaled
                dic['StellarArrayMassBins'] = loadStellarArrayMassBins() * u.Msun
            elif dset.unit == code_mass * code_velocity**2 / code_length:
                dic[key] = dset.to(standard_power)
            else:
                raise ValueError('%s with unit %s has not had its units converted'%(key, dset.unit))

        # if no code units, then manually attach units 
        else:
            dimensionless_keys = ['AllowRefinement', 'HIIMassFraction', 'HIMassFraction', 
                                  'HeIIIMassFraction', 'HeIIMassFraction', 'HeIMassFraction',
                                  'Machnumber', 'StromgrenSourceID', 'count', 'TimebinHydro', 'TimeStep',
                                  'ParentID', 'TracerID',  'ParticleIDs', 'FluidQuantities',
                                  'ElectronAbundance', 'NeutralHydrogenAbundance', 'GFM_StellarFormationTime',
                                  'IonisingPhotonRate1e49', 'NumberOfSupernovaEvents', 'NumberOfSupernovae',
                                  'BH_Progs', 'StromgrenSourceID', 'StellarFormationTime', 'LocalFlag',
                                  'BH_WindCount', 'BH_WindTimes']
            if key in dimensionless_keys:
                dic[key] *= u.dimensionless_unscaled
            elif key in ['BH_Mdot', 'BH_MdotBondi', 'BH_MdotEddington']:
                dic[key] = (dset * 10.22) * standard_massderivative
            elif key == 'BH_BPressure':
                dic[key] = (dset * (Header['HubbleParam'] / Header['Time'])**4 * code_mass * code_velocity**2 / Header['Time']**3 / code_length**3 * 4. * np.pi).to(standard_pressure)
            elif key in ['BH_Pressure', 'Pressure']:
                dic[key] = (dset * Header['Time']**(-3) * Header['HubbleParam']**2 * code_length**(-3) * code_mass * code_velocity**2).to(standard_pressure)
            elif key in ['BH_CumEgyInjection_QM', 'BH_CumEgyInjection_RM', 'BH_MPB_CumEgyHigh', 'BH_MPB_CumEgyLow']:
                dic[key] = (dset * code_mass / Header['HubbleParam'] * (Header['Time'] * code_length / Header['HubbleParam'])**2 / 
                            (code_length / code_velocity / Header['HubbleParam']**2)).to(standard_energy)
            elif key in ['CenterOfMass', 'Coordinates', 'SubfindHsml', 'BirthPos', 'StromgrenRadius', 'StellarHsml', 'BH_Hsml']:
                dic[key] = (dset * Header['Time'] / Header['HubbleParam'] * code_length).to(standard_length)
            elif key in ['Density', 'SubfindDMDensity', 'SubfindDensity', 'AmbientDensity']:
                dic[key] = (dset * code_mass / Header['HubbleParam'] / 
                            (code_length * Header['Time'] / Header['HubbleParam'])**3).to(standard_density)
            elif key in ['GFM_StellarPhotometrics']:
                dic[key] = dset * u.mag
            elif key in ['Metallicity', 'GFM_Metallicity', 'GFM_Metals', 'GFM_MetalsTagged', 'AmbientMetallicity']:
                dic[key] = dset / 0.0127 * standard_metallicity
            elif key == 'EnergyDissipation':
                dic[key] = (dset * Header['Time']**-1 * code_mass / code_length * code_velocity**3).to(standard_energy / standard_time)
            elif key == 'GFM_AGNRadiation':
                dic[key] = dset * 4. * np.pi * u.erg / u.s / u.cm**2
            elif key == 'GFM_CoolingRate':
                dic[key] = (dset * u.erg * u.cm**3 / u.s)
            elif key in ['GFM_WindDMVelDisp', 'SubfindVelDisp']:
                dic[key] = (dset * code_velocity).to(standard_velocity)
            elif key in ['GFM_WindHostHaloMass', 'Masses', 'GFM_InitialMass', 'BH_CumMassGrowth_QM', 'BH_CumMassGrowth_RM',
                         'BH_HostHaloMass', 'BH_Mass', 'HighResGasMass', 'IMFMass', 'LowMass', 'MassDeposited']:
                dic[key] = (dset / Header['HubbleParam'] * code_mass).to(standard_mass)
            elif key in ['InternalEnergy', 'InternalEnergyOld', 'BH_U']:
                dic[key] = (dset * code_velocity**2).to(standard_velocity**2)
            elif key in ['MagneticField']:
                dic[key] = (dset * Header['HubbleParam'] / Header['Time']**2 * (code_mass / code_length)**(1/2) / (code_length / code_velocity)).to(standard_pressure**(1/2))
            elif key in ['Potential']:
                dic[key] = (dset / Header['Time'] * code_velocity**2).to(standard_velocity**2)
            elif key in ['StarFormationRate']:
                dic[key] = (dset * u.Msun / u.yr).to(standard_massderivative)
            elif key in ['Velocities', 'BirthVel']:
                dic[key] = (dset * np.sqrt(Header['Time']) * code_velocity).to(standard_velocity)
            elif key == 'GrackleCoolTime':
                dic[key] = (dset / Header['HubbleParam'] * standard_length / standard_velocity).to(standard_time)
            elif key in ['GrackleTemperature', 'Temperature', 'AmbientTemperature']:
                dic[key] = dset * standard_temperature
            elif key in ['RadiationEnergyDensity']:
                dic[key] = (dset * Header['HubbleParam']**2 * code_mass * code_velocity**2 / code_length**3).to(standard_energy / standard_volume)
            elif key in ['StellarArray']:
                dic[key] = expand_all_arrays(dset.astype(np.uint64)) * u.dimensionless_unscaled
                dic['StellarArrayMassBins'] = loadStellarArrayMassBins() * u.Msun
            elif key == 'StellarLuminosity':
                dic[key] = (dset.value * code_mass * code_velocity**3 / code_length).to(standard_power)
            elif key in ['MagneticFieldDivergence', 'MagneticFieldDivergenceAlternative']:
                dic[key] *= u.def_unit('%s_units'%key)
            else:
                dic[key] *= u.def_unit('code_%s'%key)

        dset = dic[key]

        # final check that all units have been converted
        if 'code' in dic[key].unit.to_string():
            raise UserWarning('%s not converted from units %s'%(key, dic[key].unit.to_string()))

    return
            

def convertGroupUnits(basePath, snapNum, dic):
    """ Convert all loaded properties to standard units """

    Header = loadHeader(basePath, snapNum)

    for key in dic:
        dset = dic[key]

        if 'Mass' in key or '_M_' in key:
            dic[key] = (dset / Header['HubbleParam'] * code_mass).to(standard_mass)
        elif 'Mdot' in key:
            dic[key] = (dset * 10.22) * u.Msun / u.yr
        elif (key in ['SubhaloCM', 'SubhaloPos', 'SubhaloHalfmassRad', 'SubhaloHalfmassRadType', 'SubhaloStellarPhotometricsRad', 'SubhaloVmaxRad', 'GroupPos']) or '_R_' in key:
            dic[key] = (dset * Header['Time'] / Header['HubbleParam'] * code_length).to(standard_length)
        elif key in ['SubhaloVel', 'SubhaloVelDisp', 'SubhaloVmax']:
            dic[key] = (dset * code_velocity).to(standard_velocity)
        elif key in ['SubhaloSFR', 'SubhaloSFRinHalfRad', 'SubhaloSFRinMaxRad', 'SubhaloSFRinRad',
                     'GroupSFR']:
            dic[key] = (dset) * u.Msun / u.yr
        elif key == 'SubhaloSpin':
            dic[key] = (dset * Header['Time'] / Header['HubbleParam'] * code_length * code_velocity).to(standard_length * standard_velocity)
        elif key == 'GroupVel':
            dic[key] = (dset / Header['Time'] * code_velocity).to(standard_velocity)
        elif 'Metallicity' in key:
            dic[key] = (dset / 0.0127) * standard_metallicity
        elif key == 'SubhaloStellarPhotometrics':
            dic[key] = dset * u.mag
        elif key in ['SubhaloBfldDisk', 'SubhaloBfldHalo']:
            dic[key] = (dset * Header['HubbleParam'] / (Header['Time'])**2 * (code_mass / code_length)**(1/2) / (code_length / code_velocity)).to(standard_pressure**(1/2))
        elif (key in ['SubhaloGrNr', 'SubhaloIDMostbound', 'SubhaloLen', 'SubhaloLenType', 'SubhaloParent', 'SubhaloFlag',
                     'GroupCM', 'GroupFirstSub', 'GroupLen', 'GroupLenType', 'GroupNsubs', 'GroupContaminationFracByMass', 
                     'GroupContaminationFracByNumPart', 'GroupOrigHaloID', 'GroupPrimaryZoomTarget', 'GroupOffsetType', 
                     'SubhaloOrigHaloID', 'SubhaloOffsetType', 'count']) or ('MetalFractions' in key):
            dic[key] = dset * u.dimensionless_unscaled

        if not isinstance(dic[key], u.Quantity):
            print('%s not converted'%key)

    return 


def loadStellarArrayMassBins():
    """ load the histogram bin edges for the stellar array """
    with h5py.File('/virgotng/mpia/MCST/arepo1_setups/stellar_properties_old.hdf5', 'r') as f:
        r = f['Masses'][:]
        f.close()
    return r


def validateDicInputs(dic, req_keys, func):
    """ Validate that all required keys are in the dictionary for the given function. """

    if not all(key in dic for key in req_keys):
        print('Keys', req_keys, 'must be provided for %s().'%func.__name__)
        return 0

    return 1      


def validateHeader(basePath, snapNum, func):
    """ Validate that the Header file exists for func. """

    if not os.path.isfile(il.snapshot.snapPath(basePath, snapNum)):
        print('Header file not found for %s() with basePath %s and snapNum %s.'%(func.__name__, basePath, snapNum))
        return 0
    
    return 1


def computeTemperature(dic, tempSFR=1.0e3*u.K):
    """ 
    Compute gas temperature and add dataset to dic. 
    Requires InternalEnergy and ElectronAbundance datasets, 
    and defaults to set gas with StarFormationRate > 0
    to tempSFR. When not using the eEOS, then set tempSFR=0.
    If astropy units are already included, then computes temperature as u.K.
    Returns dic with added Temperature dataset.
    """

    req_keys = ['InternalEnergy', 'ElectronAbundance']
    if tempSFR is not None:
        req_keys.append('StarFormationRate')
    if not validateDicInputs(dic, req_keys, computeTemperature):
        return

    # define constants
    xh = 0.76 # hydrogen mass fraction
    gamma = 5.0 / 3.0 # adiabatic index

    mu = (4. / (1. + 3. * xh + 4. * xh * dic['ElectronAbundance'])) # mean molecular weight ~ 0.59

    if isinstance(dic['InternalEnergy'], u.quantity.Quantity):
        t = ((gamma - 1.0) * dic['InternalEnergy'] / const.k_B * mu * const.m_p).to('K')
    else:
        t = (((gamma - 1.0) * dic['InternalEnergy'] / const.k_B * mu * const.m_p * code_velocity**2).to('K')).value

    if tempSFR is None:
        t[dic['StarFormationRate'] > 0] = tempSFR

    dic['Temperature'] = t    

    return


def computeCoolingTime(dic, basePath=None, snapNum=None):
    """
    Add CoolingTime dataset to dic. Returns dic with CoolingTime added.
    If astropy units are already attached to dic['Density'] and dic['InternalEnergy'], then 
    basePath and snapNum are not necessary to load the Header.
    Computes the CoolingTime in units of standard_time. 
    """

    if 'Temperature' not in dic.keys():
        dic = computeTemperature(dic)

    req_keys = ['Density', 'ElectronAbundance', 'Temperature', 'GFM_CoolingRate']
    if not validateDicInputs(dic, req_keys, computeCoolingTime):
        return

    # define constants 
    xH = 0.76 # Hydrogen mass fraction

    if isinstance(dic['Density'], u.quantity.Quantity):
        nH = (xH * dic['Density'] / const.m_p).to(u.cm**(-3))
        ne = dic['ElectronAbundance'] * nH
        ni = (1.0 - xH) * nH
        tcool = ((-3./2.) * (ne + ni) * const.k_B * dic['Temperature'] / (ne * ni * dic['GFM_CoolingRate'])).to(standard_time)
    else:
        if not validateHeader(basePath, snapNum, computeCoolingTime):
            return
        Header = loadHeader(basePath, snapNum)
        nH = (xH * dic['Density'] * code_mass / Header['HubbleParam'] / (code_length * Header['Time'] / Header['HubbleParam'])**3 / const.m_p).to(u.cm**(-3))
        ne = dic['ElectronAbundance'] * nH
        ni = (1.0 - xH) * nH
        tcool = (((-3./2.) * (ne + ni) * const.k_B * dic['Temperature'] * u.K / (ne * ni * dic['GFM_CoolingRate'] * u.erg * u.cm**3 / u.s)).to(standard_time)).value

    dic['CoolingTime'] = tcool

    return


def computeCellSizes(dic, basePath=None, snapNum=None):
    """
    Add CellSizes dataset to dic. Returns dic with CellSizes added.
    If astropy units are already attached to dic['Masses'] and dic['Density'], then 
    CellSizes units are standard_length, and basePath and snapNum do not need to be provided.
    If there are no units attached, then basePath and snapNum but be provided to
    load the snapshot Header and compute CellSizes in stnadard_length, assuming TNG code units. 
    Returns dic with added CellSizes dataset.
    """

    req_keys = ['Masses', 'Density']
    if not validateDicInputs(dic, req_keys, computeCellSizes):
        return

    if isinstance(dic['Masses'], u.quantity.Quantity) and isinstance(dic['Density'], u.quantity.Quantity):
        CellSizes = ((dic['Masses'] / (4./3. * np.pi * dic['Density']))**(1./3.)).to(standard_length)
    else:
        if not validateHeader(basePath, snapNum, computeCellSizes):
            return
        Header = loadHeader(basePath, snapNum)
        CellSizes = ((dic['Masses'] * code_mass / Header['HubbleParam'] /
                     (dic['Density'] * 4./3. * np.pi * code_mass / Header['HubbleParam'] / 
                      (code_length * Header['Time'] / Header['HubbleParam'])**3))**(1./3.)).to(standard_length).value
    
    dic['CellSizes'] = CellSizes

    return

def computeGasPressure(dic, basePath=None, snapNum=None):
    """ 
    Add Pressure dataset to dic. Returns dic with Pressure added. 
    Assumes P = (gamma - 1) * rho * u, where gamma = 5/3.
    If astropy units are already attached, then computes Pressure in units of 
    standard_pressure. If no units are attached, then basePath and snapNum
    must be provided to load the Header.
    """
    
    gamma = 5. / 3. 

    req_keys = ['Density', 'InternalEnergy']
    if not validateDicInputs(dic, req_keys, computeGasPressure):
        return
    
    if isinstance(dic['Density'], u.quantity.Quantity):
        p = ((gamma - 1.) * dic['Density'] * dic['InternalEnergy']).to(standard_pressure)
    else:
        if not validateHeader(basePath, snapNum, computeGasPressure):
            return
        Header = loadHeader(basePath, snapNum)
        p = ((gamma - 1.) * (dic['Density'] * code_mass / Header['HubbleParam'] / (code_length * Header['Time'] / Header['HubbleParam'])**3) *
             (dic['InternalEnergy'] * code_velocity**2)).to(standard_pressure).value

    dic['Pressure'] = p

    return


def computeJeansNumber(dic, basePath=None, snapNum=None):
    """ 
    Add JeansNumber to dic. Returns dic with JeansNumber added.
    Assumes Nj = (pi^(5/2) * cs^3 / (6 * G^(3/2) * rho^(1/2)) / m, where cs = sqrt(gamma * (gamma - 1) * u), gamma = 5/3.
    If astropy units are already attached, then computes JeansNumber in units of 
    user-defined standard_JeansNumber (dimensionless_unscaled).
    If no units are attached, then basePath and snapNum must be provided to load the Header.
    """

    required_keys = ['InternalEnergy', 'Masses', 'Density']
    if not validateDicInputs(dic, required_keys, computeJeansNumber):
        return

    gamma = 5./3. # adiabatic index

    if isinstance(dic['InternalEnergy'], u.quantity.Quantity):
        cs = (gamma * (gamma - 1.) * dic['InternalEnergy'])**(1./2.)
        Nj = (np.pi**(5./2.) * cs**3 / (6 * const.G**(3./2.) * dic['Density']**(1./2.)) / dic['Masses']).to(u.dimensionless_unscaled).value * standard_JeansNumber
    else:
        if not validateHeader(basePath, snapNum, computeJeansNumber):
            return
        Header = loadHeader(basePath, snapNum)
        cs = (gamma * (gamma - 1.) * dic['InternalEnergy'] * code_velocity**2)**(1./2.)
        Nj = (np.pi**(5./2.) * cs**3 / 
              (6 * const.G**(3./2.) * 
               (dic['Density'] * code_mass / Header['HubbleParam'] / (code_length * Header['Time'] / Header['HubbleParam'])**3)**(1./2.)) / 
              (dic['Masses'] * code_mass / Header['HubbleParam'])).to(u.dimensionless_unscaled).value

    dic['JeansNumber'] = Nj

    return


def loadMCSTFiles(basePath, snapNum):
    """Create (if not already done) and add the sf_details and sn_details postprocessing files to the simulation"""

    if not validateHeader(basePath, snapNum, loadMCSTFiles):
        return
    
    Header = loadHeader(basePath, snapNum)
    HubbleParam = Header['HubbleParam']

    createMCSTFiles(basePath)    

    dic = {}

    ftypes = ['sn_details', 'sf_details']

    for ftype in ftypes:

        in_direc = os.path.join(os.path.split(basePath)[0], 'postprocessing', ftype)
        in_fname = ftype + '.hdf5'
        with h5py.File(os.path.join(in_direc, in_fname), 'r') as inf:
            Time = inf['Time'][:]
            r = {}
            for key in inf:
                dset = inf[key]
                dset_attrs = dict(dset.attrs.items())
                vals = dset[:]
                # if the dataset is higher dimensional, then use the transpose for the vector multiplication with a_scaling
                if len(dset.shape) > 1:
                    vals = vals.T
                r[key] = vals * (Time**(dset_attrs['a_scaling']) * HubbleParam**(dset_attrs['h_scaling']) * 
                                code_mass**(dset_attrs['mass_scaling']) * code_length**(dset_attrs['length_scaling']) * code_velocity**(dset_attrs['velocity_scaling'])) 
                if len(dset.shape) > 1:
                    r[key] = r[key].T
            inf.close()

        # convert units
        convertSnapshotUnits(basePath, snapNum, r)

        dic[ftype] = r

    return dic


def createSnapTimes(basePath):
    """
    Create and save postprocessing file snaptimes.hdf5.
    Creates a mapping between SnapNum, Redshift, Time (scale factor), 
    and CosmicTime, where the Header is loaded at each snapshot.
    
    """
        
    # check if file already exists
    out_fname = os.path.join(Path(basePath).parent, 'postprocessing', 'snaptimes.hdf5')
    if os.path.isfile(out_fname):
        return

    # find all snapshot output files
    snap_fnames = []
    for name in os.listdir(basePath):
        if 'snapdir' in name:
            snap_fnames.append(name)
    snap_fnames.sort()

    # initialize result dictionary
    r = {}
    r['SnapNum'] = np.zeros(len(snap_fnames), dtype=int) - 1
    r['Redshift'] = np.zeros(len(snap_fnames), dtype=float) - 1.0
    r['Time'] = r['Redshift'].copy()
    r['CosmicTime'] = r['Redshift'].copy()

    # loop over snapshots and fill in result dictionary
    for i, snap_fname in enumerate(snap_fnames):
        snapNum = int(snap_fname[-3:])
        Header = loadHeader(basePath, snapNum)
        r['SnapNum'][i] = snapNum
        r['Redshift'][i] = Header['Redshift']
        r['Time'][i] = Header['Time']
        r['CosmicTime'][i] = calcCosmicTime(basePath)

    # check that all entries are filled
    for key in r:
        assert r[key].all() != -1, 'Error: %s'%key

    # write hdf5 file    
    with h5py.File(out_fname, 'w') as f:
        for key in r:
            f.create_dataset(key, data=r[key])
        f.close()
    
    return 


def loadSnapTimes(basePath):
    """Load postprocessing file with snapshot times"""

    fname = os.path.join(Path(basePath).parent, 'postprocessing', 'snaptimes.hdf5')
    if not os.path.isfile(fname):
        createSnapTimes(basePath)
    with h5py.File(fname, 'r') as f:
        r = {}
        for key in f:
            r[key] = f[key][()]
        f.close()
    return r


def calcCosmicTime(basePath):
    """
    Compute the Cosmic Time given the cosmological parameters in the Header, which
    is loaded using basePath and assuming snapNum=0. If not all necessary parameters
    (HubbleParam, Omega0, OmegaBaryon) are in Header, then an exception must be 
    written to manually code the values, which is implemented for Illustris.
    """
    Header = loadHeader(basePath, 0)
    keys = ['HubbleParam', 'Omega0', 'OmegaBaryon']
    if all([key in Header for key in keys]):
        cosmo = FlatLambdaCDM(H0=Header['HubbleParam'] * 100.0, Om0=Header['Omega0'], Ob0=Header['OmegaBaryon'], Tcmb0=2.73)
    elif 'Illustris' in basePath:
        cosmo = FlatLambdaCDM(H0=Header['HubbleParam'] * 100.0, Om0=Header['Omega0'], Ob0=0.0456, Tcmb0=2.73)
    return cosmo.age(Header['Redshift']).value