#!/usr/bin/env python3

# wrapper script to run all analysis related to tracking subhalos across time.

### import modules
import illustris_python as il
import multiprocessing as mp
import numpy as np
import h5py
import rohr_utils as ru
import os
import yaml

# import configuration
class Configuration(dict):
    @classmethod
    def from_yaml(cls, fname):
        """ Load from a yaml file. """
        with open(fname, 'r') as fin:
            config = yaml.load(fin, yaml.SafeLoader)
        return cls(config)
    
    @classmethod
    def from_str(cls, text: str):
        """ Load from yaml-like string """
        config = yaml.load(text, yaml.SafeLoader)
        return cls(config)
    
    def __getattr__(self, key):
        """ Direct access as an attribute """
        return self[key]
    
    def add_vals(self):
        """ Add additional attributes """
        
        self.basePath = ru.loadbasePath(self.sim)
        self.outdirec, self.outfname = return_outdirec_outfname(self)
        
        self.Header = il.groupcat.loadHeader(self.basePath, self.max_snap)
        self.h = self.Header['HubbleParam']
        
        SnapNums = np.arange(self.max_snap, self.min_snap-1, -1)
        Times = np.zeros(SnapNums.size, dtype=float)
        BoxSizes = Times.copy()
        for i, SnapNum in enumerate(SnapNums):
            header = il.groupcat.loadHeader(self.basePath, SnapNum)
            Times[i] = header['Time']
            BoxSizes[i] = header['BoxSize'] * Times[i] / self.h
        self.SnapNums = SnapNums
        self.Times = Times
        self.BoxSizes = BoxSizes
        
        self.gas_ptn = il.util.partTypeNum('gas')
        self.dm_ptn = il.util.partTypeNum('dm')
        self.tracer_ptn = il.util.partTypeNum('tracer')
        self.star_ptn = il.util.partTypeNum('star')
        self.bh_ptn = il.util.partTypeNum('bh')
        self.bary_ptns = [self.gas_ptn,
                          self.star_ptn,
                          self.bh_ptn]
                          
        self.Mstar_lolim = return_Mstar_lolim(Config)
        
        return


def return_outdirec_outfname(Config):
    """
    given the simulation and flags, determine the directory and filename
    """
    
    if (Config.zooniverse_flag):
        ins_key = 'inspected'
        outfname = 'zooniverse_%s_%s_branches.hdf5'%(Config.sim, ins_key)
        outdirec = '../Output/%s_subfindGRP/'%Config.sim
        
        if (os.path.isdir(outdirec)):
            print('Directory %s exists.'%outdirec)
            if os.path.isfile(outdirec + outfname):
                print('File %s exists. Overwriting.'%(outdirec+outfname))
                return outdirec, outfname
            else:
                print('File %s does not exists. Writing.'%(outdirec+outfname))
                return outdirec, outfname
        else:
            print('Directory %s does not exist. Creating it now.'%outdirec)
            os.system('mkdir %s'%outdirec)
            return outdirec, outfname
            
    if Config.centrals_flag:
        outfname = 'central_subfind_%s_branches.hdf5'%(Config.sim)
    else:
        outfname = 'subfind_%s_branches.hdf5'%(Config.sim)
    
    outdirec = '../Output/%s_subfindGRP/'%Config.sim
    
    if (os.path.isdir(outdirec)):
        print('Directory %s exists.'%outdirec)
        if os.path.isfile(outdirec + outfname):
            print('File %s exists. Overwriting.'%(outdirec+outfname))
            return outdirec, outfname
        else:
            print('File %s does not exists. Writing.'%(outdirec+outfname))
            return outdirec, outfname
    else:
        print('Directory %s does not exist. Creating it now.'%outdirec)
        os.system('mkdir %s'%outdirec)
        return outdirec, outfname
        
        
def return_Mstar_lolim(Config):
    """ given the simulation, determine the minimum resolved Mstar mass"""
    sim = Config.sim
    if 'TNG50' in sim:
        res = 10.**(8.3)
    elif 'TNG100' in sim:
        res = 10.**(9.5)
    elif 'TNG300' in sim:
        res = 10.**(10.)
    elif 'L680n8192TNG' == sim:
        return 10**(10)
    else:
        raise ValueError('sim %s not recognized.'%sim)
    
    for i in range(1,5):
        if '-%d'%i in sim:
            Mstar_lolim = res * 8**i
        elif i == 4:
            raise ValueError('sim %s not recongized.'%sim)
            
    return Mstar_lolim
    
            
    
config_dict = Configuration.from_yaml('config.yaml')
Config = Configuration(config_dict)
Config.add_vals()

# create the indices
if Config.SubfindIndices:
    from Create_SubfindIndices import run_subfindindices
    run_subfindindices()

# create the snapshot flags
if Config.SubfindSnapshot:
    from Create_SubfindSnapshot_Flags import run_subfindsnapshot_flags
    run_subfindsnapshot_flags(Config)

# run the gas radial profile calculation
if Config.SubfindGasRadProf:
    from Create_SubfindGasRadProf import run_subfindGRP
    run_subfindGRP()

