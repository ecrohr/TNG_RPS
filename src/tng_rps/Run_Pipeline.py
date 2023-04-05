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
import glob
import argparse

class Configuration(dict):
    __slots__ = ()
    
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
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)
            
    def __setattr__(self, key, value):
        """ Set additional attributes. """
        self[key] = value
        return


    def add_vals(self):
        """ Add additional attributes """
        
        self = argparse_Config(self)
        
        self.basePath = ru.loadbasePath(self.sim)
        self.outdirec, self.outfname = return_outdirec_outfname(self)
        # for backwards compatibility
        self.GRPfname = self.outfname
        self.taufname = return_taufname(self)
        
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
        zs, cosmictimes = ru.timesfromsnap(self.basePath, SnapNums)
        self.Redshifts = zs
        self.CosmicTimes = cosmictimes / 1.0e9 # Gyr
        
        self.gas_ptn = il.util.partTypeNum('gas')
        self.dm_ptn = il.util.partTypeNum('dm')
        self.tracer_ptn = il.util.partTypeNum('tracer')
        self.star_ptn = il.util.partTypeNum('star')
        self.bh_ptn = il.util.partTypeNum('bh')
        self.bary_ptns = [self.gas_ptn,
                          self.star_ptn,
                          self.bh_ptn]
                          
        self.Mstar_lolim = return_Mstar_lolim(self)
        
        # set the subfindIDs and accompanying snapnums of interest
        # first check if the tau_dict already exists
        full_taufname = self.outdirec + self.taufname
        if os.path.isfile(full_taufname):
            # yes, so just load the SubfindID (at z=0) for these
            print('File %s exists. Using SubfindIDs and SnapNums from there.'%(full_taufname))
            with h5py.File(full_taufname, 'r') as f:
                Group = f['Group']
                SubfindIDs = Group['SubfindID'][:]
                SnapNums_SubfindIDs = np.zeros(SubfindIDs.size, dtype=int) + 99
                f.close()
        else:
            # based on the simulation and flags, find the appropriate initialziation function
            print('File %s does not exist. Initializing SubfindIDs and SnapNums elsewhere.'%(full_taufname))
            # using the zooniverse results?
            if (self.zooniverse_flag):
                SnapNums_SubfindIDs, SubfindIDs = initialize_zooniverseindices(self)
                
            # TNG-Cluster?
            elif (self.TNGCluster_flag):
                SnapNums_SubfindIDs, SubfindIDs = initialize_TNGCluster_subfindindices(self)

            # all centrals with M200c > 10.**(11.15)?
            elif (self.centrals_flag):
                SnapNums_SubfindIDs, SubfindIDs = initialize_central_subfindindices(self)
                              
            # all subhalos?
            elif (self.allsubhalos_flag):
                SnapNums_SubfindIDs, SubfindIDs = initialize_allsubhalos(self)
              
            # general satellites?
            else:
                SnapNums_SubfindIDs, SubfindIDs = initialize_subfindindices(self)
                
                
        self.SubfindIDs = SubfindIDs
        self.SnapNums_SubfindIDs = SnapNums_SubfindIDs

        if self.tracers_flag:
            if self.zooniverse_flag:
                self.tracer_outdirec = '../Output/%s_tracers_zooniverse/'%(self.sim)
            else:
                self.tracer_outdirec = '../Output/%s_tracers/'%(self.sim)

            if not os.path.isdir(self.tracer_outdirec):
                print('Directory %s does not exist. Creating'%self.tracer_outdirec)
                os.system('mkdir %s'%self.tracer_outdirec)
            else:
                print('Directory %s exists. Potentially overwriting files.'%self.tracer_outdirec)
                
        self.taudict_keys = [self.ins_key, self.jel_key, self.non_key]
                
        
        return


def argparse_Config(Config):
    """
    Parse all command line arguments and update them in the Config.
    """
    description = ('Pipeline for running all analysis scripts related to \n' +
                   'computing the RSP in TNG galaxies.')
    parser = argparse.ArgumentParser(description=description)
    
    
    # general flags
    parser.add_argument('--sim', default=None, type=str,
                        help='which simulation to use for the analysis.')
    
    parser.add_argument('--TNGCluster-flag', default=None, type=bool,
                        help='analysis for TNG-Cluster')
    parser.add_argument('--zooniverse-flag', default=None, type=bool,
                        help='analysis using CJF zooniverse results')
    parser.add_argument('--centrals-flag', default=None, type=bool,
                        help='analysis for central galaxies')
    parser.add_argument('--tracers-flag', default=None, type=bool,
                        help='use tracer post-processing catalogs in analysis.')
    parser.add_argument('--allsubhalos-flag', default=None, type=bool,
                        help='use all subhalos in the simulation.')
    # mp flags
    parser.add_argument('--mp-flag', default=None, type=bool,
                        help='use multiprocessing for analysis.')
    parser.add_argument('--Nmpcores', default=None, type=int,
                        help='Number of cores to use for multiprocessing tasks.')

    # hard coded values
    parser.add_argument('--max-snap', default=None, type=int,
                        help='max snap for merger trees and analysis.')
    parser.add_argument('--min-snap', default=None, type=int,
                        help='min snap for merger trees and analysis.')
    parser.add_argument('--first-snap', default=None, type=int,
                        help='first snap for running the tracers.')
    parser.add_argument('--last-snap', default=None, type=int,
                        help='last snap for running the tracers.')
    parser.add_argument('--tlim', default=None, type=float,
                        help='temperature limit between cold and hot gas.')
    parser.add_argument('--jellyscore-min', default=None, type=int,
                        help='minimum score to be considered a jellyfish galaxy')

    # which types of analysis should be run
    parser.add_argument('--SubfindIndices', default=None, type=bool,
                        help='flag to run Create_SubfindIndices.py.')
    parser.add_argument('--SubfindGasRadProf', default=None, type=bool,
                        help='flag to run Create_SubfindGasRadProf.py.')
    parser.add_argument('--run-SGRP', default=None, type=bool,
                        help='flag to run the main analysis in SubfindGasRadProf.py.')
    parser.add_argument('--run-SGRP-PP', default=None, type=bool,
                        help='flag to run the post processing of SGRP.')
    parser.add_argument('--SubfindSnapshot', default=None, type=bool,
                        help='flag to run Create_SubfindSnapshot_Flags.py.')
    parser.add_argument('--run-SS', default=None, type=bool,
                        help='flag to run main analysis in Create_SS_Flags.py.')
    parser.add_argument('--run-SS-PP', default=None, type=bool,
                        help='flag to run post-processing of Create_SS_Flags.py.')
    parser.add_argument('--TracerTracks', default=None, type=bool,
                        help='flag to run Create_TracerTracks.py.')
    parser.add_argument('--track-tracers', default=None, type=bool,
                        help='flag to run track_tracers().')
    parser.add_argument('--find-tracers', default=None, type=bool,
                        help='flag to run find_unmatched_tracers().')

    args = vars(parser.parse_args())
    for key in args.keys():
        if args[key]:
            Config[key] = args[key]
        
    return Config
                        

def return_outdirec_outfname(Config):
    """
    given the simulation and flags, determine the directory and GRP filename
    """
    
    if (Config.zooniverse_flag):
        ins_key = 'inspected'
        outfname = 'zooniverse_%s_%s_branches.hdf5'%(Config.sim, Config.zooniverse_key)
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
    elif Config.allsubhalos_flag:
        outfname = 'all_subfind_%s_branches.hdf5'%(Config.sim)
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
        
        
def return_taufname(Config):
    """ given the simulation and flags, determine the tau filename """
    if Config.zooniverse_flag:
        return 'zooniverse_%s_%s_clean_tau.hdf5'%(Config.sim, Config.zooniverse_key)
    elif Config.centrals_flag:
        return 'central_subfind_%s_clean_tau.hdf5'%(Config.sim)
    elif Config.allsubhalos_flag:
        return 'all_subfind_%s_clean_tau.hdf5'%(Config.sim)
    else:
        return 'subfind_%s_clean_tau.hdf5'%(Config.sim)

        
def return_Mstar_lolim(Config):
    """ given the simulation, determine the minimum resolved Mstar mass"""
    sim = Config.sim
    if 'TNG50' in sim:
        res = 10.**(8.3)
    elif 'TNG100' in sim:
        res = 10.**(9.5)
    elif 'TNG300' in sim:
        res = 10.**(10)
    elif 'L680n8192TNG' in sim:
        return 10.**(10)
    else:
        raise ValueError('sim %s not recognized.'%sim)
        
    if sim == 'TNG50-4':
        return 10.**(10)
    
    for i in range(1,5):
        if '-%d'%i in sim:
            Mstar_lolim = res * 8**(i-1)
            break
        elif i == 4:
            raise ValueError('sim %s not recongized.'%sim)
            
    return Mstar_lolim
 

def initialize_allsubhalos(Config):
    """
    Create a list of the subfindIDs for all subhalos in the simulation.
    """
    
    Nsubhalos = il.groupcat.loadSubhalos(Config.basePath, Config.max_snap, fields='SubhaloGrNr').size
    SubfindIDs = np.arange(Nsubhalos)
    SnapNums = np.ones(SubfindIDs.size, dtype=int) * 99
    
    return SnapNums, SubfindIDs
    

def initialize_subfindindices(Config):
    """
    Define SubfindIDs and SnapNums to be tracked.
    Returns all z=0 satellites with Mstar > Mstar_lolim
    and SubhaloFlag == 1.
    Returns SnapNums, SubfindIDs
    """
    
    basePath = Config.basePath
    star_ptn = Config.star_ptn
    Mstar_lolim = Config.Mstar_lolim
    h = Config.h
    
    subhalo_fields = ['SubhaloFlag', 'SubhaloMassInRadType']
    subhalos = il.groupcat.loadSubhalos(basePath, 99, fields=subhalo_fields)

    halo_fields = ['GroupFirstSub']
    GroupFirstSub = il.groupcat.loadHalos(basePath, 99, fields=halo_fields)

    Mstar = subhalos['SubhaloMassInRadType'][:,star_ptn] * 1.0e10 / h
    subfind_indices = np.where((subhalos['SubhaloFlag']) & (Mstar >= Mstar_lolim))[0]
    indices = np.isin(subfind_indices, GroupFirstSub)
    SubfindIDs = subfind_indices[~indices]
    SnapNums = np.ones(SubfindIDs.size, dtype=int) * 99

    return SnapNums, SubfindIDs


def initialize_central_subfindindices(Config):
    """
    Define SubfindIDs and SnapNums to be tracked.
    Returns the most massive z=0 central subhalos.
    """
    
    halo_fields = ['Group_M_Crit200','GroupFirstSub']
    halos = il.groupcat.loadHalos(Config.basePath, 99, fields=halo_fields)
    M200c = halos['Group_M_Crit200'] * 1.0e10 / Config.h
    indices = M200c >= 10.0**(11.5)

    GroupFirstSub = halos['GroupFirstSub']
    SubfindIDs = GroupFirstSub[indices]
    SnapNums = np.ones(SubfindIDs.size, dtype=int) * 99

    return SnapNums, SubfindIDs
    
    
def initialize_TNGCluster_subfindindices(Config):
    """
    Define the SubfindIDs at z=0 to be tracked.
    """
    
    basePath = Config.basePath
    max_snap = Config.max_snap
    star_ptn = Config.star_ptn
    h = Config.h
    Mstar_lolim = Config.Mstar_lolim
    centrals_flag = Config.centrals_flag
    
    # load all halos and find the primary zoom target IDs
    halo_fields = ['Group_M_Crit200', 'GroupFirstSub', 'GroupPrimaryZoomTarget']
    halos = il.groupcat.loadHalos(basePath, max_snap, fields=halo_fields)
    haloIDs = np.where(halos['GroupPrimaryZoomTarget'])[0]
    GroupFirstSub = halos['GroupFirstSub'][haloIDs]
    
    print('There are %d primary zoom targets in %s.'%(haloIDs.size, Config.sim))
    
    # load all subhalos and find which ones:
    # 1) are z=0 satellites of primary zooms
    # 2) have Mstar(z=0) > Mstar_lolim
    subhalo_fields = ['SubhaloGrNr', 'SubhaloMassInRadType']
    subhalos = il.groupcat.loadSubhalos(basePath, max_snap, fields=subhalo_fields)
    subhalo_indices_massive = subhalos['SubhaloMassInRadType'][:,star_ptn] * 1.0e10 / h > Mstar_lolim
    
    _, subhalo_match_indices = ru.match3(haloIDs, subhalos['SubhaloGrNr'][subhalo_indices_massive])
    
    # remove the central galaxies
    subhaloIDs = np.where(subhalo_indices_massive)[0][subhalo_match_indices]
    isin = np.isin(subhaloIDs, GroupFirstSub, assume_unique=True)
    
    if centrals_flag:
        subfindIDs = subhaloIDs[isin]
    else:
        subfindIDs = subhaloIDs[~isin]
        
    snaps = np.ones(subfindIDs.size, dtype=subfindIDs.dtype) * max_snap
    
    return snaps, subfindIDs

    
def initialize_zooniverseindices(Config):
    """
    Load all zooniverse output catalogs for the given simulation and determine
    which galaxies have been inspected at multiple snapshots. Then tabulates the last
    snapshot at which the galaxy was inspected, and the subfindID at that snapshot.
    Returns SnapNums, SubfindIDs
    """

    # load the inspected IDs dictionary
    insIDs_dict = load_zooniverseIDs(Config)

    # create empty lists
    snapnums   = []
    subfindids = []

    # initialize dictionary with empty lists that will hold the SubfindIDs,
    # corresponding to branches that have already been cataloged
    out_dict = {}
    for i in range(0, 100):
        out_dict['%03d'%i] = []
    
    # load in each file consecutively, and check for every done subhalo whether
    # it has already been cataloged. if not, add the key to the empty lists
    # start at snap 99 and work backwards, such that we only need to load the MPBs
    for snap_key in insIDs_dict.keys():

        out_dict_snap = out_dict[snap_key]
        subfindIDs    = insIDs_dict[snap_key]

        for i, subfindID in enumerate(subfindIDs):
            subfindID_key = '%08d'%subfindID

            # check if this subfindID at this snap has already been cataloged
            if subfindID_key in out_dict_snap:
                continue
            
            else:
                # load sublink_gal MPB and tabulate
                fields = ['SnapNum', 'SubfindID']
                MPB    = il.sublink.loadTree(Config.basePath, int(snap_key), subfindID,
                                             fields=fields, onlyMPB=True, treeName=Config.treeName)

                if MPB is None:
                    print('No MPB for %s snap %s subhaloID %s. Continuing'%(sim, snap_key, subfindID))
                    continue

                for j in range(MPB['count']):
                    snapkey       = '%03d'%MPB['SnapNum'][j]
                    subfindid     = MPB['SubfindID'][j]
                    subfindid_key = '%08d'%MPB['SubfindID'][j]
                    out_dict[snapkey].append(subfindid_key)
                # finish loop over the MPB

                snapnums.append(int(snap_key))
                subfindids.append(int(subfindID_key))
                
    # finish loop over the insIDs and save the keys
    snapnums   = np.array(snapnums, dtype=type(snapnums[0]))
    subfindids = np.array(subfindids, dtype=type(subfindids[0]))

    return snapnums, subfindids


def load_zooniverseIDs(Config):
    """
    Load all zooniverse catalogs. Create a dictionary with each snapshot as the key,
    and the entries are the subfindIDs of all inspected galaxies at that snapshot.
    Returns the dictionary.
    """
    
    # load in the filenames for each snapshot, starting at the last snap
    indirec  = '../IllustrisTNG/%s/postprocessing/Zooniverse_CosmologicalJellyfish/flags/'%Config.sim
    infname  = 'cosmic_jellyfish_flags_*.hdf5'
    infnames = glob.glob(indirec + infname)
    infnames.sort(reverse=True)

    # create dictionaries with snapnum as the key and lists of subfindIDs as the entires
    insIDs_dict = {}
    for filename in infnames:
        snap_key = filename[-8:-5]
        f        = h5py.File(filename, 'r')
        done     = f['done'][0]
        Score    = f['Score'][0]
        
        insIDs_dict[snap_key] = np.where(done == 1)[0]

        f.close()
    # finish loop over files
    
    return insIDs_dict

    
fname = 'config.yaml'
config_dict = Configuration.from_yaml(fname)
Config = Configuration(config_dict)
Config.add_vals()

# create the indices
if Config.SubfindIndices:
    from Create_SubfindIndices import run_subfindindices
    run_subfindindices(Config)

# create the snapshot flags
if Config.SubfindSnapshot:
    from Create_SubfindSnapshot_Flags import run_subfindsnapshot_flags
    run_subfindsnapshot_flags(Config)

# run the gas radial profile calculation
if Config.SubfindGasRadProf:
    from Create_SubfindGasRadProf import run_subfindGRP
    run_subfindGRP(Config)
        
    
if Config.TracerTracks:
    from Create_TracerTracks import create_tracertracks
    create_tracertracks(Config)

