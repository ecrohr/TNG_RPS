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
    """
    Class to store all relevant information for a given simulation and sample set.
    """
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

        #self = argparse_Config(self)

        self.zooniverse_keys = [self.ins_key, self.jel_key, self.non_key]
        self.subfindsnapshot_flags = [self.in_tree_key, self.central_key,
                                      self.in_z0_host_key, self.host_m200c_key]
        self.subfind_flags = [self.classified_flag, self.central_z0_flag,
                              self.backsplash_z0_flag, self.backsplash_prev_flag,
                              self.preprocessed_flag]

        self.gas_keys = [self.tot_key, self.gasz0_key, self.nogasz0_key]
    
        # tau dictionary keys
        if self.centrals_flag:
            self.taudict_keys = [self.all_key,
                                 self.clean_key,
                                 self.backsplash_z0_flag]
        else:
            self.taudict_keys = [self.backsplash_prev_flag,
                                 self.preprocessed_flag,
                                 self.clean_key,
                                 self.all_key]
        
        self.basePath = ru.loadbasePath(self.sim)

        self.outdirec = '../Output/%s_subfindGRP/'%self.sim
        if not os.path.isdir(self.outdirec):
            os.system('mkdir %s'%self.outdirec)
        GRPfname, taufname = self.return_fnames()
        self.outfname = self.GRPfname = GRPfname
        self.taufname = taufname

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
        self.dmlowres_ptn =il.util.partTypeNum('dmlowres')
        self.bary_ptns = [self.gas_ptn,
                          self.star_ptn,
                          self.bh_ptn]
                          
        self.Mstar_lolim = return_Mstar_lolim(self)
        self.gas_lolim = return_gas_lolim(self)
        
        # set the subfindIDs and accompanying snapnums of interest
        # first check if the tau_dict already exists
        full_taufname = self.outdirec + self.taufname
        if (self.zooniverse_flag):
            SnapNums_SubfindIDs, SubfindIDs = initialize_zooniverseindices(self)
        elif os.path.isfile(full_taufname):
            # yes, so just load the SubfindID (at z=0) for these
            print('File %s exists. Using SubfindIDs and SnapNums from there.'%(full_taufname))
            with h5py.File(full_taufname, 'r') as f:
                Group = f['Group']
                SubfindIDs = Group['SubfindID'][:]
                SnapNums_SubfindIDs = np.zeros(SubfindIDs.size, dtype=int) + 99
                f.close()
        # does the GRP file exist? If so, then use this file. 
        # Note that this assumes that the branches exist at z=0
        elif os.path.isfile(self.outdirec + self.GRPfname):
            print('File %s exists. Using SubfindIDs and SnapNums from there.'%(self.outdirec + self.GRPfname))
            with h5py.File(self.outdirec + self.GRPfname) as f:
                SubfindIDs = np.zeros(len(f.keys()), dtype=int) - 1
                SnapNums_SubfindIDs = SubfindIDs.copy() + 100
                for i, key in enumerate(f.keys()):
                    group = f[key]
                    SubfindIDs[i] = group['SubfindID'][0]
                f.close()
            if np.min(SubfindIDs) < 0:
                raise ValueError('Not all SubfindIDs are >=0 at z=0.')
        # no tau or GRP files, so initialize SnapNums and SubfindIDs here
        else:
            # based on the simulation and flags, find the appropriate initialziation function
            print('Files %s and %s do not exist. Initializing SubfindIDs and SnapNums elsewhere.'%(full_taufname,
                                                                                                   self.outdirec + self.GRPfname))
            # TNG-Cluster?
            if (self.TNGCluster_flag):
                SnapNums_SubfindIDs, SubfindIDs = initialize_TNGCluster_subfindindices(self)

            # only groups (+ clusters), all centrals with M200c > 10.**(13)?
            elif ((self.onlygroups_flag) & (self.centrals_flag)):
                SnapNums_SubfindIDs, SubfindIDs = initialize_central_groups_subfindindices(self)

            # all centrals with M200c > 10.**(11.5)?
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

        print('There are %d subhalos of interest.'%SubfindIDs.size)
        
        if self.tracers_flag:
            if self.zooniverse_flag:
                self.tracer_outdirec = '/vera/ptmp/gc/reric/Output/%s_tracers_zooniverse/'%(self.sim)
            else:
                self.tracer_outdirec = '../Output/%s_tracers/'%(self.sim)

            if not os.path.isdir(self.tracer_outdirec):
                print('Directory %s does not exist. Creating'%self.tracer_outdirec)
                os.system('mkdir %s'%self.tracer_outdirec)
            else:
                print('Directory %s exists. Potentially overwriting files.'%self.tracer_outdirec)
    
                
        return
    
    def return_fnames(self, subsample=None):
        """ Determine and return the appropriate GRP and tau filenames. """
        # input validation
        subsamples = self.taudict_keys
        if subsample and subsample not in subsamples:
            raise ValueError('given subsample %s not accepted.'%subsample) 
        
        if self.zooniverse_flag:
            sample = 'zooniverse'
        elif self.onlygroups_flag and self.centrals_flag:
            sample = 'central_groups_subfind' 
        elif self.centrals_flag:
            sample = 'central_subfind'
        elif self.allsubhalos_flag:
            sample = 'all_subfind'
        else:
            sample = 'subfind'
            
        sim = self.sim
        
        GRPfname = '%s_%s_branches'%(sample, sim)
        taufname = '%s_%s_tau'%(sample, sim)
        if (self.min_snap == self.max_snap):
            if (self.min_snap == 99):
                GRPfname += '_z0'
                taufname += '_z0'
            else:
                GRPfname += '_snapNum%03d'%self.min_snap
                taufname += '_snapNum%03d'%self.min_snap

        if not subsample:
            GRPfname += '.hdf5'
            taufname += '_all.hdf5'
        else:
            GRPfname += '_%s.hdf5'%subsample
            taufname += '_%s.hdf5'%subsample

        return GRPfname, taufname


def argparse_Config(Config):
    """
    Parse all command line arguments and update them in the Config.
    """
    description = ('Pipeline for running all analysis scripts related to \n' +
                   'computing the RPS in TNG galaxies.')
    parser = argparse.ArgumentParser(description=description)
    
    # general flags
    parser.add_argument('--sim', default=None, type=str,
                        help='which simulation to use for the analysis.')
    parser.add_argument('--TNGCluster-flag', action=argparse.BooleanOptionalAction,                   
                        help='analysis for TNG-Cluster')
    parser.add_argument('--zooniverse-flag', action=argparse.BooleanOptionalAction,                   
                        help='analysis using CJF zooniverse results')
    parser.add_argument('--centrals-flag', action=argparse.BooleanOptionalAction,                   
                        help='analysis for central galaxies')
    parser.add_argument('--tracers-flag', action=argparse.BooleanOptionalAction,                   
                        help='use tracer post-processing catalogs in analysis.')
    parser.add_argument('--allsubhalos-flag', action=argparse.BooleanOptionalAction,                   
                        help='use all subhalos in the simulation.')
    # mp flags
    parser.add_argument('--mp-flag', action=argparse.BooleanOptionalAction,                   
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
    parser.add_argument('--SubfindIndices', action=argparse.BooleanOptionalAction,                   
                        help='flag to run Create_SubfindIndices.py.')
    parser.add_argument('--SubfindGasRadProf', action=argparse.BooleanOptionalAction,                   
                        help='flag to run Create_SubfindGasRadProf.py.')
    parser.add_argument('--run-SGRP', action=argparse.BooleanOptionalAction,                   
                        help='flag to run the main analysis in SubfindGasRadProf.py.')
    parser.add_argument('--run-SGRP-PP', action=argparse.BooleanOptionalAction,                   
                        help='flag to run the post processing of SGRP.')
    parser.add_argument('--SubfindSnapshot', action=argparse.BooleanOptionalAction,                   
                        help='flag to run Create_SubfindSnapshot_Flags.py.')
    parser.add_argument('--run-SS', action=argparse.BooleanOptionalAction,                   
                        help='flag to run main analysis in Create_SS_Flags.py.')
    parser.add_argument('--run-SS-PP', action=argparse.BooleanOptionalAction,                   
                        help='flag to run post-processing of Create_SS_Flags.py.')
    parser.add_argument('--TracerTracks', action=argparse.BooleanOptionalAction,                   
                        help='flag to run Create_TracerTracks.py.')
    parser.add_argument('--track-tracers', action=argparse.BooleanOptionalAction,                   
                        help='flag to run track_tracers().')
    parser.add_argument('--find-tracers', action=argparse.BooleanOptionalAction,                   
                        help='flag to run find_unmatched_tracers().')
    parser.add_argument('--CleanSubfindGasRadProf', action=argparse.BooleanOptionalAction,                   
                        help='flag to run Clean_SubfindGasRadProf.py.')
    parser.add_argument('--run-cleanSGRP', action=argparse.BooleanOptionalAction,                   
                        help='flag to run clean_subfindGRP().')
    parser.add_argument('--run-createtau', action=argparse.BooleanOptionalAction,                   
                        help='flag to run create_taudict().')

    args = vars(parser.parse_args())
    for key in args.keys():
        if args[key]:
            Config[key] = args[key]
        
    return Config
                        

def return_outdirec_outfname(Config):
    """
    given the simulation and flags, determine the directory and GRP filename
    """
    
    outdirec = '../Output/%s_subfindGRP/'%Config.sim
    if (Config.zooniverse_flag):
        outfname = 'zooniverse_%s_%s_branches.hdf5'%(Config.sim, Config.zooniverse_key)
        #outdirec = '../Output/zooniverse/'
    elif Config.centrals_flag:
        outfname = 'central_subfind_%s_branches.hdf5'%(Config.sim)
    elif Config.allsubhalos_flag:
        outfname = 'all_subfind_%s_branches.hdf5'%(Config.sim)
    else:
        outfname = 'subfind_%s_branches.hdf5'%(Config.sim)

    # if only caring about z=0 data, then name accordingly
    if (Config.max_snap == Config.min_snap):
        if (Config.min_snap == 99):
            outfname = outfname[:-5] + '_z0.hdf5'
        else:
            outfname = outfname[:-5] + '_snapNum%03d'%(Config.min_snap)

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
        return 'central_subfind_%s_tau.hdf5'%(Config.sim)
    elif Config.allsubhalos_flag:
        return 'all_subfind_%s_clean_tau.hdf5'%(Config.sim)
    else:
        return 'subfind_%s_tau.hdf5'%(Config.sim)

        
def return_Mstar_lolim(Config):
    """ given the simulation, determine the minimum resolved Mstar mass"""
    sim = Config.sim
    if 'TNG50' in sim:
        res = 10.**(8.3)
    elif 'TNG100' in sim:
        res = 10.**(9.5)
    elif 'TNG300' in sim:
        res = 10.**(9)
    elif 'L680n8192TNG' in sim:
        if Config.min_snap == Config.max_snap == 99:
            return 10.**(9)
        else:
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

def return_gas_lolim(Config):
    """ 
    given the simulation, determine the minimum gas resolution, ~1.5 dex below the
    target resolution mass, to fill in when the simulated gas mass is 0.
    """
    sim = Config.sim
    if 'TNG50' in sim:
        res = 1.0e3
    elif 'TNG100' in sim:
        res = 1.0e5
    elif 'TNG300' in sim:
        res = 10.**(5.5)
    elif 'L680n8192TNG' in sim:
        return 10.**(5.5)
    else:
        raise ValueError('sim %s not recognized.'%sim)
    
    for i in range(1,5):
        if '-%d'%i in sim:
            res *= 8**(i-1)
            break
        elif i == 4:
            raise ValueError('sim %s not recongized.'%sim)
            
    return res

 

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

def initialize_central_groups_subfindindices(Config):
    """
    The same as initialize_central_subfindindices(), but
    only considering MPBs with M200c(z=0) > 10^13 Msun.
    """

    halo_fields = ['Group_M_Crit200','GroupFirstSub']
    halos = il.groupcat.loadHalos(Config.basePath, 99, fields=halo_fields)
    M200c = halos['Group_M_Crit200'] * 1.0e10 / Config.h
    indices = M200c >= 10.0**(13.)

    GroupFirstSub = halos['GroupFirstSub']
    SubfindIDs = GroupFirstSub[indices]
    SnapNums = np.ones(SubfindIDs.size, dtype=int) * 99

    return SnapNums, SubfindIDs


def initialize_central_subfindindices(Config):
    """
    Define SubfindIDs and SnapNums to be tracked.
    Returns the most massive central subhalos, default
    to z=0. If only considering one snapshot, that is, 
    Config.min_snap == Config.max_snap, then considers
    centrals above the M200c limit at this snapshot.
    """ 

    if Config.min_snap == Config.max_snap:
        snapNum = Config.min_snap
    else:
        snapNum = 99

    if 'TNG300' in Config.sim:
        M200c_lowlim = 10.**(12)
    elif 'TNG100' in Config.sim:
        M200c_lowlim = 10.**(11.5)
    elif 'TNG50' in Config.sim:
        M200c_lowlim = 1.0e11
    
    halo_fields = ['Group_M_Crit200','GroupFirstSub']
    halos = il.groupcat.loadHalos(Config.basePath, snapNum, fields=halo_fields)
    M200c = halos['Group_M_Crit200'] * 1.0e10 / Config.h
    indices = M200c >= M200c_lowlim

    GroupFirstSub = halos['GroupFirstSub']
    SubfindIDs = GroupFirstSub[indices]
    SnapNums = np.ones(SubfindIDs.size, dtype=int) * snapNum

    print('snapNum = %d'%snapNum)

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

    massratio_frac = Config.massratio_frac
    
    # load all halos and find the primary zoom target IDs
    halo_fields = ['GroupFirstSub', 'GroupPrimaryZoomTarget']
    halos = il.groupcat.loadHalos(basePath, max_snap, fields=halo_fields)
    haloIDs = np.where(halos['GroupPrimaryZoomTarget'] == 1)[0]
    GroupFirstSub = halos['GroupFirstSub'][haloIDs]
    
    print('There are %d primary zoom targets in %s.'%(haloIDs.size, Config.sim))
    assert haloIDs.size == 352, "Error:"
    
    # load all subhalos and find which ones:
    # 1) are z=0 satellites of primary zooms
    # 2) have Mstar(z=0) > Mstar_lolim
    # 3) have M_star^sat / M_star^host (z=0) < massratio_frac
    # 4) have SubhaloFlag == True
    subhalo_fields = ['SubhaloGrNr', 'SubhaloMassInRadType', 'SubhaloFlag']
    subhalos = il.groupcat.loadSubhalos(basePath, max_snap, fields=subhalo_fields)
    subhalo_indices_massive = subhalos['SubhaloMassInRadType'][:,star_ptn] * 1.0e10 / h > Mstar_lolim
    
    _, subhalo_match_indices = ru.match3(haloIDs, subhalos['SubhaloGrNr'][subhalo_indices_massive])
    
    # remove the central galaxies
    subhaloIDs = np.where(subhalo_indices_massive)[0][subhalo_match_indices]
    isin = np.isin(subhaloIDs, GroupFirstSub, assume_unique=True)
    
    satellite_subhaloIDs = subhaloIDs[~isin]
    central_subhaloIDs = GroupFirstSub 

    if centrals_flag:
        snaps = np.ones(central_subhaloIDs.size, dtype=central_subhaloIDs.dtype) * max_snap
        return snaps, central_subhaloIDs
    
    lowratio_subfindIDs = []
    for halo_i, haloID in enumerate(haloIDs): 

        satellite_indices = subhalos['SubhaloGrNr'][satellite_subhaloIDs] == haloID
        satelliteIDs = satellite_subhaloIDs[satellite_indices]
        Mstar_host = subhalos['SubhaloMassInRadType'][central_subhaloIDs[halo_i],star_ptn]
        Mstar_sats = subhalos['SubhaloMassInRadType'][satelliteIDs,star_ptn]
        lowratio_indices = (Mstar_sats / Mstar_host) < massratio_frac

        lowratio_subfindIDs.append(satelliteIDs[lowratio_indices])

    lowratio_subfindIDs = np.concatenate(lowratio_subfindIDs)
    subhalo_flag_indices = subhalos['SubhaloFlag'][lowratio_subfindIDs] == True
    subfindIDs = lowratio_subfindIDs[subhalo_flag_indices]
    snaps = np.ones(subfindIDs.size, dtype=subfindIDs.dtype) * max_snap
    
    return snaps, subfindIDs

    
def initialize_zooniverseindices(Config):
    """
    Load all zooniverse output catalogs for the given simulation and determine
    which galaxies have been inspected at multiple snapshots. Then tabulates the last
    snapshot at which the galaxy was inspected, and the subfindID at that snapshot.
    Returns SnapNums, SubfindIDs
    """
    
    indirec = '../IllustrisTNG/%s/postprocessing/Zooniverse_CosmologicalJellyfish/'%Config.sim
    infname = 'jellyfish.hdf5'
    
    with h5py.File(indirec + infname, 'r') as inf:
        snapnums = inf['Branches_SnapNum_LastInspect'][:]
        subfindIDs = inf['Branches_SubfindID_LastInspect'][:]
        
        inf.close()
    
    return snapnums, subfindIDs


def str2bool(v):
    """
    functionality for bool arguments to be passed via command line.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

    
fname = 'config.yaml'
config_dict = Configuration.from_yaml(fname)
Config = Configuration(config_dict)
Config.add_vals()

print(Config)

#
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
        
# run the tracer analysis
if Config.TracerTracks:
    from Create_TracerTracks import create_tracertracks
    create_tracertracks(Config)

# clean the subfind GRP dictionary
if Config.CleanSubfindGasRadProf:
    from Clean_SubfindGasRadProf import run_clean_zooniverseGRP
    run_clean_zooniverseGRP(Config)
