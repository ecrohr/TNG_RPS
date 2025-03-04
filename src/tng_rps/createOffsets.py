import illustris_python as il
import numpy as np
import os
import h5py
import glob

int_dtype = np.int64

def createOffsets(basePath, snapNum):
    """ compute and save the offsets files for the given simulation. """
    r = computeOffsets(basePath, snapNum)
    saveOffsets(basePath, snapNum, r)
    return


def computeOffsets(basePath, snapNum):
    """
    Create the offsets file for the given simulation located at basePath at snapNum.
    Per usual, basePath should point to the output directory of the simulation.
    Returns the offsets file as a dictionary.
    NB: This is only useful for the il.groupcat and il.snapshot modules, not, e.g, sublink. 
    """

    Groups = il.groupcat.loadHalos(basePath, snapNum, fields=['GroupFirstSub', 'GroupLenType'])
    Subhalos = dict(SubhaloLenType=il.groupcat.loadSubhalos(basePath, snapNum, fields=['SubhaloLenType']))

    Nparts = Groups['GroupLenType'].shape[1]

    r = {}
    r['Group'] = {}
    r['Subhalo'] = {}
    r['FileOffsets'] = {}

    # group lengths are relatively trivial
    Groups_GroupLenType_cumsum = np.vstack((np.zeros(Nparts, dtype=int_dtype), np.cumsum(Groups['GroupLenType'], axis=0)[:-1]))
    r['Group']['SnapByType'] = np.vstack((np.zeros(Nparts, dtype=int_dtype), np.cumsum(Groups['GroupLenType'], axis=0)[:-1]))

    # subhalo lengths are more complicated due to the inner fuzz
    # start by computing the cum sum just like for the groups
    Subhalos_SubhaloLenType_cumsum = np.vstack((np.zeros(Nparts, dtype=int_dtype), np.cumsum(Subhalos['SubhaloLenType'], axis=0)[:-1]))
    r['Subhalo']['SnapByType'] = np.vstack((np.zeros(Nparts, dtype=int_dtype), np.cumsum(Subhalos['SubhaloLenType'], axis=0)[:-1]))

    # starting with the first subahlo in the second group, compute and add the inner fuzz to all following subhalos
    # find the first subhalo in the second group using GroupFirstSub
    for GroupFirstSub_i, GroupFirstSub in enumerate(Groups['GroupFirstSub'][1:]):
        # if there are no subhalos in the group, then skip
        if GroupFirstSub < 0:
            continue
        Groups_StartType = Groups_GroupLenType_cumsum[GroupFirstSub_i+1]
        # compute the inner fuzz by subtracting the (offset of the second group)
        # from the ((offset of the last subhalo of the first group) plus (length of the last subhalo of the first group))
        InnerFuzzType = Groups_StartType - (r['Subhalo']['SnapByType'][GroupFirstSub-1] + Subhalos['SubhaloLenType'][GroupFirstSub-1])
        r['Subhalo']['SnapByType'][GroupFirstSub:] += InnerFuzzType


    # find all snapshot chunks at the given snapNum
    path = os.path.join(basePath, 'snapdir_%03d'%snapNum)
    Nchunks = len(glob.glob(os.path.join(path, 'snap_%03d.*.hdf5'%snapNum)))  # could also be read from the header of the first file

    # initialize output array
    FileOffsets_NumPart_ThisFile = np.zeros((Nchunks, Subhalos_SubhaloLenType_cumsum.shape[1]), dtype=int_dtype) 

    # loop over chunks and save the number of particles per type per chunk
    for chunk_i in range(Nchunks - 1):
        with h5py.File(os.path.join(path, 'snap_%03d.%d.hdf5'%(snapNum, chunk_i)), 'r') as snap:
            header = dict(snap['Header'].attrs.items())
            FileOffsets_NumPart_ThisFile[chunk_i+1] = header['NumPart_ThisFile'][:]

    # compute the offsets via the cumulative sum
    r['FileOffsets']['SnapByType'] = np.cumsum(FileOffsets_NumPart_ThisFile, axis=0)

    # find all group/subhalo chunks at the given snapNum
    # could be combined with the previous section for the snapshot chunks, but left separate in case there are differing numbers of chunks 
    # otherwise the structure is the same as above
    path = os.path.join(basePath, 'groups_%03d'%snapNum)
    Nchunks = len(glob.glob(os.path.join(path, 'fof_subhalo_tab_%03d.*.hdf5'%snapNum)))

    FileOffsets_Ngroups_ThisFile = np.zeros(Nchunks, dtype=int_dtype) 
    FileOffsets_Nsubgroups_ThisFile = FileOffsets_Ngroups_ThisFile.copy()

    for chunk_i in range(Nchunks - 1):
        with h5py.File(os.path.join(path, 'fof_subhalo_tab_%03d.%d.hdf5'%(snapNum, chunk_i)), 'r') as groups:
            header = dict(groups['Header'].attrs.items())
            FileOffsets_Ngroups_ThisFile[chunk_i+1] = header['Ngroups_ThisFile']
            FileOffsets_Nsubgroups_ThisFile[chunk_i+1] = header['Nsubgroups_ThisFile']

    # compute the offsets via the cumulative sum
    r['FileOffsets']['Group'] = np.cumsum(FileOffsets_Ngroups_ThisFile)
    r['FileOffsets']['Subhalo'] = np.cumsum(FileOffsets_Nsubgroups_ThisFile)

    return r


def saveOffsets(basePath, snapNum, offsets):
    """
    By default, the offsets file is saved in the postprocessing/offsets directory
    with the file naming convention offsets_%03d.hdf5%snapNum. 
    """
    path = os.path.join(os.path.split(basePath)[0], 'postprocessing', 'offsets')
    if not os.path.isdir(path):
        os.makedirs(path)
    out_fname = os.path.join(path, 'offsets_%03d.hdf5'%snapNum)

    # check if file already exists
    if os.path.isfile(out_fname):
        print('File %s already exists. Not overwriting.'%out_fname)
        return
    
    with h5py.File(out_fname, 'w') as f:
        for key in offsets:
            group = f.create_group(key)
            for subkey in offsets[key]:
                group.create_dataset(subkey, data=offsets[key][subkey])
        f.close()
    return


def testOffsets(basePath='/virgotng/universe/IllustrisTNG/L35n270TNG/output', snapNum=99):
    """Test the offsets file creation against a known file. Default is TNG50-4."""

    offsets = h5py.File(os.path.join(os.path.split(basePath)[0], 'postprocessing', 'offsets', 'offsets_%03d.hdf5'%snapNum), 'r')
    r = computeOffsets(basePath, snapNum)
    assert(np.all(r['Group']['SnapByType'] == offsets['Group']['SnapByType']))
    assert(np.all(r['Subhalo']['SnapByType'] == offsets['Subhalo']['SnapByType']))
    assert(np.all(r['FileOffsets']['SnapByType'] == offsets['FileOffsets']['SnapByType'][()]))
    assert(np.all(r['FileOffsets']['Group']  == offsets['FileOffsets']['Group'][()]))
    assert(np.all(r['FileOffsets']['Subhalo'] == offsets['FileOffsets']['Subhalo'][()]))    

    print('Passed all tests! Happy Coding :)')

    return
