"""
stellar_array_helpers.py
Matthew C. Smith 2023
This module contains functions to help read the stellar array used by 
SFR_MCS in combination with IMF_SAMPLING_MCS (but not IMF_SAMPLING_MCS_OLD).
The array is found on the StarP struct as "StellarArray", in snapshots
as "StellarArray" and in the IMF sampling log files
([OutputListFilename]/imf_details/imf_details_[Ntask].txt) as the column
index 15 onwards.

The stellar array is a histogram of the masses of the individual massive stars
hosted in the star particle. The number of bins is N_BINS; by default this
is 64. To save memory, the count in each bin is represented with only
STAR_BIN_SIZE bits. By default, we use STAR_BIN_SIZE = 4. The histogram is
stored as an array of unsigned 64 bit ints (uint64). Thus, it will have
STAR_BIN_SIZE * N_BINS / 64 elements (therefore, by default it has 4 elements).
In order to read the number stored in a bin of the stellar array, we must first
find which uint64 contains the relevant bits, then their position within the
full 64 bit element. This is accomplished via some bitwise operations contained in
the functions below. The equivalents of these functions used in the code can
be found at arepo/src/sfr_mcs/imf_sampling_mcs_utils.c

Example:
Get the stellar arrays for all star particles in a snapshot
then convert to an expanded histogram.
with h5py.File('snap_100.hdf5','r') as f:
	starr = f['PartType4/StellarArray'][:] #For defaults, this has shape (Nstar,4)

hist = expand_all_arrays(starr) #This has shape (Nstar,64)

The entries in hist correspond to the number of stars in each mass bin, where
the edges of the mass bins can be found in the file at StellarPropertiesPath
provided to the original simulation.

"""

STAR_BIN_SIZE = 4
N_BINS = 64

def read_stellar_array(stellar_arr,idx):
	"""
	Obtain the contents of mass bin idx of a single star
	particle's stellar array.

	Input:
	stellar_arr: numpy array, dtype = uint64, shape = (STAR_BIN_SIZE * N_BINS / 64,)
	idx: int

	Returns:
	The value of the histogram at mass bin idx as a uint64
	"""
    
    Nbin_per_element = N_BINS/STAR_BIN_SIZE
    el_loc = int(idx // Nbin_per_element)
    nibble_shift = np.uint64(STAR_BIN_SIZE * (idx % Nbin_per_element))
    
    fullblock = np.array([15],dtype='uint64')
    result = np.bitwise_and(fullblock,np.right_shift(stellar_arr[el_loc],nibble_shift))[0]
    return result

def expand_array(stellar_arr):
	"""
	Convert the stellar array of a single star particle by promoting
	the elements from STAR_BIN_SIZE bit numbers to uint64s. Obviously,
	this is less memory efficient but much easier to work with.

	Input:
	stellar_arr: numpy array, dtype = uint64, shape = (STAR_BIN_SIZE * N_BINS / 64,)

	Returns:
	numpy array, dtype = uint64, shape = (N_BINS,)
	"""

    result = np.zeros(N_BINS,dtype=uint64)
    for idx in range(N_BINS):
        result[idx] = read_stellar_array(stellar_arr,idx)
        
    return result

def expand_all_arrays(stellar_arrs):
		"""
	Convert an array of stellar arrays of Nstar star particles by promoting
	the elements from STAR_BIN_SIZE bit numbers to uint64s. Obviously,
	this is less memory efficient but much easier to work with.

	Input:
	stellar_arrs: numpy array, dtype = uint64, shape = (Nstar,STAR_BIN_SIZE * N_BINS / 64)

	Returns:
	numpy array, dtype = uint64, shape = (Nstar,N_BINS)
	"""

    result = np.zeros((stellar_arrs.shape[0],N_BINS),dtype=uint64)
    for partidx in range(stellar_arrs.shape[0]):
        result[partidx] = expand_array(stellar_arrs[partidx])
    
    return result