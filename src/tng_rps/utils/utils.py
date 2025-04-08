import numpy as np

def find_common_snaps(snaps, SnapNum_1, SnapNum_2):
    """
    Find the common snapshots in snaps where both object1 and object2,
    identified at snapshots SnapNum_1 and SnapNum_2 respectively, are identified.
    Returns the indices into snaps, SnapNum_1, and SnapNum_2, that the objects were identifed.
    NB: currently optimized for a maximum number of 100 snaps, 
    i.e., what the normal TNG boxes use; i.e., not optimized.
    For zooms or higher cadence snaps, consider optimizing.
    """

    # find the snapshots where both the subhalo and host have been identified
    indices_1 = []
    indices_2 = []
    indices_snaps = []
    for snap_index, SnapNum in enumerate(snaps):
        if ((SnapNum in SnapNum_1) & (SnapNum in SnapNum_2)):
            indices_1.append(np.where(SnapNum == SnapNum_1)[0])
            indices_2.append(np.where(SnapNum == SnapNum_2)[0])
            indices_snaps.append(snap_index)
    # note that sub, host indicies are lists of arrays, while
    # snap indices is a list of ints 
    
    # if there are no snaps in common then return three empty arays:
    if len(indices_snaps) == 0:
        return np.array([]), np.array([]), np.array([])

    indices_1 = np.concatenate(indices_1)
    indices_2 = np.concatenate(indices_2)
    indices_snaps = np.array(indices_snaps)

    return indices_snaps, indices_1, indices_2


def shift(u, v, box_length):
    """
    returns the position vector u-v in a periodic Cartesian box
    
    Parameters
    ----------
    u : N x 3 position array
    v : N x 3 OR 1 x 3 position array
    box_length : float in same units as [u], [v]
    
    Returns 
    -------
    result: N x M position vector (array)
    """

    result = u - v
    result[result > box_length / 2.0] -= box_length
    result[result < -box_length / 2.0] += box_length
    return result


def mag(u, v, box_length):
    """
    returns the distance between two physical positions in a periodic box
    assumes Cartesian coordinates
    
    Parameters
    ----------
    u : 3 x N position array
    v : 3 x N OR 3 x 1 position array
    box_length : float in same units as [u], [v]
    
    Returns 
    -------
    magnitude (float)
    
    Notes
    --------
    if u, v are 3x3 arrays, then be careful -- operations are on a row basis
    so column 0 = x; column 1 = y; column 2 = z
    """
    
    v = v.T # replace v with its transpose
            
    diff = shift(u, v, box_length)       
    
    if diff.shape[0] != 3:
        diff = diff.T

    return np.sqrt( (diff[0])**2 + (diff[1])**2 + (diff[2])**2 )
