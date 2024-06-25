""" 
This file contains utility functions to 
facilitate analysis of IllustrisTNG data.
Eric Rohr compiled this set as needed during
his PhD. There are too many others that 
assisted him in writing these, especially
the myriad of StackOverflow answers.
"""


# import modules used in functions
import numpy as np
import illustris_python as il # written by Dylan Nelson
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
import os
import six
from os.path import isfile
import h5py
from scipy.integrate import quad
from scipy import stats

#######################################
# functions related to 2d histograms

def add_phaseplot(x, y, fig=None, ax=None):
    """
    given x and y, create the scatter plot with running medians, 16/84 percentiles.
    """
    # validate that either both fig and ax are given, or both are None
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(8,5))
    elif fig is None and ax is not None:
        print('Please give fig and ax, or give neither. Returning')
        return
    elif fig is not None and ax is None:
        print('Please give fig and ax, or give neither. Returning')
        return
    # if both are given, then continue normally without creating new fig, ax
    
    bins, bincents = create_bins(x)
    bins = bins[1:] # ignore the left bin edge
    
    xmax = bins[-1]
    
    xhi = x[x > xmax]
    xlo = x[x < xmax]
    
    yhi = y[x > xmax]
    ylo = y[x < xmax]
    
    percentiles = [5, 16, 50, 84, 95]
    
    x_result = ret_binstats(xlo, xlo, bins, [50])
    x_50s = x_result['50']
    
    y_result = ret_binstats(xlo, ylo, bins, percentiles)
    y_05s = y_result['5']
    y_16s = y_result['16']
    y_50s = y_result['50']
    y_84s = y_result['84']
    y_95s = y_result['95']
    
    # begin plotting 
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # set x and y limits for plotting
    xlimfactor = 0.5
    xlolim = xlo.min() * xlimfactor
    xhilim = xhi.max() / xlimfactor
    ax.set_xlim(xlolim, xhilim)
    
    # ignore values more than 1 dex above and below the 95th percentiles
    # but don't cutoff the last few points
    ylimfactor = 0.1
    ylolim = np.min([y_05s.min()*ylimfactor, yhi.min()*xlimfactor])
    yhilim = np.max([y_95s.max()/ylimfactor, yhi.max()/xlimfactor])
    
    # check that ylolim isn't 0 for a log plot
    # replace with 2 dex below 16th percentile 
    if ylolim < 1.0e-3:
        ylolim = np.min([y_16s.min()*ylimfactor*ylimfactor, yhi.min()*xlimfactor])
    # if 16th percentile is still 0, replace with 6 dex below the upper limit
    if ylolim < 1.0e-3:
        ylolim = yhilim / 1.0e6 
    
    ax.set_ylim(ylolim, yhilim)
    
    # plot points that are outside the y-range 
    x_belowplot = xlo[ylo < ylolim]
    y_belowplot = np.ones(len(x_belowplot)) * ylolim / xlimfactor
    
    x_aboveplot = xlo[ylo > yhilim]
    y_aboveplot = np.ones(len(x_aboveplot)) * yhilim * xlimfactor
    
    if len(x_belowplot) > 0:
        ax.plot(x_belowplot, y_belowplot, marker='s', linestyle='None',
                markersize=5, alpha=1.0, color='tab:gray', fillstyle='none')
        ax.plot([xlolim, xhilim], [ylolim/(xlimfactor-0.1), ylolim/(xlimfactor-0.1)], 'k--', marker='None')
    if len(x_aboveplot) > 0:
        ax.plot(x_aboveplot, y_aboveplot, marker='s', linestyle='None',
                markersize=5, alpha=1.0, color='tab:gray', fillstyle='none')
        ax.plot([xlolim, xhilim], [yhilim*(xlimfactor-0.1), yhilim*(xlimfactor-0.1)], 'k--', marker='None')
    
    ax.plot(xlo, ylo, marker='o', linestyle='None', markersize=2, alpha=0.5, color='tab:gray', fillstyle='none', mew=0.5)
    ax.plot(xhi, yhi, marker='o', linestyle='None', markersize=6, alpha=1.0, color='tab:gray', fillstyle='none', mew=1.0)
    
    ax.plot(x_50s, y_50s, marker='None', linestyle='-', linewidth=4, color='black')
    
    ax.plot(x_50s, y_16s, marker='None', linestyle='-', linewidth=2, color='black')
    ax.plot(x_50s, y_84s, marker='None', linestyle='-', linewidth=2, color='black')
    
    ax.plot(x_50s, y_05s, marker='None', linestyle='-', linewidth=1, color='black')
    ax.plot(x_50s, y_95s, marker='None', linestyle='-', linewidth=1, color='black')    
    
    return fig, ax


### taken from StackOverflow, written by @ahwillia
# https://stackoverflow.com/questions/38208700/matplotlib-plot-lines-with-colors-through-colormap
def multiline(xs, ys, c, ax=None, **kwargs):
    """Plot lines with different colorings

    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection

    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

    Returns
    -------
    lc : LineCollection instance.
    """

    # find axes
    ax = plt.gca() if ax is None else ax

    # create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)

    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    lc.set_array(np.asarray(c))

    # add lines to axes and rescale 
    #    Note: adding a collection doesn't autoscale xlim/ylim
    ax.add_collection(lc)
    #ax.autoscale()
    return lc


def retfig2dhist(x, y, binwidth_log, fig=None, ax=None):
    """
    
    return the 2dhist of (x, y) with the medians and
    16th & 84th percentiles within each log(x) bin overplotted
    
    Parameters:
    -----------
    x:  array
        the x-axis parameter -- linear 
    y:  array
        the y-axis parameter -- linear
    binwidth_log: float
        the binwidth in log(x) for 1dhist of y

    Returns
    -------
    (fig, axs) of the plot containing 2dhist and lines
    
    """
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(8,5))
    
    # validate data for taking the log
    x = x[y > 0]
    y = y[y > 0]
    y = y[x > 0]
    x = x[x > 0]
    
    retstats = return2dhiststats(np.log10(x), np.log10(y), binwidth_log)
    
    # assume that colorbar should also be log
    h = ax.hist2d(np.log10(x), np.log10(y),
                  norm=mpl.colors.LogNorm(), bins=50, cmap='viridis')
    fig.colorbar(h[3], ax=ax, fraction=0.046, pad=0.04, 
                 label=r'$\log$ Number')
    ax.plot(retstats[0], retstats[1], linestyle='-', marker='None', color='black', lw=4)
    ax.plot(retstats[0], retstats[2], linestyle='--', marker='None', color='black', lw=2)
    ax.plot(retstats[0], retstats[3], linestyle='--', marker='None', color='black', lw=2)
    
    return (fig,ax)


def return2dhiststats_dict(x, y, bin_width, percentiles=[5, 16, 50, 84, 95]):
    """
    in the x-y plane (assumed either both log or both linear),
    calculate the percentiles [5, 16, 50, 84, 95] within each x bin. 
    Returns a dictionary that contains 'bin_cents' and the percentiles
    within each bin. 
    Note: help came from this StackOverflow answer:
    https://stackoverflow.com/questions/32159869/how-to-make-user-defined-functions-for-binned-statistic

    Parameters
    ----------
    x : array
        the log(x)-axis parameter, passed from retfig2dhsit
    y : array
        the log(y)-axis parameter, passed from retfig2dhist
    bin_width : float
        the binwidth for x to create 1dhist of y (all in log)
        passed from retfig2dhist

    Returns
    -------
    dict{bins, bin_cents, **percentiles}
    """
    
    # create bins between the min and max of x and width bin_width
    
    if type(x) is list:
        x = np.concatenate(x)
    if type(y) is list:
        y = np.concatenate(y)
    
    bin_min = floor_to_value(x.min(), bin_width)
    bin_max = ceil_to_value(x.max(), bin_width)
    bins = np.arange(bin_min, bin_max+bin_width*1.0e-3, bin_width)
    bin_cents = bins[1:] - bin_width/2.0
    result = {}
    result['bin_cents'] = bin_cents
    result['bins'] = bins
    for percentile in percentiles:
        result[percentile] = stats.binned_statistic(x, y, statistic=lambda y: np.percentile(y, percentile), bins=bins)[0]
        
    return result



def return2dhiststats(x, y, bin_width):
    """
    
    return the bin centers, medians, and 16th & 84th percentiles
    for plotting log(y) as a function of log(x)

    Parameters
    ----------
    x : array
        the log(x)-axis parameter, passed from retfig2dhsit
    y : array
        the log(y)-axis parameter, passed from retfig2dhist
    bin_width : float
        the binwidth for x to create 1dhist of y (all in log)
        passed from retfig2dhist

    Returns
    -------
    (bincents, bin_meds, bin_16s, bin_84s)
    
    """
    
    # create bins between the min and max of x and width bin_width
    
    if type(x) is list:
        x = np.concatenate(x)
    if type(y) is list:
        y = np.concatenate(y)
    
    bin_min = floor_to_value(min(x), bin_width)
    bin_max = ceil_to_value(max(x), bin_width)
    bins = np.arange(bin_min, bin_max+bin_width*1.0e-3, bin_width)
    # np.arange() requires bin_max be slightly greater than your 
    # intended max because of float rounding numbers
    
    # calculate the median, 16th and 84th percentiles of y within x bins 
    bin_meds, bin_edges, bin_number = stats.binned_statistic(x, y, 'median', bins=bins)
    bin_16s = stats.binned_statistic(x, y, statistic=percentile16, bins=bins)[0]
    bin_84s = stats.binned_statistic(x, y, statistic=percentile84, bins=bins)[0]
    
    bin_cents = bin_edges[1:] - bin_width/2.0
    
    return(bin_cents, bin_meds, bin_16s, bin_84s)


def ret_binstats(x, y, bins=None, percentiles=[50, 16, 84]):
    """

    Parameters
    ----------
    x : 1xN array
        binning array
    y : 1xN array; must be same length as x
        array to be binned by x; can be x as well
    bins : float or array
           if float, then the binwidth; if array, then right edges of bins
    percentiles : list or array like of ints

    Returns
    -------
    result -- dictionary of percentiles
    
    Notes
    -----
    2021-03-02: I should add another case if bins = int (number of bins)
    then calculate the binwidth and bins in this manner and continue

    """
    
    # check if input bins is actually a binwidth
    # then create logbins around [min(x), max(x)]
    if isinstance(bins, (float)):
        binwidth = bins
        bins = returnlogbins(x, binwidth)[0] 
        # returnlogbins() includes the left bin edge, so get rid of it
        bins = bins[1:]
    elif bins is None:
        bins = create_bins(x)[0][1:]
    
    indices = np.digitize(x, bins)
        
    # initialize dictionary of results
    result = {}
    for percentile in percentiles:
        key = '%s'%percentile
        result[key] = np.zeros(len(bins))
    
    # loop over bins, calculating the percentiles for each bin
    for i in range(len(bins)):
        data = y[indices == i]
        for j, percentile in enumerate(percentiles):
            key = list(result.keys())[j]
            result[key][i] = np.percentile(data, percentile)
        
    return result


def return1dhiststats(x, binwidth, log=True, density=False):
    """
    
    return the bin centers, medians, and 16th & 84th percentiles
    for plotting the histogram of log(x)
    
    Parameters
    ----------
    x : list of arrays/lists 
        the parameter that is being binned
    binwidth: float
              the bin_width in log(x) units for the histogram
                   
    Returns
    -------
    (bincents, bin_meds, bin_16s, bin_84s)
    
    """
    
    xtest = np.concatenate(x)
    if (log):
        bins, bincents = returnlogbins(xtest, binwidth)
    else:
        bins, bincents = returnbins(xtest, binwidth)
    n = []
    
    for i in range(len(x)):
        xi = x[i]
        n.append(np.histogram(xi, bins=bins, density=density)[0])
        
    bin_meds = np.median(n, axis=0)
    bin_16s = np.percentile(n, 16, axis=0)
    bin_84s = np.percentile(n, 84, axis=0)
    
    return bincents, bin_meds, bin_16s, bin_84s
   
    
### 2021-03-10: written by Eric Rohr to calculate bins given x and nothing else
### has a kwarg for the minimum number of points per bin, which also excludes
### the last few x-points from binning 
def create_bins(x, minpointsperbin=7):
    """
    iterate to find the optimal log bin size for dataset x
    """
            
    # sort x such that it's in ascending order
    x = x[np.argsort(x)]
    
    x_log = np.log10(x)
    
    # confirm that the last few points are NOT included in the bins
    xhi_log = x_log[-minpointsperbin:]
        
    # xmax < xhi_log.min(), so len(xhi_log) > 10
    # round down to the nearest half dex for xmax
    xmax_log = floor_to_value(xhi_log.min(), 0.5)
    xlo = x[x_log < xmax_log]
    
    # guess initial binwidth by starting with 100 bins
    xmin_log = floor_to_value(np.log10(xlo.min()), 0.5)
    spread = xmax_log - xmin_log
    nbins = 100
    binwidth = spread / nbins

    # initialize vals for histogram test
    vals = 0
    
    while np.min(vals) < minpointsperbin:
        # calculate the bins with the given binwidth
        bins = np.logspace(xmin_log, xmax_log, num=nbins)
        # check the number of points in each bin
        vals = np.histogram(xlo, bins)[0]
        # increase the binwidth for next iteration
        binwidth *= 2.0
        nbins = int(round(spread / binwidth))+1
    
    # divide binwidth by 2 to offset the while loop 
    # divide binwidth by 2 again for the bincents
    bincents = 10**(np.log10(bins)[1:] - binwidth/4.)
        
    return bins, bincents


def returnlogbins(x, binwidth_log):
    """
    
    returns the logbins of (x) with the given binwidth
    
    Parameters
    ---------
    x : array
    binwidth_log : float in units of log(x)

    """

    bin_min = floor_to_value(np.log10(min(x)), binwidth_log) - binwidth_log/2.
    bin_max = ceil_to_value(np.log10(max(x)), binwidth_log) + binwidth_log/2.
    nbins = int(round((bin_max - bin_min) / binwidth_log))+1

    logbins = np.logspace(bin_min, bin_max, num=nbins)
    bincents = 10**(np.log10(logbins)[1:] - binwidth_log/2.)

    return logbins, bincents


def returnbins(x, binwidth):
    """
    given x and binwidth, calculate the bin edges and bin centers.
    """
    bin_min = floor_to_value(min(x), binwidth) - binwidth/2.
    bin_max = ceil_to_value(max(x), binwidth) + binwidth/2.
    nbins = int(round((bin_max - bin_min)/ binwidth))
    
    bins = np.linspace(bin_min, bin_max, num=nbins+1)
    bin_cents = bins[1:] - binwidth/2.
    
    return bins, bin_cents


def floor_to_value(number,roundto):
    """
    
    returns the round-down of number to the nearest roundto
    
    Paramters
    ---------
    x : float
    y : float

    """
    return (np.floor(number / roundto) * roundto)

def ceil_to_value(number,roundto):
    """
    
    returns the round-up of number to the nearest roundto
    
    Paramters
    ---------
    x : float
    y : float

    """

    return (np.ceil(number / roundto) * roundto)

def percentile16(y):
    """
    
    returns the 16th percentile (1 sigma) of y
    
    Parameters:
    y : array
    
    """
    
    return(np.percentile(y,16))

def percentile84(y):
    """
    
    returns the 84th percentile (1 sigma) of y
    
    Parameters:
    y : array
    
    """

    return(np.percentile(y,84))

def round_to_value(number,roundto):
    """
    
    returns the round of number to the nearest roundto
    
    Paramters
    ---------
    x : float
    y : float

    """

    return (round(number / roundto) * roundto)


# utility functions to check if a list is contained in another list
# based on https://stackoverflow.com/questions/20789412/check-if-all-elements-of-one-array-is-in-another-array
def where_is_slice_in_list(s,l):
    """
    Returns the bool array of True/Flase of len(result) = len(l) - len(s) + 1.
    if result[i] == True, then l[i:i+len(s)] == s
    Example: s      = [1, 1, 1]
             l      = [1, 1, 1, 1, 0].
             result = array([True, True, False])
    """
    s     = list(s)
    l     = list(l)
    len_s = len(s) 
    bools = np.array([s == l[i:len_s+i] for i in range(len(l) - len_s+1)])
    return bools

def is_slice_in_list(s,l):
    """
    Returns bool if the the slice is within the list (or array)
    """
    bools = where_is_slice_in_list(s, l)
    return any(bools)


#############################################
##### other functions #########
########################################

def calc_tcool_dict(gas_cells, basePath, snapNum):
    """
    Compute the cooling time [Gyr] for gas_cells from an arepo sim. 
    Requires Density, ElectronAbundance, GFM_CoolingRate,
    InternalEnergy, (StarFormationRate), so for the TNG + TNG-Cluster simulations, this
    is only possible at the full snaps. In order to convert Density to physical units, 
    it is necessary to have both HubbleParam and Time from the header file, hence why
    basePath and snapNum are required arguments. Note that the dataset is returned as 
    a double since converting between Msun and cgs can lead to an OverFlowError.
    Adds 'CoolingTime' [Gyr] to gas_cells and returns gas_cells.
    """

    if 'Temperature' not in gas_cells:
        gas_cells = calc_temp_dict(gas_cells)

    keys = ['Density', 'ElectronAbundance', 'Temperature', 'GFM_CoolingRate']
    for key in keys:
        assert key in gas_cells, 'calc_tcool_dict(): Key %s is required in gas_cell to compute tcool'%key
            
    header = loadHeader(basePath, snapNum)
    h = header['HubbleParam']
    a = header['Time']

    xH = 0.76 # hydrogen mass fraction
    mp = 1.67e-24 # proton mass [g]
    kpc_to_cm = 3.086e21 # convert kpc to cm
    kb = 1.3806e-16 # Boltzmann constant [erg K^-1]
    msun = 1.988e33 # solar mass in [g]
    seconds_to_Gyr = 1. / (3.15e7 * 1.0e9) # convert s to Gyr

    nH = xH * ((gas_cells['Density'].astype('double') * 1.0e10 * msun / h) / (kpc_to_cm * a / h)**3) / mp
    ne = gas_cells['ElectronAbundance'].astype('double') * nH
    ni = (1. - xH) * nH

    tcool = -(3./2.) * (ne + ni) * kb * gas_cells['Temperature'] / (ne * ni * gas_cells['GFM_CoolingRate']) * seconds_to_Gyr
 
    gas_cells['CoolingTime'] = tcool

    return gas_cells


def calc_temp_dict(gas_cells):
    """
    return the temperature [K] of a gas cell from TNG
    gas_cells should be a dictionary with keys InternalEnergy,
    ElectronAbundance, and optionally StarFormationRate.
    Return is the dictionary with added entry Temperature.  
    """
    
    # if the proper keys do not exist, then returns the original dict
    if 'InternalEnergy' not in gas_cells or 'ElectronAbundance' not in gas_cells:
        print('Please load InternalEnergy, ElectronAbundance, and optionally StarFormationRate. Returning.')
        return gas_cells
    
    # define constans
    xh = 0.76 # hydrogen mass fraction
    mp = 1.67e-24 # proton mass [g]
    gamma = 5./3. # adiabatic index
    kb = 1.3806e-16 # Boltzmann constant [erg K^-1]
    unitconversion = 1.0e10 # code unit conversion
    
    mu = (4. / (1. + 3. * xh + 4. * xh * gas_cells['ElectronAbundance'])) * mp # mean molecular weight
    t = (gamma - 1.) * (gas_cells['InternalEnergy'] / kb) * unitconversion * mu # temperature [K]
    
    if 'StarFormationRate' in gas_cells:
        t[gas_cells['StarFormationRate'] > 0] = 1.0e3
        
    gas_cells['Temperature'] = t
    
    return gas_cells
   

def calc_temp(InternalEnergy, ElectronAbundance, StarFormationRate):
    """
    return the temperature [K] of a gas cell from TNG
    
    Parameters
    ----------
    InternalEnergy, ElectronAbundance, StarFormationRate -- all are defined in TNG documentation
    
    Returns
    -------
    gas cell temperature in K
    
    """
    
    # define constans
    xh = 0.76 # hydrogen mass fraction
    mp = 1.67e-24 # proton mass [g]
    gamma = 5./3. # adiabatic index
    kb = 1.3806e-16 # Boltzmann constant [erg K^-1]
    unitconversion = 1.0e10 # code unit conversion
    
    mu = (4. / (1. + 3. * xh + 4. * xh * ElectronAbundance)) * mp # mean molecular weight
    t = (gamma - 1.) * (InternalEnergy / kb) * unitconversion * mu # temperature [K]
    
    if type(t) is float:
        if StarFormationRate > 0:
            t = 1.0e3 
    else:
        t[StarFormationRate > 0] = 1.0e3
    
    return t

def calc_temp_NOSFR(InternalEnergy, ElectronAbundance):
    """
    return the temperature [K] of a gas cell from TNG
    
    Parameters
    ----------
    InternalEnergy, ElectronAbundance -- both are defined in TNG documentation
    
    Returns
    -------
    gas cell temperature in K
    
    """
    
    # define constans
    xh = 0.76 # hydrogen mass fraction
    mp = 1.67e-24 # proton mass [g]
    gamma = 5./3. # adiabatic index
    kb = 1.3806e-16 # Boltzmann constant [erg K^-1]
    unitconversion = 1.0e10 # code unit conversion
    
    mu = (4. / (1. + 3. * xh + 4. * xh * ElectronAbundance)) * mp # mean molecular weight
    t = (gamma - 1.) * (InternalEnergy / kb) * unitconversion * mu # temperature [K]
        
    return t


def calc_mag_dict(parts, coordinates, center, box_length):
    """
    Add the dataset "Radii" to parts based on coordinates, centered 
    on center in a periodic box of size box_length
    """

    parts['Radii'] = mag(coordinates, center, box_length)
    return parts


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

def printname(name):
    """ 
    function to print used for "visit"ing hdf5 files
    """
    
    print(name)
    
    return


def return_zs_costimes():
    """
    return the redshifts and cosmic times from snap 99 to 0, by 
    integrating the Friedmann equation using the redshifts and 
    cosmology that are hardcoded and used in TNG

    Returns
    -------
    (zs, times) :   arrays (float) of the redshift and cosmic time 
                    at each snapnum

    """
    zs = np.array([20.05, 14.99, 11.98, 10.98, 10.00, 9.39, 9.00,
                   8.45,  8.01,  7.60,  7.24,  7.01,  6.49, 6.01,
                   5.85,  5.53,  5.23,  5.00,  4.66,  4.43, 4.18,
                   4.01,  3.71,  3.49,  3.28,  3.01,  2.90, 2.73,
                   2.58,  2.44,  2.32,  2.21,  2.10,  2.00, 1.90,
                   1.82,  1.74,  1.67,  1.60,  1.53,  1.50, 1.41,
                   1.36,  1.30,  1.25,  1.21,  1.15,  1.11, 1.07,
                   1.04,  1.00,  0.95,  0.92,  0.89,  0.85, 0.82,
                   0.79,  0.76,  0.73,  0.70,  0.68,  0.64, 0.62,
                   0.60,  0.58,  0.55,  0.52,  0.50,  0.48, 0.46,
                   0.44,  0.42,  0.40,  0.38,  0.36,  0.35, 0.33,
                   0.31,  0.30,  0.27,  0.26,  0.24,  0.23, 0.21,
                   0.20,  0.18,  0.17,  0.15,  0.14,  0.13, 0.11,
                   0.10,  0.08,  0.07,  0.06,  0.05,  0.03, 0.02,
                   0.01,  0.00])
    times = np.zeros(len(zs))

    H0 = 67.84 # [km s^-1 Mpc^-1]
    Omega0 = 0.31
    OmegaL = 0.69
        
    for index, z in enumerate(zs):        
        times[index] = cosmictime(z, H0, Omega0, OmegaL) # [yr]
        
    return(zs, times)
    
    
def timesfromsnap(basePath, snapnum):
    """
    return the redshifts and cosmic times given snapshot, via 
    integrating the Friedmann equation using the redshift and 
    cosmology given in the basePath/snapdir_snapnum/*.hdf5
    headers

    Parameters
    ----------
    basePath :  string
                filepath leading to ~simulation~/output
    snapnum :   int or array/list of floats
                the snapnum used for openning the hdf5 headers

    Returns
    -------
    (zs, times) :   arrays (float) of the redshift and cosmic time 
                    at each snapnum

    """
    
    # check if snapnum is one number vs list or array
    if type(snapnum) == int:
        snapnum = [snapnum] # make a list of the one number
    zs = np.zeros(len(snapnum)) 
    times = np.zeros(len(snapnum))
        
    for index, snap in enumerate(snapnum):
        hdr = il.groupcat.loadHeader(basePath, snap)
        
        if index == 0:
            # H0, Omega0, OmegaL are cononstant, so only load them once
            H0 = hdr['HubbleParam'] * 100. # [km s^-1 Mpc^-1]
            Omega0 = hdr['Omega0']
            OmegaL = hdr['OmegaLambda']

        z = hdr['Redshift']
        
        zs[index] = z
        times[index] = cosmictime(z, H0, Omega0, OmegaL) # [yr]
        
    return zs, times
        
def cosmictime(z, H0, OmegaM, OmegaL):
    """
    return the integrated Friedmann equation from 0 to a = 1/(1+z), 
    given the cosmologic parameters
    
    Parameters
    ----------
    z :         float -- redshift
    H0 :        float -- [km s^-1 Mpc^-1] Hubble Parameter at z=0
    OmegaM :    float -- cosmological matter density parameter at z=0
    OmegaL :    float -- cosmological dark energy density parameter at z=0
    
    Returns
    -------
    (t) :   cosmic time [yr] since Big Bang 

    """
    
    integral = quad(Friedmann, 0, 1. / (1.+z), args=(OmegaM,OmegaL))[0]
    result = integral / H0 / 1.022e-12 # converts [s Mpc km^-1] to [yr]
    return result
    
def Friedmann(a, OmegaM, OmegaL):
    """
    returns the integrand of the Friedmann equation, given the 
    cosmological parameters and assuming neutrino and radiaiton 
    density parameters are 0. this is used in cosmictime()
    
    a = 1/1+z is the scale factor 
    
    """
    return(1./np.sqrt(OmegaM * a**(-1) + OmegaL * a**2))


def latex_float(f, num):
    """
    convert a given float f to a latex string, with num decimal places
    """
    float_str = ("{0:.%dg}"%(num)).format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(round(float(base), num), int(exponent))
    else:
        return float_str


def RunningMedian(x, N):
    """
    Calculate the symmetric running median of Mx1 array x over N elements.
    For example with N = 3, calculates the median based on the current,
    immediately previous, and immediately following values.
    N must be a positive odd integer >= 3.
    based on code found at https://stackoverflow.com/questions/37671432/how-to-calculate-running-median-efficiently
    """
    result = np.zeros(len(x), dtype=x.dtype) - 1

    if (N < 3) or (N % 2 != 1) or (type(N) != int):
        print('Error, N must be a positive odd integer >= 3. Returning')
        return result
    
    start = int((N - 1) / 2)
    indices = np.arange(N) + np.arange(len(x)-N+1)[:,None]
    result[start:-start] = np.array(list(map(np.median,x[indices])))
    for i in range(start):
        result[i] = np.median(x[:2*i+1])
        result[-(i+1)] = np.median(x[-(2*i+1):])
        
    return result


########################################################
###### functions related to hdf5 I/O ###############
########################################################
    
# these first two functions are largely based on Dylan
# Nelson's functions from the public release of TNG.
# I am merely adapting them to work with subboxes 
# rather than the group catalogs they were intended for.

def loadSubboxSubset(basePath, snapNum, subboxNum, partType, fields=None, subset=None, mdi=None, sq=True, float32=False):
    """ Load a subset of fields for all particles/cells of a given partType.
        If offset and length specified, load only that subset of the partType.
        If mdi is specified, must be a list of integers of the same length as fields,
        giving for each field the multi-dimensional index (on the second dimension) to load.
          For example, fields=['Coordinates', 'Masses'] and mdi=[1, None] returns a 1D array
          of y-Coordinates only, together with Masses.
        If sq is True, return a numpy array instead of a dict if len(fields)==1.
        If float32 is True, load any float64 datatype arrays directly as float32 (save memory). """
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


### 2020-11-25: written by Eric Rohr to load specific fields of a single object
### includes fields validation, but no longer the "sq" bool as an input
### you should be able to use fields=None (the default) and have an identical 
### result to using loadSingle
def loadSingleFields(basePath, snapNum, haloID=-1, subhaloID=-1, fields=None):
    """ Return complete group catalog information for one halo or subhalo. """
    if (haloID < 0 and subhaloID < 0) or (haloID >= 0 and subhaloID >= 0):
        raise Exception("Must specify either haloID or subhaloID (and not both).")

    gName = "Subhalo" if subhaloID >= 0 else "Group"
    searchID = subhaloID if subhaloID >= 0 else haloID
    
    # make sure fields is not a single element
    if isinstance(fields, six.string_types):
        fields = [fields]

    # old or new format
    if 'fof_subhalo' in il.groupcat.gcPath(basePath, snapNum):
        # use separate 'offsets_nnn.hdf5' files
        with h5py.File(il.groupcat.offsetPath(basePath, snapNum), 'r') as f:
            offsets = f['FileOffsets/'+gName][()]
    else:
        # use header of group catalog
        with h5py.File(il.groupcat.gcPath(basePath, snapNum), 'r') as f:
            offsets = f['Header'].attrs['FileOffsets_'+gName]

    offsets = searchID - offsets
    fileNum = np.max(np.where(offsets >= 0))
    groupOffset = offsets[fileNum]

    # load halo/subhalo fields into a dict
    result = {}

    with h5py.File(il.groupcat.gcPath(basePath, snapNum, fileNum), 'r') as f:
        
        # if fields not specified, load everything
        if not fields:
            fields = list(f[gName].keys())

        # loop over each requested field
        for field in fields:
            # verify existence
            if field not in f[gName].keys():
                raise Exception("Group catalog does not have requested field [" + field + "]!")
                
            result[field] = f[gName][field][groupOffset]

    # only a single field? then return the array instead of a single item dict
    if len(fields) == 1:
        return result[fields[0]]

    return result

### 2020-12-09: Written by Eric Rohr to load specific fields of multiple objects.
### This is very similar to loadSingleField, and it should still work for a single
### (sub)haloID, but it is designed for multiple objects. However, this requires
### that all of the objects belong to the same chunk. For example, load in all
### the subhalo masses belonging to a given FoF at snapNum. This also only works
### properly for IDs that are consecutive, such as range(IDmin, IDmax). In the
### future I plan to generalize this for any given list of IDs.
def loadMultipleFields(basePath, snapNum, haloID=None, subhaloID=None, fields=None):
    """ Return complete group catalog information for one halo or subhalo. """
    if (haloID is None and subhaloID is None) or (haloID is not None and subhaloID is not None):
        raise Exception("Must specify either haloID or subhaloID (and not both).")

    gName = "Subhalo" if subhaloID is not None else "Group"
    searchID = subhaloID if subhaloID is not None else haloID
    
    if type(searchID) == list:
        searchID == np.array(searchID)
    
    # make sure fields is not a single element
    if isinstance(fields, six.string_types):
        fields = [fields]


    # old or new format
    if 'fof_subhalo' in il.groupcat.gcPath(basePath, snapNum):
        # use separate 'offsets_nnn.hdf5' files
        with h5py.File(il.groupcat.offsetPath(basePath, snapNum), 'r') as f:
            offsets = f['FileOffsets/'+gName][()]
    else:
        # use header of group catalog
        with h5py.File(il.groupcat.gcPath(basePath, snapNum), 'r') as f:
            offsets = f['Header'].attrs['FileOffsets_'+gName]

    offsetsmin = np.min(searchID) - offsets
    fileNummin = np.max(np.where(offsetsmin >= 0))
    groupOffsetmin = offsets[fileNummin]
    
    offsetsmax = np.max(searchID) - offsets
    fileNummax = np.max(np.where(offsetsmax >= 0))
    groupOffsetmax = offsetsmax[fileNummax]
    
    if fileNummin == fileNummax:
        fileNum = fileNummin
    else:
        raise Exception("FoFs or Subs come from different chunks.")
    

    # load halo/subhalo fields into a dict
    result = {}

    with h5py.File(il.groupcat.gcPath(basePath, snapNum, fileNum), 'r') as f:
        
        # if fields not specified, load everything
        if not fields:
            fields = list(f[gName].keys())

        # loop over each requested field
        for field in fields:
            # verify existence
            if field not in f[gName].keys():
                raise Exception("Group catalog does not have requested field [" + field + "]!")
                
            if groupOffsetmin != groupOffsetmax:
                result[field] = f[gName][field][groupOffsetmin:groupOffsetmax]
            else:
                result[field] = f[gName][field][groupOffsetmin]

    # only a single field? then return the array instead of a single item dict
    if len(fields) == 1:
        return result[fields[0]]


    return result



### 2020-11-26: written by Eric Rohr to load specific fields from the infall
### catalogs. Currently there is no verification that the file exists.
### I should add a subset kwarg only to load the fields for given indices 
### 2021-03-09: updated basePath to lead to ~simulation/output rather than
### ~simulation; added the kwarg indices -- only load infall catalog for 
### specifi infall indices NOT z=0 fof indices 
def loadInfallFields(basePath, fields=None, infallindices=None):
    """
    experimental function to load the infall catalogs easily
    """
    
    # load infall catalog into a dict
    result = {}
    
    ### note that basePath should lead to ~simulation/output
    with h5py.File(basePath + '../postprocessing/InfallCatalog/infall_catalog_099.hdf5', 'r') as f:
        
        # if fields not specificed load everything
        if fields is None:
            fields = list(f.keys())
        
        # loop over each requested field
        for field in fields:
            # verify existence
            if field not in f.keys():
                raise Exception("Group catalog does not have requested field [" + field + "]!")
            
            if infallindices is not None:
                result[field] = f[field][infallindices]
            else:
                result[field] = f[field][:]
        
    # if only a single field then return the array instead of a single item dict
    if len(fields) == 1:
        return result[fields[0]]
    
    return result


def ret_basePath(sim):
    """
    given the simulation of interest, return the basePath
    """
    IllustrisTNG_sims = ['TNG50', 'TNG100', 'TNG300', 'L35n', 'L75n', 'L205n']
    if any(IllustrisTNG_sim in sim for IllustrisTNG_sim in IllustrisTNG_sims):
        return ('../IllustrisTNG/%s/output/'%sim)
        
    TNGCluster_sims = ['L680n', 'TNG-Cluster']
    if any(TNGCluster_sim in sim for TNGCluster_sim in TNGCluster_sims):
        return ('../TNG-Cluster/%s/output/'%sim)
        
    raise ValueError('Input simulation %s has no defined basePath.'%sim)
    

def return_basePath(sim):
    """
    alias for ret_basePath
    """
    return ret_basePath(sim)
    
    
def loadbasePath(sim):
    """
    alias for ret_basePath
    """
    return ret_basePath(sim)


def loadHeader(basePath, snapNum):
    """ Load the snapshot catalog header. """
    direc = basePath + 'snapdir_%03d/'%snapNum
    fname = 'snap_%03d.0.hdf5'%snapNum
    
    with h5py.File(direc + fname, 'r') as f:
        header = dict(f['Header'].attrs.items())

        f.close()

    return header

def loadMainTreeBranch(sim, snap, subfindID, fields=None, treeName='SubLink_gal',
                       min_snap=0, max_snap=99):
    """
    Return the entire main branch (progenitor + descendant) of a given subhalo.
    When snap == 99, then just returns the MPB.
    Has the option only to return the tree between min and max snaps. 
    """
    
    basePath = ret_basePath(sim)
    
    
    # start by loading the MPB
    subMPB = il.sublink.loadTree(basePath, snap, subfindID, treeName=treeName,
                                 fields=fields, onlyMPB=True)
    
    if not subMPB:
        return 
    
    if snap == 99:
        tree = subMPB

    else:
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
        for key in subMPB.keys():
            if key == 'count':
                tree[key] = subMDB[key] + subMPB[key] - 1
            else:
                tree[key] = np.concatenate([subMDB[key][:-1], subMPB[key]])


    indices = (tree['SnapNum'] >= min_snap) & (tree['SnapNum'] <= max_snap)
    for field in fields:
        tree[field] = tree[field][indices]
    tree['count'] = len(indices[indices])
    
    return tree


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


# from Dylan Nelson
def match3(ar1, ar2, firstSorted=False, parallel=False):
    """ Returns index arrays i1,i2 of the matching elements between ar1 and ar2. While the elements of ar1 
        must be unique, the elements of ar2 need not be. For every matched element of ar2, the return i1 
        gives the index in ar1 where it can be found. For every matched element of ar1, the return i2 gives 
        the index in ar2 where it can be found. Therefore, ar1[i1] = ar2[i2]. The order of ar2[i2] preserves 
        the order of ar2. Therefore, if all elements of ar2 are in ar1 (e.g. ar1=all TracerIDs in snap, 
        ar2=set of TracerIDs to locate) then ar2[i2] = ar2. The approach is one sort of ar1 followed by 
        bisection search for each element of ar2, therefore O(N_ar1*log(N_ar1) + N_ar2*log(N_ar1)) ~= 
        O(N_ar1*log(N_ar1)) complexity so long as N_ar2 << N_ar1. """
    if not isinstance(ar1,np.ndarray): ar1 = np.array(ar1)
    if not isinstance(ar2,np.ndarray): ar2 = np.array(ar2)
    assert ar1.ndim == ar2.ndim == 1
    
    if not firstSorted:
        # need a sorted copy of ar1 to run bisection against
        
        # currently not supported for parallel
        if parallel:
            #index = p_argsort(ar1)
            raise ValueError('Parallel search currently not implemented.')
        else:
            index = np.argsort(ar1)
        
        ar1_sorted = ar1[index]
        ar1_sorted_index = np.searchsorted(ar1_sorted, ar2)
        ar1_sorted = None
        ar1_inds = np.take(index, ar1_sorted_index, mode="clip")
        ar1_sorted_index = None
        index = None
    else:
        # if we can assume ar1 is already sorted, then proceed directly
        ar1_sorted_index = np.searchsorted(ar1, ar2)
        ar1_inds = np.take(np.arange(ar1.size), ar1_sorted_index, mode="clip")

    mask = (ar1[ar1_inds] == ar2)
    ar2_inds = np.where(mask)[0]
    ar1_inds = ar1_inds[ar2_inds]

    if not len(ar1_inds):
        return None,None

    return ar1_inds, ar2_inds
