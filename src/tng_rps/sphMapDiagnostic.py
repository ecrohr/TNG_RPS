#! /usr/bin/python3

# can be run with python3 ./sphMapDiagnostic.py

import numpy as np
import illustris_python as il
import matplotlib.pyplot as plt
import matplotlib as mpl
from tenet.util import sphMap
import h5py
import os 

def savePlot(basePath, snapNum, haloID, loadOriginalZoom=False, mgas_flag=False, sphMap_flag=True):
    """
    Make and return the cool gas column denisty map using sphMap.sphMap(). No returns.
    """
    header = loadHeader(basePath, 99)
    h = header['HubbleParam']
    gas_ptn = il.util.partTypeNum('gas')
    gas_hsml_fact = 1.5
    Tcoollim = 10.**(4.5)
    axes = [0,1]

    a = header['Time']
    boxsize = header['BoxSize'] * a / h

    halo = il.groupcat.loadSingle(basePath, snapNum, haloID=haloID)
    HaloPos = halo['GroupPos'] * a / h

    if loadOriginalZoom:
        gas_cells = il.snapshot.loadOriginalZoom(basePath, snapNum, haloID, gas_ptn)
    else:
        gas_cells = il.snapshot.loadHalo(basePath, snapNum, haloID, gas_ptn)
    gas_cells = calcTempDict(gas_cells)

    Masses = gas_cells['Masses'] * 1.0e10 / h
    Coordinates = gas_cells['Coordinates'] * a / h
    Densities = gas_cells['Density'] * 1.0e10 / h / (a / h)**3
    Sizes = (Masses / (Densities * 4./3. * np.pi))**(1./3.) * gas_hsml_fact
    Temperatures = gas_cells['Temperature']

    temp_mask = Temperatures <= Tcoollim
    mask = temp_mask

    if mgas_flag:
        mgas_mask = Masses < 1.0e8
        mask = temp_mask & mgas_mask

    for key in gas_cells:
        if key == 'count':
            gas_cells[key] = mask[mask].size
        else:
            gas_cells[key] = gas_cells[key][mask]

    R200c = halo['Group_R_Crit200'] * a / h
    halo_pos = halo['GroupPos'] * a / h
    pos = Coordinates[:,axes]
    hsml = Sizes
    mass = Masses
    quant = None
    boxSizeImg = [3.*R200c, 3.*R200c] # kpc
    boxSizeSim = [boxsize, boxsize, boxsize]
    boxCen = halo_pos[axes]    
    ndims = 3

    nPixels_list = [128, 256, 512, 1024, 2048]
    outdirec = 'CoolGasMaps/'
    if loadOriginalZoom:
        save_str = 'OriginalZoom'
    else:
        save_str = 'Halo'
    if mgas_flag:
        save_str += '-mgas<1e8'

    for nPixels in nPixels_list:
        if sphMap_flag:
            coolgasmap = sphMap.sphMap(pos, hsml, mass, quant, axes, boxSizeImg, boxSizeSim, boxCen, [nPixels, nPixels], ndims, colDens=True)
            proj_str = 'sphMap'
        else:
            area = (boxSizeImg[0] / nPixels)**2
            extent = [-boxSizeImg[0]/2., boxSizeImg[0]/2., -boxSizeImg[1]/2., boxSizeImg[1]/2.]
            range = [[-boxSizeImg[0]/2., boxSizeImg[0]/2.], [-boxSizeImg[0]/2., boxSizeImg[0]/2.]]
            _pos = shift(pos, boxCen, boxsize)
            coolgasmap = np.histogram2d(_pos[:,0], _pos[:,1], bins=nPixels, weights=mass / area, range=range)[0].T
            proj_str = 'histogram2d'
        fig, _ = makePlot(coolgasmap, boxSizeImg, R200c, haloID, nPixels)

        if not os.path.isdir(outdirec):
            os.makedirs(outdirec)
        outfname = '%s_snapNum%03d_haloID%s_CoolGasColDens_%s_%s_nPixels%d.pdf'%(sim, snapNum, haloID, save_str, proj_str, nPixels)
        fig.savefig(outdirec + outfname, bbox_inches='tight')

        fig.clf()

    return

def makePlot(coolgasmap, boxSizeImg, R200c, haloID, nPixels):
    """ 
    Given the result (2D array) from sphMap, make the plot.
    Returns fig, ax
    """

    fig, ax = plt.subplots()
    cmap = mpl.cm.magma.copy()
    cmap.set_under('black')
    cmap.set_bad('black')
    extent = [-boxSizeImg[0]/2., boxSizeImg[0]/2., -boxSizeImg[1]/2., boxSizeImg[1]/2.]
    img = ax.imshow(coolgasmap, norm=mpl.colors.LogNorm(vmin=1.0e7, vmax=3.0e8), cmap=cmap, origin='lower', extent=extent)
    cbar = fig.colorbar(img, ax=ax, extend='both', label=r'Cool Gas Column Density $[{\rm M_\odot}\, {\rm kpc^{-2}}]$')
    ax.set_xticks([])
    ax.set_yticks([])

    circle_r200c = plt.Circle((0., 0.), R200c, color='white', fill=False)
    ax.add_patch(circle_r200c)

    x0 = -boxSizeImg[0]/2. + (boxSizeImg[0]/2. * 0.075)# kpc
    y0 = boxSizeImg[1]/2. -(boxSizeImg[1]/2. * 0.075)# kpc
    length = 1000 # kpc
    ax.plot([x0, x0+length], [y0, y0], color='white', marker='None', ls='-')
    ax.text(x0+length/2., y0-(boxSizeImg[1]/2. * 0.075), '%d pMpc'%int(length / 1.0e3), ha='center', va='top', color='white')
    ax.text(0.975, 0.975, r'$x-y$ proj.', ha='right', va='top', transform=ax.transAxes, color='white')

    text = (r'TNG-Cluster $z=0$' + '\n' +
            'haloID: %d \n'%haloID)
    ax.text(0.025, 0.025, text, ha='left', va='bottom', ma='left', transform=ax.transAxes, color='white')

    ax.text(0.975, 0.025, r'$N_{\rm pixels} = %d^2$'%nPixels, ha='right', va='bottom', transform=ax.transAxes, color='white')

    return fig, ax


def loadHeader(basePath, snapNum):
    """ Load the snapshot catalog header. """
    direc = basePath + 'snapdir_%03d/'%snapNum
    fname = 'snap_%03d.0.hdf5'%snapNum
    
    with h5py.File(direc + fname, 'r') as f:
        header = dict(f['Header'].attrs.items())

        f.close()

    return header


def calcTempDict(gas_cells):
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


if __name__ == '__main__':
    sim = 'TNG-Cluster'
    basePath = '/virgotng/mpia/TNG-Cluster/TNG-Cluster/output/'
    haloID = 5348819
    snapNum = 99

    savePlot(basePath, snapNum, haloID, loadOriginalZoom=False, mgas_flag=False)
    savePlot(basePath, snapNum, haloID, loadOriginalZoom=True, mgas_flag=False)
    savePlot(basePath, snapNum, haloID, loadOriginalZoom=True, mgas_flag=True)

