import tarfile
import os
import h5py
import numpy as np
import glob
import csv

Ncolumns_dict = dict(sf_details=11, sn_details=15)

keys_dict = dict(ParticleIDs=dict(h_scaling=0, a_scaling=0, mass_scaling=0, length_scaling=0, velocity_scaling=0),
                 Time=dict(h_scaling=0, a_scaling=0, mass_scaling=0, length_scaling=0, velocity_scaling=0),
                 Coordinates=dict(h_scaling=-1, a_scaling=1, mass_scaling=0, length_scaling=1, velocity_scaling=0),
                 Velocities=dict(h_scaling=0, a_scaling=-1, mass_scaling=0, length_scaling=0, velocity_scaling=1),
                 AmbientDensity=dict(h_scaling=2, a_scaling=-3, mass_scaling=1, length_scaling=-3, velocity_scaling=0),
                 AmbientTemperature=dict(h_scaling=0, a_scaling=0, mass_scaling=0, length_scaling=0, velocity_scaling=0),
                 AmbientMetallicity=dict(h_scaling=0, a_scaling=0, mass_scaling=0, length_scaling=0, velocity_scaling=0),
                 MassDeposited=dict(h_scaling=-1, a_scaling=0, mass_scaling=1, length_scaling=0, velocity_scaling=0),
                 EnergyDeposited=dict(h_scaling=-1, a_scaling=-2, mass_scaling=1, length_scaling=0, velocity_scaling=2),
                 Age=dict(h_scaling=0, a_scaling=0, mass_scaling=0, length_scaling=0, velocity_scaling=0),
                 LocalFlag=dict(h_scaling=0, a_scaling=0, mass_scaling=0, length_scaling=0, velocity_scaling=0))   


def createMCSTFiles(basePath):
    """ 
    Helper function to read the sf_details and sn_details files and create a single hdf5 file for each.
    The input directory basePath should lead to a given simulation output directory per usual.
    Note that this function assumes that the user has write access to the simulations directory, specifically
    to the postprocessing directory i.e., os.path.split(basePath)[0] + 'postprocessing'. 
    """

    ftypes = ['sf_details', 'sn_details']
    for ftype in ftypes:
        Ncolumns = Ncolumns_dict[ftype]
        direc = os.path.join(os.path.split(basePath)[0], 'postprocessing', ftype)

        # check if the directory already exists. if not, then create it
        if not os.path.isdir(direc):
            c_path = os.path.join(basePath, 'txt-files')
            p_path = os.path.split(direc)[0]
            _fname = ftype + '.tar.gz'
            
            # copy the tar file from c_path to p_path
            os.system('cp %s %s'%(os.path.join(c_path, _fname), os.path.join(p_path, _fname)))
            tar = tarfile.open(os.path.join(p_path, _fname), 'r:gz')
            tar.extractall(path=p_path)
            tar.close()

        # check if otuput file already exists. if so, then nothing to be done.
        outfname = os.path.join(direc, ftype + '.hdf5')
        if os.path.isfile(outfname):
            print('File %s already exists. Nothing to be done here.'%outfname)
            return 
        
        # get all the filenames
        files = glob.glob(os.path.join(direc, '*'))
        files.sort()

        # laod the data from each file
        r = {}
        count = 0
        for file in files:
            r[file] = {}
            # read in file, while ignoring bad rows
            data = read_file_with_standard_columns(file, Ncolumns)
            
            # remove duplicate rows, in case they exist
            unique_data = np.unique(data, axis=0)

            r[file]['data'] = unique_data
            r[file]['count'] = len(unique_data)
            count += len(unique_data)

        # concatenate the data from each file into a single dictionary
        dic = {}
        offset = 0
        for file_i, file in enumerate(r):
            # initialize the output dictionary
            if file_i == 0:
                dic['ParticleIDs'] = np.zeros(count)
                dic['Time'] = np.zeros(count, dtype=np.float64) - 1
                dic['Coordinates'] = np.zeros((count, 3), dtype=np.float64) - 1
                dic['Velocities'] = dic['Coordinates'].copy()
                dic['AmbientDensity'] = dic['Time'].copy()
                dic['AmbientTemperature'] = dic['Time'].copy()
                dic['AmbientMetallicity'] = dic['Time'].copy()
                if ftype == 'sn_details':
                    dic['MassDeposited'] = dic['Time'].copy()
                    dic['EnergyDeposited'] = dic['Time'].copy()
                    dic['Age'] = dic['Time'].copy()
                    dic['LocalFlag'] = np.zeros(count, dtype=bool)
            
            length = r[file]['count']
            dic['ParticleIDs'][offset:offset+length] = r[file]['data'][:,0]
            dic['Time'][offset:offset+length] = r[file]['data'][:,1]
            dic['Coordinates'][offset:offset+length,:] = r[file]['data'][:,2:5]
            dic['Velocities'][offset:offset+length,:] = r[file]['data'][:,5:8]
            dic['AmbientDensity'][offset:offset+length] = r[file]['data'][:,8]
            dic['AmbientTemperature'][offset:offset+length] = r[file]['data'][:,9]
            dic['AmbientMetallicity'][offset:offset+length] = r[file]['data'][:,10]
            if ftype == 'sn_details':
                dic['MassDeposited'][offset:offset+length] = r[file]['data'][:,11]
                dic['EnergyDeposited'][offset:offset+length] = r[file]['data'][:,12]
                dic['Age'][offset:offset+length] = r[file]['data'][:,13]
                dic['LocalFlag'][offset:offset+length] = r[file]['data'][:,14]
            offset += length

        # check that all indicies have been set
        for key in dic:
            dset = dic[key]
            if isinstance(dset, (np.uint8, np.uint16, np.uint32, np.uint64)):
                if len(dset[dset == 0]) > 1:
                    raise ValueError('key %s has mutliple 0 entires'%key)

        # save the resulting dictionary to an hd5f file
        with h5py.File(outfname, 'w') as outf:
            for key in dic:
                outf.create_dataset(key, data=dic[key])
                # add attributes to the dataset
                for key_attr in keys_dict[key]:
                    outf[key].attrs[key_attr] = keys_dict[key][key_attr]
                
            outf.close()

    return


def read_file_with_standard_columns(filename, Ncolumns, return_invalid=False):
    """read in the files, given that there may be errors in the files """
    invalid_rows = []
    valid_rows = []
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter=' ')
        for row_i, row in enumerate(reader):
            if len(row) == Ncolumns:
                valid_rows.append(row)
            else:
                invalid_rows.append([filename, row_i])
    valid_rows = np.array(valid_rows)
    if return_invalid:
        return valid_rows, invalid_rows
    return valid_rows
