import h5py
import numpy as np

FOLDER_PATH = "./data"

def load(name):
    ''' Load RT data from the hdf5 file
    '''
    path_src = '{}/{}.hdf5'.format(FOLDER_PATH, name)
    return h5py.File(path_src, 'r')['data'][:]

def save_output(name, data, name_dset, version=""):
    path_src = "./output/{}_{}.hdf5".format(name, version)
    f = h5py.File(path_src, 'a')
    f.create_dataset(name_dset, data=data)
    print("{} is saved successfully".format(name_dset))

class Data():
    data = None
    name = ""
    idx = 0
    
    def __init__(self):
        pass

    def clear(self):
        self.data = None
    
    def create(self, size, name):
        self.data = np.full(size, np.nan)
        self.name = name
    
    def add_row(self, idx, values):
        self.data[idx] = values

    def add_rows(self, rows):
        self.data[self.idx:self.idx+len(rows)] = rows
        self.idx = self.idx+len(rows)

    def add_dictionary(self, idx, dictionary):
        self.data[idx] = np.array(list(dictionary.values()))
    
    def save(self):
        with h5py.File("data/"+self.name+".hdf5", 'a') as file:
            if 'data' in file:
                dset = file['data']
                self.append_to_file(dset, self.data)
            else:
                dset = file.create_dataset('data', data=self.data, compression='gzip', maxshape=(None, 20))
        self.clear()

    def append_to_file(self, dset:h5py._hl.dataset.Dataset, data:np.ndarray):
        ''' Appends new rows to the dataset in hdf5 file. '''
        if(dset.shape[1] != data.shape[1]):
            raise ValueError("Data shapes don't match. Check your the number of columns.")
        else:
            new_shape = (dset.shape[0] + data.shape[0], dset.shape[1])
            prev_num_rows = dset.shape[0]
            dset.resize(new_shape)
            dset[prev_num_rows:] = data