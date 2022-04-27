import h5py
import matplotlib.pyplot as plt
import numpy as np

# ----------------
#  configureation
# ----------------


FOLDER_PATH = "./data"
FILE_DETAIL = ""

def load(name):
    ''' Load RT data from the hdf5 file
    '''
    path_src = '{}/{}.hdf5'.format(FOLDER_PATH, name, FILE_DETAIL)
    return h5py.File(path_src, 'r')['data'][:]

def load_output(name):
    ''' Load RT data from the hdf5 file
    '''
    path_src = '{}/{}.hdf5'.format("./output", name, FILE_DETAIL)
    return h5py.File(path_src, 'r')['data'][:]

def load_dset(name, dset_name, folder_path="."):
    ''' Load dataset from the hdf5 file
    '''
    path_src = "{}/{}.hdf5".format(folder_path, name)
    f = h5py.File(path_src, 'r')
    return f[dset_name][:]

def delete_dset(name, dset_name, folder_path="."):
    ''' Load dataset from the hdf5 file
    '''
    path_src = "{}/{}.hdf5".format(folder_path, name)
    with h5py.File(path_src,  "a") as f:
        del f[dset_name]
        print("{} delete successfull".format(dset_name))

def load_dset_list(name, n_dimension=2, folder_path="."):
    ''' Load dataset from the hdf5 file
    '''
    path_src = "{}/{}.hdf5".format(folder_path, name)
    f = h5py.File(path_src, 'r')
    cond = [(key.split('_')) if len(key.split('_'))==1+n_dimension else None for key in f.keys()]
    cond =  [x for x in cond if x != None]
    print("^^^^^")
    print(f.keys())
    print("_____")
    return set([l[0] for l in cond]), set([l[1] for l in cond]), set([l[-1] for l in cond])


# NOTE Look the cases that use this function

def save_dset(path, dset_name, data):
    path_src = path
    f = h5py.File(path_src, 'a')
    f.create_dataset(dset_name, data=data)
    print("{} is saved successfully".format(dset_name))


def save_new(name, data, name_dset, version=""):
    path_src = "./output/{}_{}.hdf5".format(name, version)
    f = h5py.File(path_src, 'a')
    f.create_dataset(name_dset, data=data)
    print("{} is saved successfully".format(name_dset))


def draw(data, name, titles=[]):
    '''
    Draw Image
    '''
    if len(data.shape) < 3:
        raise ValueError("data's rank should be larget than 3")
    n = data.shape[-1]

    for i in range(n):

        nan_mask = np.isnan(data[:,:,i])
        d = data[:,:,i][~nan_mask]
        d = d.reshape(-1)
        d = d[~np.isnan(d)]

        plt.figure(figsize=(6, 8))
        plt.subplot(211)
        plt.imshow(data[:,:,i].T, cmap='jet')
        plt.title("({} : min {:.3f}, max {:.3f} mean {:.3f})".format(titles[i], d.min(), d.max(), d.mean()))

        plt.subplot(212)
        plt.figure(1)
        plt.title(titles[i] + "histogram")
        if ~((len(d) == 0) | (d.min() == d.max())):
            plt.hist(d, bins=np.linspace(d.min(), d.max(), 100))

        plt.tight_layout()
        plt.savefig("output/fig_{}_{}_{}.png".format(name, VERSION, titles[i]))
        plt.show()

def draw_new(name, data, idx, title=""):
    ''' Draw Image
    '''
    nan_mask = np.isnan(data[:,:,idx])
    d = data[:,:,idx][~nan_mask]
    plt.imshow(data[:,:,idx].T, cmap='jet')
    plt.title("({} : min {:.3f}, max {:.3f} mean {:.3f})".format(title, d.min(), d.max(), d.mean()))
    plt.savefig("fig_{}_{}_{}.png".format(name, VERSION, title))
    plt.show()



def categorize(data, name):
    '''
    Draw Image
    '''
    
    d = np.argmax(data[:,:,:], axis=2)
    plt.title(name)
    plt.imshow(d.T)
    plt.show()

    # plt.title("Categorization : {}".format(name))
    # plt.savefig("fig_categorization_{}.png".format(name)





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
        # print(idx)
        # print(dictionary.values())
        # print(self.data[idx])
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

























# ================== deprecated ===================

def load_weights(name):
    ''' will be removed
    '''
    path_src = "./{}.hdf5".format(name)
    f = h5py.File(path_src, 'r')
    return f['weight'][:]

def load_params(name):
    ''' will be removed
    '''
    path_src = "./{}.hdf5".format(name)
    f = h5py.File(path_src, 'r')
    return f['params'][:]

# def save(name, weights, params, frequencies):
#     ''' will be removed
#     '''
#     path_src = "./output/{}_{}.hdf5".format(name, version="")
#     f = h5py.File(path_src, 'a')
#     f.create_dataset('weight', data=weights)
#     f.create_dataset('params', data=params)
#     f.create_dataset('frequencies', data=frequencies)
#     print("saved successfully")