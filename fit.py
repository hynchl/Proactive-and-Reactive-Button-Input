import sys, argparse, datetime, time, multiprocessing, parmap
import numpy as np
from util import em
from util import data
from util.util import   align_button_interval, remove_most_frequent_bins, get_mask_interval

VERSION = "21080616" # YYMMDDHH
PER_THRESHOLD = 5 # NOTE This should be set from references

# arguments parser
parser = argparse.ArgumentParser(description='Usage Test')
parser.add_argument('--task', required=True)
parser.add_argument('--pid', required=False, type=str, help='data to be analyze')
parser.add_argument('--sound', required=False, type=str, help='participant ID')
parser.add_argument('--key', required=False, type=str, help='target button') # space for dino, all for tabsonic
parser.add_argument('--easing', required=False, type=str, help='easing animation type') # for expanding target
parser.add_argument('--duration', required=False, type=str, help='duration of animation') # for expanding target



def run_em(info, components):
    rt = info['rt']
    C = em.set_starting_values(rt, components)
    result = em.execute_em(rt, C.copy())
    result['x'] = info['x']
    result['y'] = info['y']
    return result



if __name__ == "__main__":

    # get configuration from user
    t0 = time.time() 
    args = parser.parse_args()
    task = args.task
    merged_data = None
    button_intervals = np.array([])



    # for aggregating inputs over several recordings
    # if you want to replicate our study, specify the name of the task (et, dino, tabsonic)
    if task == 'et':
        easing, duration = args.easing, args.duration
        if easing == '' and duration == '':
            file_name = "{}".format(args.pid)
        else:
            file_name = '{}_{}{}'.format(args.pid, duration, easing)

        P = [args.pid]
        for p in P:
            if easing == '' and duration == '':
                name = p
            else:
                name = '{}_{}{}'.format(p, duration, easing)
            d = data.load(name)
            button_input = np.unique(d[:,1])
            button_interval = button_input[1:] - button_input[:-1]
            button_intervals = np.concatenate((button_intervals, button_interval))
            if type(merged_data) != type(None):
                merged_data = np.vstack([merged_data, d])
            else:
                merged_data = d

    elif task == 'dino':
        file_name = args.pid +  '_space'
        merged_data = data.load(file_name)
        button_input = np.unique(merged_data[:,1])
        button_interval = button_input[1:] - button_input[:-1]
        button_intervals = np.concatenate((button_intervals, button_interval))

    elif task == 'tabsonic':
        pid, sound, key = args.pid, args.sound, args.key
        file_name = '{}_{}_{}'.format(pid, sound, key)
        MUSICS = ['A', 'B', 'C', 'D', 'E']
        
        for music in MUSICS:
            name = '{}_{}{}_{}'.format(pid, music, sound, key)
            d = data.load(name)
            button_input = np.unique(d[:,1])
            button_interval = button_input[1:] - button_input[:-1]
            button_intervals = np.concatenate((button_intervals, button_interval))

            if type(merged_data) != type(None):
                merged_data = np.vstack([merged_data, d])
            else:
                merged_data = d

    else:
        file_name = args.pid
        merged_data = data.load(file_name)
        button_input = np.unique(merged_data[:,1])
        button_interval = button_input[1:] - button_input[:-1]
        button_intervals = np.concatenate((button_intervals, button_interval))



    # extract IOI (Input-to-Output Interval)
    T = merged_data[:, 0] # time of pixel transition(visual impulse)
    T_i = merged_data[:, 1] # time of button input
    X = merged_data[:, 3].astype(np.int) # x position of pixel transition
    Y = merged_data[:, 2].astype(np.int) # y position of pixel transition
    dl, da, db =merged_data[:,10], merged_data[:,11], merged_data[:,12] # delta L, delta a, delta b
    IOI = T_i-T 
    del T_i, T

    mask_interval = get_mask_interval(IOI, -2.5, 2.5)
    IOI = IOI[mask_interval]
    X = X[mask_interval]
    Y = Y[mask_interval]
    dl=dl[mask_interval]
    da=da[mask_interval]
    db=db[mask_interval]

    mask_dlab = (np.abs(da)>PER_THRESHOLD) | (np.abs(db)>PER_THRESHOLD) | (np.abs(dl)>PER_THRESHOLD)
    IOI = IOI[mask_dlab]
    X = X[mask_dlab]
    Y = Y[mask_dlab]



    # define components of EM
    bi = align_button_interval(button_intervals)
    components = [{ "name":"reactive",
                    "pdf":em.reactive.pdf,
                    "map":em.reactive.fit_map,
                    "theta": (3, 1, 0.125)},
                    {"name":"anticipatory",
                    "pdf":em.proactive.pdf,
                    "map":em.proactive.fit_map,
                    "theta": (-0.05, 0.05)},
                    {"name":"irrelevant",
                    "pdf":None, 
                    "map":None,  
                    "theta": (0, 1), 
                    "data":bi}]
    LEN_PARAMS = sum([len(c["theta"]) for c in components])
    


    # define container
    width, height = int(X.max())+1, int(Y.max())+1
    weight_map = np.full((width, height, len(components)), np.nan)
    param_map = np.full((width, height, LEN_PARAMS), np.nan)
    frequency_map = np.full((width, height), np.nan)
    llh_map = np.full((width, height), np.nan)
    itr_map = np.full((width, height), np.nan)
    time_map = np.full((width, height), np.nan)
    totaltime = np.zeros(1)


    # calculate the max frequency 
    for x in range(width):
        for y in range(height):
            mask_pixel = (X==x)&(Y==y)
            ioi = IOI[mask_pixel]
            frequency_map[x, y] = len(ioi)
    max_frequency = frequency_map.max()



    # fitting using multiprocessing
    output = []
    for x in range(width):
        for y in range(height):
            if frequency_map[x, y] < 1 / 10 * max_frequency: 
                continue

            mask_pixel = (X==x) & (Y==y)
            ioi = IOI[mask_pixel]
            ioi = remove_most_frequent_bins(ioi)

            if(len(ioi) == 0):
                continue
            
            output.append(dict(x=x, y=y, rt=ioi.copy()))



    print("EM Fitting started at {}".format(datetime.datetime.now()))
    num_cores = multiprocessing.cpu_count() - 1 
    results = parmap.map(run_em, output, components, pm_pbar=True, pm_processes=num_cores)
    for result in results:
        x = result['x']
        y = result['y']
        weight_map[x, y, :] = result['weights']
        param_map[x, y, :] = result['params']
        llh_map[x, y] = result['llh']
        itr_map[x, y] = result['itr_num']
        time_map[x, y] = result['time']
    totaltime = np.array([time.time()-t0])
    print("EM fitting ended at {}".format(datetime.datetime.now()))
    
    

    # save results
    data.save_new(file_name, frequency_map, 'f', version=VERSION)
    data.save_new(file_name, weight_map, 'w', version=VERSION)
    data.save_new(file_name, param_map, 'p', version=VERSION)
    data.save_new(file_name, llh_map, 'llh', version=VERSION)
    data.save_new(file_name, itr_map, 'itr_num', version=VERSION)
    data.save_new(file_name, time_map, 'time', version=VERSION)
    data.save_new(file_name, totaltime, 'total_time', version=VERSION)
    print("Outputs are Successfuly saved at {}".format(datetime.datetime.now()))