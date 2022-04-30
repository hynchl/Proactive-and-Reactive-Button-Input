import numpy as np
from scipy.stats import entropy, skew


def align_button_interval(button_interval):
    bi_aligned = np.concatenate(((button_interval/2), -(button_interval/2)))
    bi_aligned.sort()
    return bi_aligned


def make_align_button_interval(button_input):
    print(button_input)
    button_interval = button_input[1:] - button_input[:-1]
    bi_aligned = np.concatenate(((button_interval/2), -(button_interval/2)))
    bi_aligned.sort()
    return bi_aligned

def get_interval_mask(array, start=-np.inf, end=np.inf):
    '''
    Get mask arrayay with an interval between `start` and `end`.
    'start' value is in and 'end' value not.
    '''
    return (array>start) & (array<end)

def get_interval_mask_2d(array1, array2, cond1=(-np.inf, np.inf), cond2=(-np.inf, np.inf)):
    '''
    Get mask arrayay with an interval between `start` and `end`.
    'start' value is in and 'end' value not.
    '''
    return (array1>cond1[0]) & (array1<cond1[1]) & (array2>cond2[0]) & (array2<cond2[1])

def divide_by_same_size(array, n_section):
    '''
    '''
    size = int(len(array)/n_section)
    intervals = []
    for i in range(n_section):
        section = array[size*i:size*(i+1)] if (i != n_section) \
            else array[size*i:]
        intervals.append((section[0], section[-1]))
    return intervals

def divide_by_same_interval(array, n_section):
    
    intervals_lengths = np.linspace(array.min(), array.max(), n_section+1)
    intervals = []
    for i in range(n_section):
        intervals.append((intervals_lengths[i], intervals_lengths[i+1]))

    print(intervals)
    return intervals

def calculate_entropy(array, interval=(-2.5, 2.5), bin_width=1/15):
    hist = np.histogram(array, bins=np.arange(interval[0], interval[1], bin_width))[0]
    hist = hist[hist != 0]
    return entropy(hist/np.sum(hist))

def calculate_skewness(array, axis=0):
    return skew(array, axis=axis)

def remove_most_frequent_bins(data):
    hist = np.histogram(data, bins=np.arange(-2.5, 2.5, (1/60)))
    max_idx = hist[0].argmax()
    interval = (hist[1][max_idx], hist[1][max_idx+1])
    data = data[~((data<interval[1])&(data>interval[0]))]
    return data

def get_peak_value(data, interval):
    hist = np.histogram(data, bins=np.arange(-2.5, 2.5, (1/60)))
    max_idx = hist[0].argmax()
    interval = (hist[1][max_idx], hist[1][max_idx+1])
    return (interval[0] + interval[1])/2


def get_above_threshold(data:np.array, threshold):
    data[data>threshold] = np.nan
    return data

def get_mask(mask_name, array, persistence, threshold):
    if mask_name == "l":
        mask = ((array//1)%2!=0) & (persistence > threshold)
    elif mask_name == "a":
        mask = ((array//2)%2!=0) & (persistence > threshold)
    elif mask_name == "b":
        mask = ((array//4)%2!=0) & (persistence > threshold)
    elif mask_name == "dl":
        mask = ((array//8)%2!=0) & (persistence > threshold)
    elif mask_name == "da":
        mask = ((array//16)%2!=0) & (persistence > threshold)
    elif mask_name == "db":
        mask = ((array//32)%2!=0) & (persistence > threshold)
    return mask

def get_threshold_mask(array, value):
    return array > value

def get_mask_interval(arr, start, end):
    return (arr>start)&(arr<end)

def put(frame, y, x):
    W, H = frame.shape[0], frame.shape[1]

    if len(frame.shape) == 2:
        new_frame = np.zeros((W*2, H*2))
    else:
        new_frame = np.zeros((W*2, H*2, 3))
    
    x = int(np.asscalar(x))if type(x) != int else int(x)
    y = int(np.asscalar(y))if type(y) != int else int(y)
    
    w_start = int((W/2)+x)
    h_start = int((H/2)+y)

    new_frame[w_start:w_start+W, h_start:h_start+H] = frame

    return new_frame