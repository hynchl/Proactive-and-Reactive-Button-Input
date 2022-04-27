import numpy as np
import cupy as cp
from colormath.color_objects import LabColor, sRGBColor
import parmap, multiprocessing
from preprocessing.at import util as util
from util.persistence1d import RunPersistence
from util.custom_colormath_cp import convert_color_cp

from util.data import Data

class g():
    """For Global Variables
    """
    CPU_NUM = 50
    PERSISTENCE_THRESHOLD = 0.1
    CHUNK_SIZE = 100



def get_extrema_mask(data):
    T, H, W = data.shape
    mask = np.full(data.shape, False)
    persistence = np.full(data.shape, 0)

    arguments = []
    for h in range(H):
        for w in range(W):
            sequence = data[:,h,w].reshape(-1)
            if (sequence.max() - sequence.min()) < 0.001:
                continue
            arguments.append((T, h, w, sequence))
    results = parmap.map(_get_extrema_mask, arguments, pm_pbar=True, pm_processes=multiprocessing.cpu_count()-2)
    
    for i, result in enumerate(results):
        h, w = arguments[i][1], arguments[i][2]
        mask[:,h,w] = result[0]
        persistence[:,h,w] = result[1]
    
    return (mask, persistence)



def _get_extrema_mask(args):
    sequence = args[3]
    extremas = RunPersistence(sequence)
    filtered_mask = [int(e[0]) for e in extremas if ((e[1] > g.PERSISTENCE_THRESHOLD) & ~np.isinf(e[0]))]
    filtered_threshold = [int(e[1]) if e[1] != np.inf else 0 for e in extremas if ((e[1] > g.PERSISTENCE_THRESHOLD) & ~np.isinf(e[0]))]

    mask = np.full(args[0], False)
    mask[filtered_mask] = True
    threshold = np.full(args[0], np.nan)
    threshold[filtered_mask] = np.array(filtered_threshold)
    return mask, threshold



def rgb_to_lab(rgb):
    original_shape = rgb.shape
    rgb = cp.asarray(rgb)
    rgb = rgb.reshape((-1, 3))
    srgb = sRGBColor(rgb[:,0], rgb[:,1], rgb[:,2], is_upscaled=True)
    color = convert_color_cp(srgb, LabColor, through_rgb_type=sRGBColor)
    lab = cp.vstack((color.lab_l, color.lab_a, color.lab_b)).T
    result = lab.reshape(original_shape)
    return result.get()
