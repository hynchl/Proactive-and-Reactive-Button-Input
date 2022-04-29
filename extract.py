import time, argparse, time, math
import numpy as np
import parmap, multiprocessing
import util.video as video
from util.converter import Converter
from util.util import put
from util.extract import rgb_to_lab 
from util.persistence1d import RunPersistence
from util.data import Data

THRESHOLD = 5
PERSISTENCE_THRESHOLD = 0.1

parser = argparse.ArgumentParser(description='Usage Test')
parser.add_argument('--path', required=True, type=str, help='data to be analyze')
parser.add_argument('--device', default='keyboard', type=str, help='trigger type')
parser.add_argument('--keys', default='all')
parser.add_argument('--width', default=1680, type=int)
parser.add_argument('--height', default=1050, type=int)
parser.add_argument('--divider', default=32, type=int)
parser.add_argument('--relative', default=False, type=bool)
parser.add_argument('--task', default='')
parser.add_argument('--chunk', required=True, type=int)


class Config():
    is_relative = False
    divider = 32
    name = ""
    task = ""
    width = 1680
    height = 1050
    env_width = 1680 # for relative mode
    env_height = 1050 # for relative
    device = "keyboard"
    keys = []
    output_name = ""
    chunk_size = 1000



def extract(T_o, T_i, rgb, output_name):

    lab = rgb_to_lab(rgb)
    dlab = lab[1:]-lab[:-1]
    mask = None

    if True:
        l_mask, l_p = get_extrema_mask(lab[:,:,:,0])
        a_mask, a_p = get_extrema_mask(lab[:,:,:,1])
        b_mask, b_p = get_extrema_mask(lab[:,:,:,2])
        l_mask, l_p = l_mask[1:], l_p[1:]
        a_mask, a_p = a_mask[1:], a_p[1:]
        b_mask, b_p = b_mask[1:], b_p[1:]
    
        dl_mask, dl_p = get_extrema_mask(dlab[:,:,:,0])
        da_mask, da_p = get_extrema_mask(dlab[:,:,:,1])
        db_mask, db_p = get_extrema_mask(dlab[:,:,:,2])
    
        mask = (l_mask | a_mask | b_mask) | (dl_mask | da_mask | db_mask)
        mask &= (np.abs(dlab[:,:,:,0])>THRESHOLD) | (np.abs(dlab[:,:,:,1])>THRESHOLD) | (np.abs(dlab[:,:,:,2])>THRESHOLD)
    # else:
    #     mask = (dlab[:,:,:,0]>THRESHOLD) | (dlab[:,:,:,1]>THRESHOLD) | (dlab[:,:,:,2]>THRESHOLD)
    
    indices = np.where(mask)

    extracted_data = Data()
    extracted_data.create((len(indices[0]), 13), output_name)
    for i in range(len(indices[0])):
        t, y, x = indices[0][i], indices[1][i], indices[2][i]
        row = np.array([
            T_o[t+1], # time
            T_i[t+1], # button-input time
            x, 
            y, 
            rgb[t+1, y, x, 0], #raw rgb color
            rgb[t+1, y, x, 1],
            rgb[t+1, y, x, 2], 
            lab[t+1, y, x, 0], #lab color
            lab[t+1, y, x, 1], 
            lab[t+1, y, x, 2], 
            dlab[t, y, x, 0], #dL*a*b color
            dlab[t, y, x, 1], 
            dlab[t, y, x, 2], 
            ])
        extracted_data.add_row(i, row)

    extracted_data.save()
    extracted_data.clear()



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
    results = parmap.map(_get_extrema_mask, arguments, pm_pbar=True, pm_processes=multiprocessing.cpu_count()-1)
    
    for i, result in enumerate(results):
        h, w = arguments[i][1], arguments[i][2]
        mask[:,h,w] = result[0]
        persistence[:,h,w] = result[1]
    
    return (mask, persistence)



def _get_extrema_mask(args):
    
    sequence = args[3]
    extremas = RunPersistence(sequence)
    filtered_mask = [int(e[0]) for e in extremas if ((e[1] > PERSISTENCE_THRESHOLD) & ~np.isinf(e[0]))]
    filtered_threshold = [int(e[1]) if e[1] != np.inf else 0 for e in extremas if ((e[1] > PERSISTENCE_THRESHOLD) & ~np.isinf(e[0]))]
    
    mask = np.full(args[0], False)
    mask[filtered_mask] = True
    threshold = np.full(args[0], np.nan)
    threshold[filtered_mask] = np.array(filtered_threshold)
    
    return mask, threshold



if __name__ == "__main__":  
    t0 = time.time()
    args = parser.parse_args()
    Config.width, Config.height = args.width, args.height
    Config.divider = args.divider
    Config.is_relative = args.relative
    Config.device = args.device
    Config.name = args.path
    Config.keys = args.keys.split('/') if Config.device == 'keyboard' else ['']
    Config.chunk_size = args.chunk

    for k in Config.keys:

        if k== '':
            c = Converter(Config.name, ratio=1, offset=0)
            Config.output_name = "{}".format(Config.name.split('/')[-1])
        else:
            c = Converter('_'.join([Config.name, k]), ratio=1, offset=0) # this value is real resoultion length /recode resolution length
            Config.output_name = "{}_{}".format(Config.name.split('/')[-1], k)
        
        c.set_trigger(Config.device)
        c.set_position_enabled(True)

        intervals = c.get_intervals(trigger=Config.device, trim=1/2)
        if c.len_clicks < 1: continue

        v = video.Video(Config.name)
        
        chunk = np.full((Config.chunk_size, math.ceil(Config.height/Config.divider), math.ceil(Config.width/Config.divider), 3), -1)
        chunk_T = np.full(Config.chunk_size, np.nan)
        chunk_T_b = np.full(Config.chunk_size, np.nan)
        chunk_idx = 0

        idx = 0
        for frame in v.frames():
            if (type(frame) == type(None)) or (idx >= c.log.shape[0]): break  
            
            # Check which interval of button inputs includes this frame
            T_press = [] 
            for interval in intervals:
                # names and values, 
                # interval[0] <--- start, interval[1] <--- current, interval[2] <--- end
                # TODO : add description
                if idx >= interval[0] and idx <= interval[2]:
                    i = interval[1]
                    T_press.append(c.get_input_t(i))

            if T_press:
                if(len(T_press) != 1):
                    print("the number of `T_press` : ", end=" ")

                # Load data
                x_cursor, y_cursor, press, t = c.get_input(idx)
                f = frame[:Config.height:Config.divider, :Config.width:Config.divider, :]
                x_center, y_center = (f.shape[1])/2, (f.shape[0])/2

                if Config.is_relative:
                        if Config.task == 'et':
                            # manually remove the borders of video which can be artifacts.
                            f[:,:2] = 0
                            f[:2,:] = 0 
                        # Remap positions of cursor
                        x_cursor = np.interp(x_cursor, (0, Config.env_width), (0, f.shape[1]))
                        y_cursor = np.interp(y_cursor, (0, Config.env_height), (0, f.shape[0]))
                        h, w, ch = f.shape
                        f = put(f, x_center - x_cursor, y_center - y_cursor)[h//4:h//4+h, w//4:w//4+w, :]

                chunk[chunk_idx] = f
                chunk_T[chunk_idx] = c.get_input_t(idx)[0]
                chunk_T_b[chunk_idx] = T_press[0][0]
                chunk_idx += 1

            
            idx += 1
            if chunk_idx >= Config.chunk_size:
                extract(chunk_T, chunk_T_b, chunk, Config.output_name)
                chunk[0] = chunk[-1]
                chunk_T[0] = chunk_T[-1]
                chunk_T_b[0] = chunk_T_b[-1]

                #inialize
                chunk[1:] = -1
                chunk_T[1:] = np.nan
                chunk_T_b[1:] = np.nan
                
                chunk_idx = 1

        extract(chunk_T[:chunk_idx], chunk_T_b[:chunk_idx], chunk[:chunk_idx], Config.output_name)

        print("{} ==> {}".format(Config.chunk_size, time.time() - t0))