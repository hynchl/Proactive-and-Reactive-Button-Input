import numpy as np
import sys, argparse



# This script is written for prprocessing input logs of keyboard.
# In our logging system, all the pressed keys are collected as a string at every frame (e.g. 'a/s/d').
# If you want to analyze a specific button inputs, you should extract specific inputs to analyze and convert it to integer (1 or 0).
# This script is an example for extracting the button of interest. (e.g. 'a/s/d' == extracting 'a' ==> 1 ) 



parser = argparse.ArgumentParser()
parser.add_argument('--task', required=True, type=str, help='data to be analyze')



def get_key_down(inputs:np.array):
    '''extract timings when a button gets pressed
    returnsdd np.array of boolean
    '''

    prev = inputs[:-1]
    cur = inputs[1:]
    down = np.full(len(inputs), False)
    down[1:] = (prev == False) & (cur == True)
    return down



def get_key_up(inputs:np.array):
    '''extract timings when a button gets released
    '''

    prev = inputs[:-1]
    cur = inputs[1:]
    up = np.full(len(inputs), False)
    up[1:] = (prev == True) & (cur == False)
    return up



def extract_key_pressed(key_inputs:np.array, key:str):
    '''parse string(pressed inputs) to array 
    '''

    return np.asarray([(key in key_input.split('/')) for key_input in key_inputs])



def execute_dino_preprocessing():
    ''' extract 'space' input signals
    '''

    PID = ["".join(['S', str(200+i)]) for i in range(1, 21)]
    for pid in PID:
        data = np.loadtxt('rawdata/dino/{}.csv'.format(pid), delimiter=',', dtype=str)
        key_inputs = data[:,3]
        target_key = extract_key_pressed(key_inputs, 'space')
        target_key_down = get_key_down(target_key)
        new_key_inputs = np.zeros(len(key_inputs))
        new_key_inputs[target_key_down] = 1
        data[:,3] = new_key_inputs

        np.savetxt("rawdata/dino/{}_space.csv".format(pid), data, fmt='%s', delimiter=',')



def execute_tabsonic_preprocessing():
    '''extract 's', 'd', 'f', 'j', 'k', and 'l' input signals. 
    each key is encoded in binary
    '''

    PID = ["".join(['S', str(200+i)]) for i in range(1, 21)]
    MUSIC = ['A', 'B', 'C', 'D', 'E']
    SOUND = ['O', 'X']
    KEY = ['s', 'd', 'f', 'j', 'k', 'l']

    for pid in PID:
        for music in MUSIC:
            for sound in SOUND:
                data = np.loadtxt('rawdata/tabsonic/{}_{}{}.csv'.format(pid, music, sound), delimiter=',', dtype=str)
                key_inputs = data[:,3] #get raw data
                new_key_inputs = np.zeros(len(key_inputs)) # create an empty column

                for i, k in enumerate(KEY):
                    # 1. for each key, extract timing at which the target key is down
                    # 2. update 'new key inputs' after encoding
                    target_key = extract_key_pressed(key_inputs, k)
                    target_key_down = get_key_down(target_key)
                    new_key_inputs[target_key_down] += 2**i 
                
                # save data
                data[:,3] = new_key_inputs
                np.savetxt('rawdata/{}_{}{}_all.csv'.format(pid, music, sound), data, fmt='%s', delimiter=',')



def execute_custom_preprocessing():
    '''write your own if needed
    '''
    pass



if __name__ == "__main__":

    args = parser.parse_args()
    
    if args.task == 'dino':
        execute_dino_preprocessing()
    elif args.task == 'tabsonic':
        execute_tabsonic_preprocessing()
    else:
        execute_custom_preprocessing()