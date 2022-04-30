import numpy as np
import cv2 as cv



class Converter():

    result = np.array([])
    position_enabled = True
    len_clicks = 0



    def __init__(self, name, ratio, offset = 0, type='mouse'):
        self.name = name
        self.log = np.genfromtxt(name + '.csv', delimiter=",")[offset:]
        self.scale = 1/ratio # if the size of recoded-screen were scaled, this value should be multiplied to cursor positions
        self.trigger = 'mouse'


    def set_trigger(self, _trigger):
        '''
        'mouse', 'keyboard', 'both' can be an argument.
        '''
        self.trigger = _trigger


    def set_position_enabled(self, _enabled:bool):
        self.position_enabled = _enabled


    def get_inputs(self):
        if(self.position_enabled):
            X = self.log[:, 0] * self.scale
            Y = self.log[:, 1] * self.scale
        else:
            X, Y = None, None
        
        pressed = None
        if self.trigger == 'mouse':
            pressed = self.log[:, 2]
        elif self.trigger == 'keyboard':
            pressed = self.log[:, 3]
        elif self.trigger == 'both':
            pressed = self.log[:, 2] + self.log[:, 3]

        T = self.log[:, -2]
        return X, Y, pressed, T


    def get_input(self, idx):
        if(self.position_enabled):
            x = self.log[idx, 0] * self.scale
            y = self.log[idx, 1] * self.scale
        else:
            x, y = None, None

        pressed = None
        if self.trigger == 'mouse':
            pressed = self.log[idx, 2]
        elif self.trigger == 'keyboard':
            pressed = self.log[idx, 3]
        elif self.trigger == 'both':
            pressed = self.log[idx, 2] + self.log[idx, 3]

        t = self.log[idx, -2]

        return x, y, pressed, t


    def get_input_t(self, idx):
        t = self.log[idx, -2]
        button = self.log[idx, 3] if self.trigger == 'keyboard' else self.log[idx, 2]
        return t, button


    def get_intervals(self, trim=1/2, trigger='mouse'):

        if trigger == 'both':
            indices = np.where((self.log[:,2] == 1 | self.log[:,3] != ""))
        elif trigger == 'mouse':
            indices = np.where((self.log[:,2] == 1))[0]
        elif trigger == 'keyboard':            
            indices = np.where((self.log[:,3] != 0))[0]
        elif trigger == 'key-enter':
            raw = self.log[:,3]
            mask_pressed = (raw[1:] - raw[:-1] == 1)
            pressed = np.zeros(len(raw))
            print(pressed)
            pressed[1:][mask_pressed] = 1
            indices = np.where(pressed == 1)[0]
        else:
            raise ValueError

        self.len_clicks = len(indices)
        g_start, g_end  = 0, len(self.log) - 1 # indices
        idx = 0

        intervals = []

        while True:
            if idx >= self.len_clicks: 
                return intervals

            current_press = indices[idx]

            if(idx == 0): # When it starts
                prev_press = g_start
                next_press = indices[idx+1] if len(indices) >= 2 else current_press + 360
            elif (idx == self.len_clicks-1): # When it ends
                prev_press = indices[idx-1]
                next_press = g_end
            else:
                prev_press = indices[idx-1]
                next_press = indices[idx+1]
            idx += 1

            before_current = min(180, abs(current_press - prev_press) * trim) # 180 means 3sec
            after_current = min(180, abs(next_press - current_press) * trim) # 180 means 3sec
            start = int(current_press - before_current)
            end = int(current_press + after_current) -1

            intervals.append((start, current_press, end))
    