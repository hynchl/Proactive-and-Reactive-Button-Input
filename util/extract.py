import numpy as np
import cupy as cp
from colormath.color_objects import LabColor, sRGBColor
from util.custom_colormath_cp import convert_color_cp

def rgb_to_lab(rgb):
    original_shape = rgb.shape
    rgb = cp.asarray(rgb)
    rgb = rgb.reshape((-1, 3))
    srgb = sRGBColor(rgb[:,0], rgb[:,1], rgb[:,2], is_upscaled=True)
    color = convert_color_cp(srgb, LabColor, through_rgb_type=sRGBColor)
    lab = cp.vstack((color.lab_l, color.lab_a, color.lab_b)).T
    result = lab.reshape(original_shape)
    return result.get()
