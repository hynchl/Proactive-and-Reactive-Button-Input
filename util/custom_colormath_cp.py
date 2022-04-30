import numpy as np
import cupy as cp
from colormath import color_constants
from colormath.color_objects import LabColor, sRGBColor, ColorBase, BaseRGBColor
from colormath.color_conversions import convert_color, apply_RGB_matrix
from colormath.color_objects import ColorBase,  sRGBColor, LabColor, BaseRGBColor, IlluminantMixin
from colormath.chromatic_adaptation import apply_chromatic_adaptation_on_color
from colormath.color_diff import delta_e_cie2000
import time
cp.cuda.Device(0).use()



class CColorBase(ColorBase):
    """
    A base class holding some common methods and values.
    """

    # Attribute names containing color data on the sub-class. For example,
    # sRGBColor would be ['rgb_r', 'rgb_g', 'rgb_b']
    VALUES = []
    # If this object as converted such that its values passed through an
    # RGB colorspace, this is set to the class for said RGB color space.
    # Allows reversing conversions automatically and accurately.
    _through_rgb_type = None


    def __str__(self):
        """
        String representation of the color.
        """
        retval = self.__class__.__name__ + ' ('
        for val in self.VALUES:
            value = getattr(self, val, None)
            if value is not None:
                retval += '\n{}:{}'.format(val, getattr(self, val))
        return retval.strip() + ')'

    def get_value_tuple(self):
        """
        Returns a tuple of the color's values (in order). For example,
        an LabColor object will return (lab_l, lab_a, lab_b), where each
        member of the tuple is the float value for said variable.
        """
        retval = tuple()
        for val in self.VALUES:
            retval += (getattr(self, val),)
        return retval

class CBaseRGBColor(CColorBase):
    """
    Base class for all RGB color spaces.

    .. warning:: Do not use this class directly!
    """

    VALUES = ['rgb_r', 'rgb_g', 'rgb_b']

    def __init__(self, rgb_r, rgb_g, rgb_b, is_upscaled=False):
        """
        :param float rgb_r: R coordinate. 0...1. 1-255 if is_upscaled=True.
        :param float rgb_g: G coordinate. 0...1. 1-255 if is_upscaled=True.
        :param float rgb_b: B coordinate. 0...1. 1-255 if is_upscaled=True.
        :keyword bool is_upscaled: If False, RGB coordinate values are
            beteween 0.0 and 1.0. If True, RGB values are between 1 and 255.
        """
        super(CBaseRGBColor, self).__init__()
        if is_upscaled:
            self.rgb_r = rgb_r / 255.0
            self.rgb_g = rgb_g / 255.0
            self.rgb_b = rgb_b / 255.0
        else:
            self.rgb_r = rgb_r.astype(cp.float)
            self.rgb_g = rgb_g.astype(cp.float)
            self.rgb_b = rgb_b.astype(cp.float)
        self.is_upscaled = is_upscaled

    def _clamp_rgb_coordinate(self, coord):
        """
        Clamps an RGB coordinate, taking into account whether or not the
        color is upscaled or not.

        :param float coord: The coordinate value.
        :rtype: float
        :returns: The clamped value.
        """
        if not self.is_upscaled:
            return cp.min(cp.max(coord, 0.0), 1.0)
        else:
            return cp.min(cp.max(coord, 1), 255)

    @property
    def clamped_rgb_r(self):
        """
        The clamped (0.0-1.0) R value.
        """
        return self._clamp_rgb_coordinate(self.rgb_r)

    @property
    def clamped_rgb_g(self):
        """
        The clamped (0.0-1.0) G value.
        """
        return self._clamp_rgb_coordinate(self.rgb_g)

    @property
    def clamped_rgb_b(self):
        """
        The clamped (0.0-1.0) B value.
        """
        return self._clamp_rgb_coordinate(self.rgb_b)

    def get_upscaled_value_tuple(self):
        """
        Scales an RGB color object from decimal 0.0-1.0 to int 0-255.
        """
        # Scale up to 0-255 values.
        rgb_r = (cp.floor(0.5 + self.rgb_r * 255)).astype(cp.int)
        rgb_g = (cp.floor(0.5 + self.rgb_g * 255)).astype(cp.int)
        rgb_b = (cp.floor(0.5 + self.rgb_b * 255)).astype(cp.int)

        return rgb_r, rgb_g, rgb_b

class CXYZColor(IlluminantMixin, CColorBase):
    """
    Represents an XYZ color.
    """

    VALUES = ['xyz_x', 'xyz_y', 'xyz_z']

    def __init__(self, xyz_x, xyz_y, xyz_z, observer='2', illuminant='d50'):
        """
        :param float xyz_x: X coordinate.
        :param float xyz_y: Y coordinate.
        :param float xyz_z: Z coordinate.
        :keyword str observer: Observer angle. Either ``'2'`` or ``'10'`` degrees.
        :keyword str illuminant: See :doc:`illuminants` for valid values.
        """
        super(CXYZColor, self).__init__()
        #: X coordinate
        self.xyz_x = xyz_x.astype(cp.float)
        #: Y coordinate
        self.xyz_y = xyz_y.astype(cp.float)
        #: Z coordinate
        self.xyz_z = xyz_z.astype(cp.float)

        #: The color's observer angle. Set with :py:meth:`set_observer`.
        self.observer = None
        #: The color's illuminant. Set with :py:meth:`set_illuminant`.
        self.illuminant = None

        self.set_observer(observer)
        self.set_illuminant(illuminant)

    def apply_adaptation(self, target_illuminant, adaptation='bradford'):
        """
        This applies an adaptation matrix to change the XYZ color's illuminant.
        You'll most likely only need this during RGB conversions.
        """
        # logger.debug("  \- Original illuminant: %s", self.illuminant)
        # logger.debug("  \- Target illuminant: %s", target_illuminant)

        # If the XYZ values were taken with a different reference white than the
        # native reference white of the target RGB space, a transformation matrix
        # must be applied.
        if self.illuminant != target_illuminant:
            # logger.debug("  \* Applying transformation from %s to %s ",
            #              self.illuminant, target_illuminant)
            # Sets the adjusted XYZ values, and the new illuminant.
            apply_chromatic_adaptation_on_color(
                color=self,
                targ_illum=target_illuminant,
                adaptation=adaptation)

class CLabColor(IlluminantMixin, CColorBase):
    """
    Represents a CIE Lab color. For more information on CIE Lab,
    see `Lab color space <http://en.wikipedia.org/wiki/Lab_color_space>`_ on
    Wikipedia.
    """

    VALUES = ['lab_l', 'lab_a', 'lab_b']

    def __init__(self, lab_l, lab_a, lab_b, observer='2', illuminant='d50'):
        """
        :param float lab_l: L coordinate.
        :param float lab_a: a coordinate.
        :param float lab_b: b coordinate.
        :keyword str observer: Observer angle. Either ``'2'`` or ``'10'`` degrees.
        :keyword str illuminant: See :doc:`illuminants` for valid values.
        """
        super(CLabColor, self).__init__()
        #: L coordinate
        self.lab_l = lab_l.astype(cp.float)
        #: a coordinate
        self.lab_a = lab_a.astype(cp.float)
        #: b coordinate
        self.lab_b = lab_b.astype(cp.float)

        #: The color's observer angle. Set with :py:meth:`set_observer`.
        self.observer = None
        #: The color's illuminant. Set with :py:meth:`set_illuminant`.
        self.illuminant = None

        self.set_observer(observer)
        self.set_illuminant(illuminant)



def _get_lab_color1_vectors(colors):
    """
    Converts an LabColor into a NumPy vector.

    :param LabColor color:
    :rtype: numpy.ndarray
    """
    if not colors.__class__.__name__ == 'CLabColor':
        raise ValueError(
            "Delta E functions can only be used with two LabColor objects.")
    return cp.array([colors.lab_l, colors.lab_a, colors.lab_b]).T

def _get_lab_color2_matrices(colors):
    """
    Converts an LabColor into a NumPy matrix.

    :param LabColor color:
    :rtype: numpy.ndarray
    """
    if not colors.__class__.__name__ == 'CLabColor':
        raise ValueError(
            "Delta E functions can only be used with two LabColor objects.")
    # return cp.array([[(color.lab_l, color.lab_a, color.lab_b)] for color in colors2])
    return cp.array([colors.lab_l, colors.lab_a, colors.lab_b]).T

def convert_color_cp(color, target_cs, through_rgb_type=sRGBColor,
                  target_illuminant=None, *args, **kwargs):

    if isinstance(target_cs, str):
        raise ValueError("target_cs parameter must be a Color object.")
    if not issubclass(target_cs, ColorBase):
        raise ValueError("target_cs parameter must be a Color object.")

    conversions = [RGB_to_XYZ, XYZ_to_Lab]
    if issubclass(target_cs, BaseRGBColor):
        through_rgb_type = target_cs
    target_rgb = through_rgb_type

    new_color = color
    for func in conversions:
        if func:
            new_color = func(
                new_color,
                target_rgb=target_rgb,
                target_illuminant=target_illuminant,
                *args, **kwargs)

    return new_color
    
def XYZ_to_Lab(cobj, *args, **kwargs):
    """
    Converts XYZ to Lab.
    """

    # print("XYZ-to-Lab")
    illum = cobj.get_illuminant_xyz()
    temp_x, ttemp_x = cobj.xyz_x / illum["X"], cobj.xyz_x / illum["X"]
    temp_y, ttemp_y = cobj.xyz_y / illum["Y"], cobj.xyz_y / illum["Y"]
    temp_z, ttemp_z = cobj.xyz_z / illum["Z"], cobj.xyz_z / illum["Z"]

    mask_x = temp_x > color_constants.CIE_E
    temp_x[mask_x] = cp.power(ttemp_x[mask_x], (1.0 / 3.0))
    temp_x[~mask_x] = (7.787 * ttemp_x[~mask_x]) + (16.0 / 116.0)
    # if temp_x > color_constants.CIE_E:
    #     temp_x = cp.power(temp_x, (1.0 / 3.0))
    # else:
    #     temp_x = (7.787 * temp_x) + (16.0 / 116.0)

    mask_y = temp_y > color_constants.CIE_E
    temp_y[mask_y] = cp.power(ttemp_y[mask_y], (1.0 / 3.0))
    temp_y[~mask_y] = (7.787 * ttemp_y[~mask_y]) + (16.0 / 116.0)
    # if temp_y > color_constants.CIE_E:
    #     temp_y = cp.power(temp_y, (1.0 / 3.0))
    # else:
    #     temp_y = (7.787 * temp_y) + (16.0 / 116.0)

    mask_z = temp_z > color_constants.CIE_E
    temp_z[mask_z] = cp.power(ttemp_z[mask_z], (1.0 / 3.0))
    temp_z[~mask_z] = (7.787 * ttemp_z[~mask_z]) + (16.0 / 116.0)
    # if temp_z > color_constants.CIE_E:
    #     temp_z = cp.power(temp_z, (1.0 / 3.0))
    # else:
    #     temp_z = (7.787 * temp_z) + (16.0 / 116.0)

    lab_l = (116.0 * temp_y) - 16.0
    lab_a = 500.0 * (temp_x - temp_y)
    lab_b = 200.0 * (temp_y - temp_z)

    return CLabColor(
        lab_l, lab_a, lab_b, observer=cobj.observer, illuminant=cobj.illuminant)

def RGB_to_XYZ(cobj, target_illuminant=None, *args, **kwargs):
    """
    RGB to XYZ conversion. Expects 0-255 RGB values.

    Based off of: http://www.brucelindbloom.com/index.html?Eqn_RGB_to_XYZ.html
    """

    # Will contain linearized RGB channels (removed the gamma func).
    linear_channels = {}

    if isinstance(cobj, sRGBColor):
        for channel in ['r', 'g', 'b']:
            V = getattr(cobj, 'rgb_' + channel)
            A = cp.zeros(V.shape)
            mask =  V<=0.04045
            r1 = V / 12.92
            r2 = cp.power((V + 0.055) / 1.055, 2.4)
            A[mask] = r1[mask]
            A[~mask] = r2[~mask]
            linear_channels[channel] = A

            # if V <= 0.04045:
            #     linear_channels[channel] = V / 12.92
            # else:
            #     linear_channels[channel] = cp.power((V + 0.055) / 1.055, 2.4)
    else:
        # If it's not sRGB...
        gamma = cobj.rgb_gamma

        for channel in ['r', 'g', 'b']:
            V = getattr(cobj, 'rgb_' + channel)
            linear_channels[channel] = cp.power(V, gamma)

    # Apply an RGB working space matrix to the XYZ values (matrix mul).
    xyz_x, xyz_y, xyz_z = apply_RGB_matrix(
        linear_channels['r'], linear_channels['g'], linear_channels['b'],
        rgb_type=cobj, convtype="rgb_to_xyz")

    if target_illuminant is None:
        target_illuminant = cobj.native_illuminant

    # The illuminant of the original RGB object. This will always match
    # the RGB colorspace's native illuminant.
    illuminant = cobj.native_illuminant
    xyzcolor = CXYZColor(xyz_x, xyz_y, xyz_z, illuminant=illuminant)
    # This will take care of any illuminant changes for us (if source
    # illuminant != target illuminant).
    xyzcolor.apply_adaptation(target_illuminant)

    return xyzcolor

def apply_RGB_matrix(var1, var2, var3, rgb_type, convtype="xyz_to_rgb"):
    """
    Applies an RGB working matrix to convert from XYZ to RGB.
    The arguments are tersely named var1, var2, and var3 to allow for the
    passing of XYZ _or_ RGB values. var1 is X for XYZ, and R for RGB. var2 and
    var3 follow suite.
    """
    convtype = convtype.lower()
    # Retrieve the appropriate transformation matrix from the constants.
    rgb_matrix = cp.asarray(rgb_type.conversion_matrices[convtype])

    # Stuff the RGB/XYZ values into a NumPy matrix for conversion.
    var_matrix = cp.array((
        var1, var2, var3
    ))

    result_matrix = cp.dot(rgb_matrix, var_matrix)
    rgb_r, rgb_g, rgb_b = result_matrix[0,:], result_matrix[1,:], result_matrix[2,:], 

    # Clamp these values to a valid range.
    rgb_r[rgb_r<0] = 0.0 # rgb_r = cp.max(rgb_r, 0.0)
    rgb_g[rgb_g<0] = 0.0 # rgb_g = cp.max(rgb_g, 0.0)
    rgb_b[rgb_b<0] = 0.0 # rgb_b = cp.max(rgb_b, 0.0)
    return rgb_r, rgb_g, rgb_b

def delta_e_cie2000_cp(color1, color2, Kl=1, Kc=1, Kh=1):
    """
    Calculates the Delta E (CIE2000) of two colors.
    """
    length = len(color1.lab_l)
    color1_vector = cp.zeros((length,3)) # 임시, _get_lab_color1_vector를 최종적으로 고쳐야함
    color1_vector[:,:] = _get_lab_color1_vectors(color1)
    # print(color1_vector)

    color2_matrix = cp.zeros((length,3)) # 임시, _get_lab_color2_matrix를 최종적으로 고쳐야함
    color2_matrix[:,:] = _get_lab_color2_matrices(color2)
    # print(color2_matrix)

    delta_e = _delta_e_cie2000(color1_vector, color2_matrix, Kl=Kl, Kc=Kc, Kh=Kh)

    return delta_e #cp.asscalar(delta_e)

def _delta_e_cie2000(lab_color_vector, lab_color_matrix, Kl=1, Kc=1, Kh=1):
    """
    Calculates the Delta E (CIE2000) of two colors.
    """

    L = lab_color_vector[:,0]
    a = lab_color_vector[:,1]
    b = lab_color_vector[:,2]
    # print(L, a, b)

    avg_Lp = (L + lab_color_matrix[:, 0]) / 2.0
    # print(avg_Lp)

    C1 = cp.sqrt(cp.sum(cp.power(lab_color_vector[:, 1:], 2), axis=1))
    C2 = cp.sqrt(cp.sum(cp.power(lab_color_matrix[:, 1:], 2), axis=1))
    avg_C1_C2 = (C1 + C2) / 2.0
    # print(C1, C2, avg_C1_C2)

    G = 0.5 * (1 - cp.sqrt(cp.power(avg_C1_C2, 7.0) / (cp.power(avg_C1_C2, 7.0) + cp.power(25.0, 7.0))))
    # print(G)

    a1p = (1.0 + G) * a
    a2p = (1.0 + G) * lab_color_matrix[:, 1]
    # print(a1p, a2p)

    C1p = cp.sqrt(cp.power(a1p, 2) + cp.power(b, 2))
    C2p = cp.sqrt(cp.power(a2p, 2) + cp.power(lab_color_matrix[:, 2], 2))
    # print(C1p, C2p)

    avg_C1p_C2p = (C1p + C2p) / 2.0
    # print(avg_C1p_C2p)

    h1p = cp.degrees(cp.arctan2(b, a1p))
    h1p += (h1p < 0) * 360

    h2p = cp.degrees(cp.arctan2(lab_color_matrix[:, 2], a2p))
    h2p += (h2p < 0) * 360

    avg_Hp = (((cp.abs(h1p - h2p) > 180) * 360) + h1p + h2p) / 2.0
    # print(h1p, h2p, avg_Hp)

    T = 1 - 0.17 * cp.cos(cp.radians(avg_Hp - 30)) + \
        0.24 * cp.cos(cp.radians(2 * avg_Hp)) + \
        0.32 * cp.cos(cp.radians(3 * avg_Hp + 6)) - \
        0.2 * cp.cos(cp.radians(4 * avg_Hp - 63))
    # print(T)

    diff_h2p_h1p = h2p - h1p
    delta_hp = diff_h2p_h1p + (cp.abs(diff_h2p_h1p) > 180) * 360
    delta_hp -= (h2p > h1p) * 720
    # print(diff_h2p_h1p, delta_hp)

    delta_Lp = lab_color_matrix[:, 0] - L
    delta_Cp = C2p - C1p
    delta_Hp = 2 * cp.sqrt(C2p * C1p) * cp.sin(cp.radians(delta_hp) / 2.0)
    # print(delta_Lp, delta_Cp, delta_Hp)

    S_L = 1 + ((0.015 * cp.power(avg_Lp - 50, 2)) / cp.sqrt(20 + cp.power(avg_Lp - 50, 2.0)))
    S_C = 1 + 0.045 * avg_C1p_C2p
    S_H = 1 + 0.015 * avg_C1p_C2p * T
    # print(S_L, S_C, S_H)

    delta_ro = 30 * cp.exp(-(cp.power(((avg_Hp - 275) / 25), 2.0)))
    R_C = cp.sqrt((cp.power(avg_C1p_C2p, 7.0)) / (cp.power(avg_C1p_C2p, 7.0) + cp.power(25.0, 7.0)))
    R_T = -2 * R_C * cp.sin(2 * cp.radians(delta_ro))
    # print(delta_ro, R_C, R_T)

    return cp.sqrt(
        cp.power(delta_Lp / (S_L * Kl), 2) +
        cp.power(delta_Cp / (S_C * Kc), 2) +
        cp.power(delta_Hp / (S_H * Kh), 2) +
        R_T * (delta_Cp / (S_C * Kc)) * (delta_Hp / (S_H * Kh)))

def calculate_labcolor2(color):
    r1, g1, b1, r2, g2, b2 = color
    if (r1==r2) & (g1==g2) & (b1==b2): return 0

    t0 = time.time()
    lab1 = convert_color(sRGBColor(r1, g1, b1, is_upscaled=True), LabColor, through_rgb_type=sRGBColor)
    lab2 = convert_color(sRGBColor(r2, g2, b2, is_upscaled=True), LabColor, through_rgb_type=sRGBColor)
    return (lab1, lab2)

def calculate_dE_cp(colors:np.array):
    colors = cp.asarray(colors) # convert to cp array
    r1, g1, b1, r2, g2, b2 = colors[:,0], colors[:,1], colors[:,2], colors[:,3], colors[:,4], colors[:,5]

    srgb1 = sRGBColor(r1, g1, b1, is_upscaled=True)
    srgb2 = sRGBColor(r2, g2, b2, is_upscaled=True)
    lab1 = convert_color_cp(srgb1, LabColor, through_rgb_type=sRGBColor)
    lab2 = convert_color_cp(srgb2, LabColor, through_rgb_type=sRGBColor)
    result = delta_e_cie2000_cp(lab1, lab2)

    return result 

def calculate_dE(color):
    r1, g1, b1, r2, g2, b2 = color
    if (r1==r2) & (g1==g2) & (b1==b2): 
        return 0
    lab1 = convert_color(sRGBColor(r1, g1, b1, is_upscaled=True), LabColor, through_rgb_type=sRGBColor)
    lab2 = convert_color(sRGBColor(r2, g2, b2, is_upscaled=True), LabColor, through_rgb_type=sRGBColor)
    return delta_e_cie2000(lab1, lab2)