"""
This module contains all the methods used for image processing.

"""

__author__ = 'Christian Pieringer'
__created__ = '2019.05.26'

import numpy as np
from cv2 import filter2D, CV_64F
from cv2 import GaussianBlur, medianBlur
from skimage.feature import canny
from scipy.signal import convolve2d

from .helpers import kfunctions as kfunc
# from pyxvis.processing.helpers import kfunctions as kfunc


def dual_energy(img_1, img_2, lut):
    """
    Allows for dual energy image computation.

    Args:
        img_1 (ndarray): first image
        img_2 (ndarray): second image
        lut (ndarray): the lookup table use during computation

    Raises:
        TypeError: invalid input type for images

    Returns:
        _dual_energy (numpy array): the dual energy image.

    """
    # Check input parameters
    if not (isinstance(img_1, np.ndarray) and isinstance(img_2, np.ndarray)):
        raise TypeError('{}/{} are not an accepted input type for img_1 or img_2'.format(type(img_1), type(img_2)))

    nrows, ncols = img_1.shape
    _dual_energy = np.zeros((nrows, ncols))

    for i in range(nrows):
        for j in range(ncols):
            _dual_energy[i, j] = lut[img_2[i, j], img_1[i, j]]

    return _dual_energy


def shading(mat_x, mat_r1, mat_r2, i1, i2):
    """
    This method performs a shading correction. There are two reference images: 
    image R1, of a thin plate, image r2, of a thick plate. The gray values i1 
    and i2 are the ideal gray values for the first and second image, respectively.
    From r1, r2, i1 and i2, offset and gain, correction matrices A and B are 
    calculated assuming a linear transformation between the original X-ray 
    image X and corrected X-ray image y.
    """

    mat_a = (i2 - i1) / (mat_r2 - mat_r1)
    mat_b = i1 - (mat_r1 * mat_a)
    mat_y = (mat_a * mat_x) + mat_b

    return mat_y


def fspecial(kernel_name, size=3, sigma=1.0):
    """
    This method mimic the 'fspecial' gaussian MATLAB function to create 
    special kernel functions used in computer vision.

    Reference: https://src-bin.com/en/q/1064ef9

    :param kernel_name: a string to define the type of window:
                'sobel': Sobel kernel
                'prewitt': Prewitt kernel
                'gaussian': standard Gaussian window
                'log': Laplacian of Gaussian Kernel

    :param size: integer to define the length of the window
    :param sigma: a float number to define the sigma of the kernel
    """

    _kernel_kwarg = {
        'sobel': kfunc.sobel_kernel,
        'prewitt': kfunc.prewitt_kernel,
        'gaussian': kfunc.gaussian_kernel,
        'log': kfunc.log_kernel
    }

    # _img_generate kernel function
    kernel = _kernel_kwarg[kernel_name](size, sigma)

    return kernel


# def bwareaopen(img, area, connectivity=4):
#     """
#     This method emulate the behavior of the Matlab(c) 'bwareaopen'.
#
#     :param img: a binary ndarray that holds
#     :param area:
#     :param connectivity:
#
#     :returns: a binary ndarray with the result
#
#     """
#     num_labels, labels, stats, centroids = connectedComponentsWithStats(
#         img.astype(np.uint8), connectivity)
#     bw_labels = np.zeros_like(labels, dtype=np.uint8)
#
#     # Loop throught labels and remove areas according to the size
#     #     areas = stats[1:, cv.CC_STAT_AREA]  #[s[4] for s in stats]
#     #     print(areas[areas > area])
#     #     sorted_idx = np.argsort(areas)
#
#     for idx in range(1, num_labels - 1):
#         x, y, w, h, size = stats[idx, :]
#
#         if size > area:
#             bw_labels[labels == idx] = 1
#
#     return bw_labels


def im_grad(img, kernel):
    """
    This method compute the gradient of a image 'img'. It uses the OpenCV
    function "filter2D". In the filter function 'same' is the default and the
    only one option.

    :param img: a uint8 input image.
    :param kernel: a kernel as nd-array. Commonly used kernels are Prewitt,
                    _img_gaussians, or Sobel.

    :returns: The images of magnitude (mag) and angle (angle).

    """

    # TODO: check arguments

    gradi = filter2D(img.astype('double'), CV_64F, kernel)
    gradj = filter2D(img.astype('double'), CV_64F, kernel.T)

    mag = np.sqrt(gradi ** 2 + gradj ** 2)
    angle = np.arctan2(gradj, gradi)

    return mag, angle


def linimg(img, t=255):
    """
    This method computes the Linear enhancement of image I from 0 to 255,
    or 0 to 1.

    :param img: a grayvalue image I.
    :param t: if t==255 output will be a unit8 image with gray values going from
              0 to 255. Else, output will a double image with gray values from
              0 to t.

    :returns: enhanced image J (uint8 or double) so that:
                J = m*I + b; where min(J) = 0, max(J) = 255 or 1.
    
    """

    img = np.double(img.copy())
    mi = np.min(img.flatten())
    mx = np.max(img.flatten())
    d = mx - mi

    if d == 0:
        out = (img / mx) * (t / 2)
    else:
        out = ((img - mi) / d) * t

    if t == 255:
        out = np.round(out)

    return out.astype('uint8')


def im_median(_img, k):
    """
    Applies median filter to an gray-scaled image.

    Args:
    _img (ndarray): a gray-scale image.
    k (int): value to define number of pixels of the median mask
    
    Returns:
     img_j (ndarray): filtered image.
    """
    img_j = medianBlur(_img, k)

    return img_j


def im_gaussian(_img, k, sigma=None):
    """
    This method applies a gaussian filtering on image _img.

    Args:
        _img (ndarray): a gray-valued image
        k (int): value to define the mask size
        sigma (float): value to control the sigma parameter. Default k/8.5.
    
    Returns:
        img_j (ndarray): filtered image
    """
    if not sigma:
        sigma = k / 8.5

    img_j = GaussianBlur(_img, (k, k), sigma)

    return img_j


def im_equalization(img, gamma=1.0):
    """
    TODO: write documentation.
    """

    img_x = img.copy()
    img_x = 255 * ((img_x - img_x.min()) / (img_x.max() - img_x.min())) ** gamma

    return img_x


def im_average(input_img, k):
    """
    It performs a linear average filtering of "input_img".
    
    Args:
        img (array): a grayvalue image.
        k (int): size of the average window  

    Returns:
        out_img (array): filtered image so that out_img is convolution of "input_img" with an average 
                         mask h = ones(k,k)/k^2.
    """
    if not isinstance(input_img, np.ndarray):
        raise TypeError('Function expects an ndarray for input image but received {}.'.format(type(input_img)))

    if not isinstance(k, int):
        raise TypeError('Function expects an ndarray for k but received {}.'.format(type(k)))

    h = np.ones((k, k)) / (k ** 2)
    out_img = convolve2d(np.double(input_img), h, 'same')

    return out_img


def hist_forceuni(img, show=False):
    """
    TODO: write documentation.
    """

    img_x = im_equalization(img)
    n, m = img_x.shape
    y = np.zeros((n * m, 1), dtype=np.uint8)
    j = np.argsort(img_x.flatten())
    z = np.zeros((n * m, 1), dtype=np.uint8)
    d = np.int(np.fix((n * m / 256) + 0.5))

    for i in range(255):
        z[i * d:(i + 1) * d] = i * np.ones((d, 1))  # , dtype=np.uint8)
    z[255 * d:n * m] = 255 * np.ones((n * m - 255 * d, 1))  # , dtype=np.uint8)

    y[j] = z
    y = y.reshape(n, m)

    return y


class Edge:
    """
    This class implements methods for edge detection required for examples used 
    in the book.
    """

    def __init__(self, _method, _thres=None, _sigma=1.0):
        """
        Args:
            method (str): It defines the detector method:
                'log': Laplacian of Gaussian
                'canny': Canny edge detection
            thres (float): It defines the threshold to noise supression
            sigma (float): It specifies the scale of the kernel
        """

        _valid_methods = ['log', 'canny', 'sobel']

        super().__init__()

        # Check input parameters
        if _method not in _valid_methods:
            raise ValueError('{} is not a valid method for edge detection.'.format(_method))

        self.method = _method
        self.thres = _thres
        self.sigma = _sigma
        self._returns = None  # Keep track outputs

    def detect_zero_crossing(self, edges):
        """
        Performs zero crossing detection.
        
        Args:
            edges (float ndarray): input image in edges
    
        Return:
            out_img (ndarray): image of edges.
        """

        # def any_neighbor_zero(img, i, j):
        #     """
        #     This function verifies neighborhood around boundary pixels.
        #
        #     """
        #     for k in range(-1, 2):
        #         for l in range(-1, 2):
        #             if img[i + k, j + k] == 0:
        #                 return True
        #     return False
        #
        # # Makes a copy of the original image and
        # # marks as 1 pixels greater that 0.
        # _edges = edges.copy()
        # _edges[_edges > self.thres] = 1
        # _edges[_edges <= self.thres] = 0
        #
        # # _edges = np.abs(_edges)
        # # _edges = _edges > self.thres
        #
        # # print(np.max(_edges), np.min(_edges))
        #
        # out_img = np.zeros(_edges.shape)
        #
        # for i in range(1, _edges.shape[0] - 1):
        #     for j in range(1, _edges.shape[1] - 1):
        #         if _edges[i, j] > 0 and any_neighbor_zero(_edges, i, j):
        #             out_img[i, j] = 1

        # Check zero crossing four directions and aggregate the output.
        out = np.diff(np.sign(edges), prepend=np.ones((edges.shape[0], 1)), axis=1) != 0
        out = np.bitwise_or(out, np.diff(np.sign(edges), prepend=np.ones((1, edges.shape[1])), axis=0) != 0)
        out = np.bitwise_or(out, (np.diff(np.sign(edges)[:, ::-1], prepend=np.ones((1, edges.shape[1])), axis=0) != 0)[:, ::-1])
        out = np.bitwise_or(out, (np.diff(np.sign(edges)[::-1, :], prepend=np.ones((1, edges.shape[1])), axis=0) != 0)[::-1, :])
        # Check gradients is on the threshold.

        grads_diff = np.abs(np.diff(edges, prepend=np.ones((edges.shape[0], 1)), axis=1)) > self.thres
        grads_diff = np.bitwise_or(grads_diff, np.abs(np.diff(edges, prepend=np.ones((1, edges.shape[0])), axis=0)) > self.thres)
        # grads_diff = np.bitwise_or(grads_diff, (np.abs(
        #     np.diff(edges[:, ::-1], prepend=np.ones((edges.shape[0], 1)),
        #             axis=1)) > self.thres)[:, ::-1])
        # grads_diff = np.bitwise_or(grads_diff, (np.abs(
        #     np.diff(edges[::-1, :], prepend=np.ones((1, edges.shape[0])),
        #             axis=0)) > self.thres)[::-1, :])

        out_img = np.bitwise_and(out, grads_diff)

        return out_img

    def fit(self, img):
        """
        This method detect edges in an image. Method must be choose during class
        instantiation.

        Args:
            img: input image in numpy array format.

        Returns:
            None
        """

        self._returns = dict()
        self._returns.update({'edges': None})

        if self.method == 'log':
            _ksize = int(2 * np.ceil(3 * self.sigma) + 1)
            _log_kernel = fspecial(self.method, _ksize, self.sigma)
            _log = filter2D(img.astype('double'), CV_64F, _log_kernel)
            _log = _log / np.sum(_log.flatten())  # Normalize gradients to sum 0
            _edges = self.detect_zero_crossing(_log)

            self._returns.update({'kernel': _log_kernel})  # Keep used kernel

        elif self.method == 'canny':
            """
            Canny edge detector. This option implement a wrapper to apply the 
            scikit image implementation for this edge detector. It present an 
            interface with similar parameters to the MATLAB version.    
            """
            _edges = canny(
                img.astype('double'),
                sigma=self.sigma,
                low_threshold=self.thres,
                high_threshold=self.thres / 0.4
            )

        # Keep the output if does exists
        self._returns.update({'edges': _edges})

    @property
    def edges(self):
        return self._returns['edges']

    @property
    def kernel(self):
        return self._returns['kernel'] if self.method != 'canny' else None


def res_minio(_img_g, h, method='minio'):
    """

    Args:
        _img_g: blured image as ndarray.
        h: PSF as ndarray.
        method: integer value to select minimization method:
                'minio': ||f_n - g|| -> min
                'other': ||f|| -> min

    Returns:
        _img_f: the restored image as ndarray.
    
    Raises:
        ValueError: if h is not ndarray.
        Exception
    """

    _valid_methods = ['minio', 'other']

    # Check input parameters
    # Input image _img_g
    if not isinstance(_img_g, np.ndarray):
        raise Exception('{} is not valid input format for the blured image _img_g.'.format(type(_img_g)))

    # PSF h
    if not isinstance(h, np.ndarray):
        raise Exception('{} is not a valid format for PSF.'.format(type(h)))

    # Method
    if method not in _valid_methods:
        raise ValueError('method do not meet {} as a valid method'.format(method))

    # Main
    nh = h.shape[-1]
    n = _img_g.shape[-1]
    m = n + nh - 1
    _mat_h = np.zeros((n, m))

    # Composing H matrix
    for i in range(n):
        _mat_h[i, i:i + nh] = h.ravel()

    _lambda = 1e6  # Lagrange multiplier

    if method == 'minio':
        _mat_p = np.hstack([np.eye(n, dtype='float'), np.zeros((n, nh - 1))])
        _img_f = _img_g @ (_lambda * _mat_h + _mat_p)
        _img_f = _img_f @ (np.linalg.inv((_lambda * _mat_h).T @ _mat_h + _mat_p.T @ _mat_p)).T
    else:
        _img_f = _img_g @ _mat_h
        _img_f = _img_f @ (np.linalg.inv((_lambda * _mat_h).T @ _mat_h + np.eye(m, dtype='float'))).T

    # Clip the image between 0 and 255
    _img_f[_img_f < 0] = 0
    _img_f[_img_f > 255] = 255

    return _img_f

# Added by D.Mery on June, 2nd, 2020

from scipy.ndimage.filters import convolve

def conv2(x, y, mode='same'):
    """
    Emulate the function conv2 from Mathworks.

    Args:
        x: 
    Usage:

    z = conv2(x,y,mode='same')

    TODO: 
     - Support other modes than 'same' (see conv2.m)
    """

    if not(mode == 'same'):
        raise Exception("Mode not supported")

    # Add singleton dimensions
    if (len(x.shape) < len(y.shape)):
        dim = x.shape
        for i in range(len(x.shape),len(y.shape)):
            dim = (1,) + dim
        x = x.reshape(dim)
    elif (len(y.shape) < len(x.shape)):
        dim = y.shape
        for i in range(len(y.shape),len(x.shape)):
            dim = (1,) + dim
        y = y.reshape(dim)

    origin = ()

    # Apparently, the origin must be set in a special way to reproduce the results of scipy.signal.convolve and Matlab.
    for i in range(len(x.shape)):
        if ( (x.shape[i] - y.shape[i]) % 2 == 0 and
             x.shape[i] > 1 and
             y.shape[i] > 1):
            origin = origin + (-1,)
        else:
            origin = origin + (0,)

    z = convolve(x, y, mode='constant', origin=origin)

    return z


def gradlog(img, sigma, th):
    """
    Edge detection using LoG and gradient.

    Args:
        img (ndarray): input image
        sigma (float): sigma parameters for Edge instance
        th (float): threshold

    Returns:
        edge (ndarray): edge detection. It is or(E_log, E_grad), where E_log is edge detection using LoG, and E_grad is
                        edge detection using gradient of I thresholding by th. Gradient of I is computed after Gaussian 
                        low pass filtering (using sigma).
    """
    detector = Edge('log', 1e-10, sigma)
    detector.fit(img)
    elog = detector.edges
    hsize = np.round((8.5 * sigma) / 2) * 2 + 1
    g = kfunc.gaussian_kernel(hsize, sigma)
    kernel = np.array([-1.0, 1.0])
    h = conv2(g, kernel)
    grad, _ = im_grad(img,h)
    egrad = grad > th
    edge = egrad | elog

    return edge

# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# This file was originally part of the octave-forge project
# Ported to python by Luis Pedro Coelho <luis@luispedro.org> (February 2008)
# Copyright (C) 2006       Soren Hauberg
# Copyright (C) 2008-2010  Luis Pedro Coelho (Python port)
# 
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2, or (at your option)
# any later version.
# 
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details. 
# 
# You should have received a copy of the GNU General Public License
# along with this file.  If not, see <http://www.gnu.org/licenses/>.

# changed by skimage.segmentation.find_boundaries
#def bwperim(bw, n=4):
#    """
#    Find the perimeter of objects in binary images. A pixel is part of an object perimeter if its value is one and 
#    there is at least one zero-valued pixel in its neighborhood. By default the neighborhood of a pixel is 4 nearest 
#    pixels, but if `n` is set to 8 the 8 nearest pixels will be considered.
#
#    Args:
#        bw (ndarray): the input binary image
#        n (int): Connectivity. Must be 4 or 8 (default: 4)
#    
#    Returns:
#        perim (bool): A binary image with the perimeter of the input mask.
#
#    Raises:
#        ValueError: Not valid values for n. Method requires that n be 4 or 8.
#    """
#    if n not in (4, 8):
#        raise ValueError('bwperim: Not valid value for n. Method expect n equals to 4 or 8.')
#
#    rows,cols = bw.shape
#
#    # Translate image by one pixel in all directions
#    north = np.zeros((rows,cols))
#    south = np.zeros((rows,cols))
#    west = np.zeros((rows,cols))
#    east = np.zeros((rows,cols))
#
#    north[:-1,:] = bw[1:,:]
#    south[1:,:]  = bw[:-1,:]
#    west[:,:-1]  = bw[:,1:]
#    east[:,1:]   = bw[:,:-1]
#    idx = (north == bw) & \
#          (south == bw) & \
#          (west  == bw) & \
#          (east  == bw)
#
#    if n == 8:
#        north_east = np.zeros((rows, cols))
#        north_west = np.zeros((rows, cols))
#        south_east = np.zeros((rows, cols))
#        south_west = np.zeros((rows, cols))
#        north_east[:-1, 1:]   = bw[1:, :-1]
#        north_west[:-1, :-1] = bw[1:, 1:]
#        south_east[1:, 1:]     = bw[:-1, :-1]
#        south_west[1:, :-1]   = bw[:-1, 1:]
#        idx &= (north_east == bw) & \
#               (south_east == bw) & \
#               (south_west == bw) & \
#               (north_west == bw)
#
#    return ~idx * bw
#