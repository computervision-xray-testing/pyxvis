"""
This module contains a collection of special functions use generally in 
2D filtering task.

"""

__author__ = 'Christian Pieringer'
__created__ = '2020.01.14'

import numpy as np


def prewitt_kernel(size=3, *args):
    """
    This function create a Prewitt Kernel. The default size is 3 x 3 kernel.
    :param size:
    :param args:
    :return:
    """

    prewitt = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype='double')
    return prewitt


def sobel_kernel(size=3, *args):
    """
    This method create a Sobel Kernel. The standar size is a 3 x 3 kernel.

    :param size: int value to define the kernel size.

    :return:
    """

    sobel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype='double')
    return np.transpose(sobel)


def gaussian_kernel(size, sigma, *args):
    """
    This method creates a symmetric Gaussian kernel.

    :param sigma: float number that defines the width of the Gaussian
            distribution
    :param size: inf number to specify the size of the kernel.
            Typical numbers 3, 5, 9, 11.
    """
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1,
           -size // 2 + 1:size // 2 + 1]

    kernel = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    kernel = kernel / kernel.sum()

    return kernel.astype('double')


def log_kernel(size, sigma, *args):
    """
    This option creates a Laplacian of Gaussian Kernel centered at 
    the middle of the kernel

    :param size: integer value to define the size of the kernel.
    :param sigma: float value for the scale factor.
    :returns: a numpy array with the LoG kernel.
    
    """

    x, y = np.meshgrid(np.arange(-size - 1 / 2 + 1, size - 1 / 2 + 1),
                       np.arange(-size - 1 / 2 + 1, size - 1 / 2 + 1))

    # LoG filter
    # normal = -1.0 / (2.0 * np.pi * sigma ** 2)

    # kernel = ((x ** 2 + y ** 2 - (2.0 * sigma ** 2)) / sigma ** 4)
    # kernel = kernel * np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2))  
    # kernel = kernel / normal

    kernel = x ** 2 + y ** 2
    kernel = (-1 / (np.pi * sigma ** 4)) * (
                1 - kernel / (2 * sigma ** 2)) * np.exp(
        -kernel / (2 * sigma ** 2))

    # Normalize ensuring 0 DC. See Peter Kovesi's Toolbox
    kernel = kernel - np.mean(kernel.flatten())

    return kernel
