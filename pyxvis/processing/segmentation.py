"""
This module contains all the methods related to images segmentation.

"""

__author__ = 'Christian Pieringer'
__created__ = '2019.12.26'

import numpy as np

from cv2 import MSER_create
from skimage.filters import threshold_otsu, gaussian
from skimage.measure import find_contours
from skimage.morphology import binary_closing
from skimage.morphology import binary_dilation
from skimage.morphology import disk
from skimage.morphology import flood
from skimage.morphology import remove_small_holes, remove_small_objects

from skimage.segmentation import clear_border, find_boundaries
from skimage.measure import label, regionprops

from pyxvis.processing.images import Edge 


def seg_bimodal(img, p=-0.1):
    """

    :param img: numpy array for a gray-scale input image.
    :param p: float to denote the offset of threshold with p between -1 and 1. A
                positive value is used to dilate the segmentation, the negative
                to erode.
    :return: a tuple with the binary image and the contour of the mask.

    """
    img_d = img.astype('double')
    imax = np.max(img_d.flatten())
    imin = np.min(img_d.flatten())
    img_j = (img_d - imin) / (imax - imin)  # Image normalization [0, 1]

    ni = int(np.fix(img_j.shape[0] / 4))
    nj = int(np.fix(img_j.shape[1] / 4))

    # An heuristic method to ensure that the segmented object will
    # be labeled as 1.
    if np.mean(img_j[0:ni, 0:nj].flatten()) > 0.3:
        img_j = 1 - img_j

    # Transform a double image to unit8
    img_jj = (255 * img_j).astype('uint8')

    # Apply Gaussian filtering to remove noise and compute Otsu thresholding.
    # This step improve the thresholding
    img_jj = gaussian(img_jj, 2.0)
    level = threshold_otsu(img_jj)
    mask = img_jj > (level + p)

    a = remove_small_objects(mask, np.fix(len(mask.flatten()) / 100))
    c = binary_closing(a.astype('double'), disk(7))

    # Why 8?
    r = remove_small_holes(c, 8)
    contours = find_contours(r, 0.1)

    return r, contours


def region_growing(img, seed_point, tolerance=20):
    """
    Args:
        img (ndarray): image
        seed_point (tuple): define the seed point.
        tolerance: a float or int value to define the maximal difference of grayvalues in the region. According to the
                   documentation in scikit-image, if tolerance is provided, adjacent points with values within plus 
                   or minus tolerance from the seed point are filled (inclusive).

    Returns:
        mask (ndarray): output binary image
    """
    mask = flood(img, seed_point, tolerance=tolerance)
    mask = binary_dilation(mask, np.ones((3, 3)))

    return mask


def seg_log_feature(_img, _mask=None, area=None, gray=None, contrast=None, sigma=None):
    """
    
    """
    if _mask.all():
        _mask = np.ones(_img.shape)

    # Unpackage parameters
    amin = area[0]
    amax = area[1]
    gmin = gray[0]
    gmax = gray[1]
    cmin = contrast[0]
    cmax = contrast[1]
    
    se = disk(3)  # Structural element
    re = binary_dilation(_mask, se)

    e = clear_border(find_boundaries(_mask, connectivity=1, mode='inner')).astype(np.uint8)
    edge = Edge('log', 1e-8, sigma)  # Edge detection using log
    edge.fit(_img)

    b = edge.edges.astype(np.uint8)
    b = np.bitwise_and(b, re)
    b = np.bitwise_or(b, e)
    b = clear_border(np.bitwise_not(b))

    f, m = label(b, connectivity=1, return_num=True)  # Connectivity=1 == 4 Neighboors

    d = np.zeros(_img.shape).astype(np.uint8)
    
    for i in range(m):
        r = f == i
        b = np.bitwise_xor(r, binary_dilation(r, np.ones((17, 17)))).astype(np.uint8)
        ir = r == 1
        ib = b == 1
        area = np.sum(ir.flatten())
        gray_r = np.mean(_img[ir].flatten())
        gray_b = np.mean(_img[ib].flatten())
        contrast = gray_r / gray_b

        if (area >= amin) and (area <= amax) and (gray_r >= gmin) and (gray_r <= gmax) and (contrast >= cmin) and (contrast <= cmax):
            d = np.bitwise_or(d, r)

    f, m = label(d, connectivity=1, return_num=True)  # Connectivity=1 == 4 Neighboors

    return f, m


def seg_mser(_img, area=(60, 20000), min_div=0.2, max_var=0.25, delta=5, area_threshold=200):
    """
    Image segmentation using MSER algorithm.

    Args:
        _img (ndarray): input image (uint8)
        area (int tuple): Range of areas to prune (min_area, max_area)
        min_div (float): Trace back to cut off MSER with diversity less than min_diversity
        max_var (float): Prune the area that have simliar size to its children
        delta (int): Controls how the stability is calculated comparing (sizei - sizei-delta) / sizei-delta. [1]
        area_threshold (float): the area threshold to cause re-initialize in color images [1]

    Returns:
       mask (ndarray): Binary image with detected objects
       label_regions (ndarray): Image with labels of the detected objects
       bboxes (ndarray): Bounding boxes for each detection in label_regions (x0, y0, width, height), where (x0, y0) is
                         the uppper-left corner.
    

    [1] Please check the OpenCV documentation for more details.
    """
    
    mser_opts = {
        '_min_area': area[0],
        '_max_area': area[1],
        '_max_variation': max_var,
        '_min_diversity': min_div,
        '_delta': delta,
        '_area_threshold': area_threshold
    }
    
    mser = MSER_create(**mser_opts)
    regions, bbox = mser.detectRegions(_img)
    
    mask = np.zeros(_img.shape)

    for dd, region in enumerate(regions):
        mask[region[:,1], region[:,0]] = 1

    label_regions, m = label(mask, connectivity=1, return_num=True)
    props = regionprops(label_regions)
    bboxes = np.array([[p.bbox[1], p.bbox[0], p.bbox[3] - p.bbox[1], p.bbox[2] - p.bbox[0]] for p in props])
    
    return mask, label_regions, bboxes
