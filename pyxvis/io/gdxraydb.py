"""
This module implements classes and methods to use the GDXray dataset included in the book.
"""

__author__ = 'Christian Pieringer'
__created__ = '2019.05.08'

import os.path as _path
from os import listdir
from os.path import isdir

import scipy.io as sio
from cv2 import imread, IMREAD_GRAYSCALE
from numpy import genfromtxt, loadtxt, array


# Constants
GDXRAY_PATH = _path.join(_path.expanduser('~'), 'GDXray')


class DatasetBase:
    """
    This class is an abstraction of the images sets class included in this module.
    """

    def __init__(self, group_name, group_prefix, dataset_path=None):
        """
        Init class instance.

        Args:
            dataset_path: string to denote the path of the dataset.

        """

        if not dataset_path:
            root_path = GDXRAY_PATH
        else:
            root_path = dataset_path

        self.root_path = root_path
        self.group_name = group_name
        self.group_prefix = group_prefix
        self.dataset_path = _path.join(self.root_path, self.group_name)

    def __repr__(self):
        class_info = "Class Info:\n"
        class_info += f"Group: {self.group_name}\n"
        class_info += f"Prefix: {self.group_prefix}\n"
        class_info += f"Path: {self.dataset_path}\n"
        class_info += f"Root_path: {self.root_path}\n"
        return class_info

    def describe(self):
        """
        Aggregate dataset information, such as, series names, size, number of images, etc.

        Returns: a python dictionary that aggregates the information.
        """
        # Create a dictionary for count items in each series
        table = {'series': 0, 'images': 0, 'size': 0}

        # Read series within a group
        series = [s for s in listdir(self.dataset_path) if isdir(_path.join(self.dataset_path, s))]

        num_images = 0
        size_bytes = 0

        for s in series:
            images = [i for i in listdir(_path.join(self.dataset_path, s)) if i.endswith('.png')]
            num_images += len(images)

            for f in images:
                size_bytes += _path.getsize(_path.join(self.dataset_path, s, f))

        table['series'] = len(series)
        table['images'] = num_images
        table['size'] = size_bytes / (1024.0 ** 2)

        return table

    def get_dir(self, series):
        """
        Returns the path of an image series.

        Args:
            series (int): the series number

        Returns:
            (str) the series path. It raise an exception if the path does not exist.

        """
        set_path = '{}{:04d}'.format(self.group_prefix, series)
        set_path = _path.join(self.dataset_path, set_path)

        # Check if the directory exist. If not, raise a ValueError exception.
        if not _path.exists(set_path):
            raise ValueError('The series does not exist. Please check the parameters or the root path.')

        return set_path

    def load_gt(self, series, k):
        """
        Load ground truth

        Args:
            series: series number
            k: image number

        Returns:

        """
        _filename = _path.join(self.dataset_path, '{:s}{:04d}'.format(self.group_prefix, series))
        _filename = _path.join(_filename, '{:s}{:04d}_{:04d}.txt'.format(self.group_prefix, series, k))
        ground_truth = genfromtxt('my_file.csv', delimiter=',')

        return ground_truth

    def load_image(self, series, k):
        """
        Loads an image from the database.

        Args:
            series (str): defines the set in the images set.
            k (int): defines the image index.

        Returns:
            (numpy array) loaded image
        """
        _filename = _path.join(self.dataset_path, '{0:s}{1:04d}'.format(self.group_prefix, series))
        _filename = _path.join(_filename, '{0:s}{1:04d}_{2:04d}.png'.format(self.group_prefix, series, k))

        try:
            assert _path.exists(_filename)
        except AssertionError:
            raise Exception('File not found')

        img = imread(_filename, IMREAD_GRAYSCALE)

        return img

    def load_data(self, series, data_type=None):
        """
        Loads an image from GDXray database.

        Args:
            series (str): defines the set in the images set.
            data_type (bool): defining the image index.

        Raises:
            AssertionError: file not found

        Returns:
                _data (numpy array): the output image
        """
        _filename = _path.join(self.dataset_path, '{0}{1:04d}'.format(self.group_prefix, series))
        _filename = _path.join(_filename, '{0}.mat'.format(data_type))

        try:
            assert _path.exists(_filename)
        except AssertionError:
            raise Exception('File not found: {}'.format(_filename))

        # Load the LUT from mat file.
        data = sio.loadmat(_filename)

        _data = None

        if not data_type:
            _data = None
        elif data_type == 'DualEnergyLUT':
            _data = data['LUT']
        elif data_type == 'points':
            _data = data
        else:
            _data = data

        return _data


class Baggages(DatasetBase):

    def __init__(self, group_name='Baggages', group_prefix='B', *args, **kwargs):
        super().__init__(group_name, group_prefix, *args, **kwargs)
        # self.group_name = 'Baggages'
        # self.group_prefix = self.group_names[self.group_name]
        # self.dataset_path = _path.join(self.dataset_path, self.group_name)

    def load_image(self, series, k):
        _filename = _path.join(self.dataset_path, '{:s}{:04d}'.format(self.group_prefix, series))
        _filename = _path.join(_filename, '{:s}{:04d}_{:04d}.png'.format(self.group_prefix, series, k))

        try:
            assert _path.exists(_filename)
        except AssertionError:
            raise Exception('File not found: {}'.format(_filename))

        img = imread(_filename, IMREAD_GRAYSCALE)

        return img

    def load_data(self, series, data_type=None):
        _filename = _path.join(self.dataset_path, 'B{:04d}'.format(series))
        _filename = _path.join(_filename, '{}.mat'.format(data_type))

        try:
            assert _path.exists(_filename)
        except AssertionError:
            raise Exception('File not found')

        # Load the LUT from mat file.
        data = sio.loadmat(_filename)

        _data = None

        if not data_type:
            _data = None
        elif data_type == 'DualEnergyLUT':
            _data = data['LUT']
        else:
            _data = data

        return _data


class Settings(DatasetBase):
    """
    This class implements an instance of the Settings image set.
    """

    def __init__(self, group_name='Settings', group_prefix='S', *args, **kwargs):
        super().__init__(group_name, group_prefix, *args, **kwargs)
        # self.dataset_path = _path.join(self.dataset_path, self.group_name)
        #self.group_prefix = 'S'


class Welds(DatasetBase):
    """
    This class implements an instance of the Settings image set.
    """

    def __init__(self, group_name='Welds', group_prefix='W', *args, **kwargs):
        super().__init__(group_name, group_prefix, *args, **kwargs)
        #self.dataset_path = _path.join(self.dataset_path, 'Welds')
        # self.group_prefix = 'W'


class Nature(DatasetBase):
    """
    This class implements an instance of the Settings image set.
    """

    def __init__(self, group_name='Nature', group_prefix='N', *args, **kwargs):
        super().__init__(group_name, group_prefix, *args, **kwargs)
        # self.group_prefix = 'N'
        # self.group_name = self.group_names[self.group_prefix]
        #self.dataset_path = _path.join(self.dataset_path, self.group_name)


class Castings(DatasetBase):
    """
    This class implements an instance of the Settings image set.
    """

    def __init__(self, group_name='Castings', group_prefix='C', *args, **kwargs):
        super().__init__(group_name, group_prefix, *args, **kwargs)
        #self.dataset_path = _path.join(self.dataset_path, 'Castings')
        # self.group_prefix = 'C'

    def load_data(self, series, data_type=None):
        # In this case the method loads the file defined as argument.
        _filename = _path.join(self.dataset_path, '{0}{1:04d}'.format(self.group_prefix, series))
        _filename = _path.join(_filename, '{0}'.format(data_type))

        data = loadtxt(_filename)

        return data


def load_image_set(set_name, *args, **kwargs):
    image_sets = {
        'baggages': Baggages,
        'settings': Settings,
        'nature': Nature,
        'welds': Welds,
        'castings': Castings
    }

    return image_sets[set_name.lower()](*args, **kwargs)


class ImageSet(DatasetBase):
    """
    This class create a generic image set container.
    """

    def __init__(self, *args, **kwargs):
        """

        :param collection_dir:
        """
        super().__init__(*args, **kwargs)
        # print(self.dataset_path)

    def load_image(self, serie_name, k):
        """

        :param serie_name: serie name (str)
        :param k: image index (int)
        :return: ndarray with the image
        """
        _filename = _path.join(self.dataset_path, serie_name)
        _filename = _path.join(_filename, '{:s}_{:04d}.png'.format(serie_name, k))

        try:
            assert _path.exists(_filename)
        except AssertionError:
            raise Exception('File not found')

        img = imread(_filename, IMREAD_GRAYSCALE)

        return img


def xgdx_stats(root_dir=None):
    """
    Compute a complete set of statistics on the dataset, such as, the total number of groups, the total number of
    series per group, and the size of each group in MB.

    Args:
        root_dir:

    Raises:
        ValueError: dataset directory not found

    Returns:
        table (dict): a summary of dataset statistics.
    """

    if not root_dir:
        root_dir = GDXRAT_PATH

    # Check if the directory exist. If not, raise a ValueError exception.
    if not _path.exists(root_dir):
        raise ValueError('Directory not found. Please check the root path.')

    # Create a dictionary for count items in each series
    table = {
        g:
            {'series': 0,
             'images': 0,
             'size': 0
             }
        for g in listdir(root_dir)
    }

    for g in table:
        series_dir = _path.join(root_dir, g)
        series = [d for d in listdir(series_dir) if
                  _path.isdir(_path.join(series_dir, d))]

        num_images = 0
        size_bytes = 0

        for s in series:
            images = [i for i in listdir(_path.join(root_dir, g, s)) if
                      i.endswith('png')]
            num_images += len(images)

            for f in images:
                size_bytes += _path.getsize(_path.join(root_dir, g, s, f))

        table[g]['series'] = len(series)
        table[g]['images'] = num_images
        table[g]['size'] = size_bytes / (1024.0 ** 2)

    return table


def load_gt(img_set, series, k, gt_filename='ground_truth.txt'):
    """
    Load ground truth from a series

    Args:
        img_set (object): XGDX image set instance
        series: series number
        k: image number
        gt_filename:

    Returns:

    """
    _filename = _path.join(img_set.dataset_path, '{:s}{:04d}'.format(img_set.group_prefix, series))
    _filename = _path.join(_filename, gt_filename)

    # Check if the series contains ground truth
    if not _path.isfile(_filename):
        raise IOError('Ground truth file not found')

    _ground_truth = genfromtxt(_filename)
    _ground_truth = _ground_truth[_ground_truth[:, 0] == k][:, 1:]

    return _ground_truth


if __name__ == '__main__':

    from pyxvis.io import gdxraydb

    # Testing the class instantiation
    set_names = ['Castings', 'Welds', 'Nature', 'Settings', 'Baggages']

    for set_name in set_names:
        image_set = load_image_set(set_name)
        print(image_set)
        print(image_set.describe())

    # Testing loading ground truth
    image_set = gdxraydb.Baggages()
    img = image_set.load_image(2, 4)
