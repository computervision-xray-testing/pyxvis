"""
Backend module for GUI XGDX_BROWSE
"""

import os

import matplotlib.pylab as plt

from pyxvis.io.gdxraydb import load_image_set, load_gt
from pyxvis.io.visualization import xvis_sincolormap, plot_bboxes

from numpy import append, newaxis

# Check this way to create the backend
# https://stackoverflow.com/questions/55296264/front-end-and-back-end-separation-in-pyqt5-with-qml

# Color definitions
cmaps = {
    'Gray': 'gray',
    'Jet': 'jet',
    'Hsv': 'hsv',
    'Viridis': 'viridis',
    'Hot': 'hot',
    'Rainbow': 'rainbow',
    'Sinmap': xvis_sincolormap(),
    'Autumn': 'autumn',
    'Spring': 'spring',
}


def load_dataset(collection_name):
    """
    Load a image set instantiating a XGDX_ray group class.
    Args:
        collection_name (str): Group name

    Returns:
        image_set (object): DataBase class instance
    """
    image_set = load_image_set(collection_name)
    return image_set


def get_series_id(series_name):
    series_id = int(series_name.strip('0')[1:].strip('0'))
    return series_id


def load_image(image_set, series_name, image_id):
    """

    :param serie_name: serie name (str)
    :param k: image index (int)
    :return: ndarray with the image
    """
    series_id = get_series_id(series_name)

    return image_set.load_image(series_id, image_id)


def get_dir_list(input_path):
    """
    This method returns a list of available image collections.

    :param input_path:
    :return:
    """
    print(input_path)
    series = [s for s in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, s))]

    # return os.listdir(input_path)
    return series


def check_last_image(curr_idx, last):
    return curr_idx < last


def check_first_image(curr_idx):
    return curr_idx > 1


def get_series_length(group_path, series_name):
    path_name = os.path.join(group_path, series_name)
    images = [i for i in os.listdir(path_name) if i.endswith('.png')]
    return len(images)


def compose_image_message(self):
    _text_bottom = f"[ Group {self.image_set.group_name} with {self.series_num} series ]"
    _text_bottom += "  "
    _text_bottom += f"[ Series {self.series_name} with {self.series_len} images ]"
    _filename = compose_filename(self.series_name, self.series_current_idx)
    messages = {
        'filename': _filename,
        'text_bottom': _text_bottom
    }

    return messages


def show_image(img, messages=None, fig=None, ax=None, cmap='Gray'):

    if not ax or not fig:
        w, h = plt.figaspect(img.shape[0] / img.shape[1])
        fig = plt.figure(figsize=(w, h))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    bbox_opts = dict(ec='k', fc='k')
    ax.clear()
    img_axes = ax.imshow(img, cmap=cmaps[cmap])
    ax.axis('off')
    ax.text(0.03, 0.97, messages['filename'], fontsize=8, color='w', ha='left', va='top', transform=ax.transAxes,
            bbox=bbox_opts)
    ax.text(0.03, 0.03, messages['text_bottom'], fontsize=8, color='w', ha='left', va='bottom',
            transform=ax.transAxes, bbox=bbox_opts)
    fig.canvas.draw()
    plt.autoscale(tight=True)
    fig.show()

    return fig, ax, img_axes


def compose_filename(series, k):
    """
    Concatenate string to build the image name.

    Args:
        group_prefix:
        series:
        k:

    Returns:
        A string with the filename
    """
    return '{:s}_{:04d}.png'.format(series, k)


def show_ground_truth(image_set, series_name, k, fig, ax):
    """
    Plot bounding boxes on the current image

    Args:
        image_set:
        series_name:
        k:
        fig:
        ax:

    Returns:

    """
    series_id = get_series_id(series_name)

    # Load and preprocessing ground truth. Data laid on the next format:
    # [image_number, Y0, Y1, X0, X1]
    gt = load_gt(image_set, series_id, k)  # load_gt only returns the boubding boxes (bb) for image k: (Nbb, 4)
    new_gt = append(gt[:, [0, 2]], (gt[:, 1] - gt[:, 0])[:, newaxis], axis=1)  # Transform [X0, Y0, Width, Height]
    new_gt = append(new_gt, (gt[:, 3] - gt[:, 2])[:, newaxis], axis=1)

    # Plot bounding_boxes
    ax = plot_bboxes(new_gt, color='lawngreen', linewidth=1.5, ax=ax)
    fig.canvas.draw()

    return fig, ax


def close_figure(fig):
    plt.close(fig)
