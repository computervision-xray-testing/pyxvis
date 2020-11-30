"""
"""
__author__ = 'Christian Pieringer'
__created__ = '2019.05.08'

import matplotlib.pyplot as plt

import numpy as np
from cv2 import (resize, INTER_AREA)
from matplotlib.colors import ListedColormap
from matplotlib import animation
from skimage.color import label2rgb
from skimage.measure import label
from skimage.morphology import binary_dilation
from skimage.segmentation import clear_border
from mpl_toolkits.mplot3d import Axes3D
from skimage.feature import plot_matches as _plot_matches


def xvis_sincolormap(w=None, k=None, a=None, th=None):
    """
    This method build the sinmap colormap defined in Appendix B.

    :param w:
    :param k:
    :param a:
    :param th:
    :return:
    """
    # Check input parameters
    if not w:
        w = np.array(
            [1.25 * np.pi / 255, 1.25 * np.pi / 255, 1.25 * np.pi / 255]).T

    if not k:
        k = np.ones((3, 1))

    if not a:
        a = np.zeros((3, 1))

    if not th:
        th = [np.pi / 3, np.pi / 2, -4 * np.pi / 3]

    # Compute the colormap according to Equation (1.5)
    cmap = np.zeros((256, 3))
    x = np.array(range(1, 257))

    cmap[:, 0] = abs(a[0] + k[0] * (np.cos(w[0] * x + th[0])))
    cmap[:, 1] = abs(a[1] + k[1] * (np.cos(w[1] * x + th[1])))
    cmap[:, 2] = abs(a[2] + k[2] * (np.cos(w[2] * x + th[2])))

    # cmap = np.concatenate((cmap, np.ones((256, 1))), axis=1)
    cmap = ListedColormap(cmap)

    return cmap


def show_xray_image(xray_images, color_map='gray'):
    """

    :param xray_images: Numpy array or list of Numpy array with input images
    :param color_map: string with the required colormap. Default is gray.
    :return: None.

    """
    if isinstance(xray_images, np.ndarray):
        fig = plt.figure(figsize=(6, 8))
        ax = fig.add_subplot(1, 1, 1)
        plt.imshow(xray_images, cmap=color_map)
        plt.axis('off')

    elif isinstance(xray_images, list):
        fig = plt.figure(figsize=(12, 8))
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        for i, img in enumerate(xray_images):
            ax = fig.add_subplot(1, 2, i + 1)
            ax = plt.imshow(img, cmap=color_map)
            plt.axis('off')
    plt.show()


def show_color_array(img, scale=None):
    """
    This function display an array of croped images using
    various colormaps defined previously.

    :params img: the input image in numpy format.

    :returns: None

    """

    # Color definitions
    pyxvix_cmaps = {
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

    # Generate a copy of the original image.
    img_resized = img.copy()

    # Keep aspect ratio
    if scale:
        width = int(img_resized.shape[1] * scale / 100)
        height = int(img_resized.shape[0] * scale / 100)
        new_size = (width, height)

        # Resize image
        img_resized = resize(img_resized, new_size, interpolation=INTER_AREA)

    ncols, nrows = 3, 3
    fig, fig_axes = plt.subplots(
        ncols=ncols,
        nrows=nrows,
        constrained_layout=False,
        figsize=(16, 8))

    i = 0
    cm = list(pyxvix_cmaps)
    for r in range(nrows):
        for c in range(ncols):
            fig_axes[r, c].imshow(img_resized, cmap=pyxvix_cmaps[cm[i]])
            fig_axes[r, c].axis('off')
            fig_axes[r, c].set_title(cm[i])
            i += 1
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    fig.suptitle("Colormap Visualization", fontsize=16)
    plt.show()


def dynamic_colormap(img, cmap, n=128, channels=None):
    """
    Performs the dynamic pseudocolor representations of image 'img'. This will be displayed using different color maps
    as a movie. The first color map is the original ('map'), the next maps will be constructed by circularly shifting
    the the rows of the map in one direction. When a whole shift is done, the direction is changed. This process is
    performed 3 times. Parameter 'channels' defines which columns of the map are shifted. Default is 1:3 (all three
    channels).

    This program can be used for manual inspection. The objects of interest can be more distinguishable when displaying
    the X-ray image as video in different color maps.

    Args:
        img: a numpy array with image
        cmap: a string to define the used colormap.
        n: a integer to define the number of levels in the colormap, default=128.
        channels: a list to define the channels to be changed, default=None.

    Returns: None

    """
    def animate(i, _channels):
        cmap = c['cmap']

        if i >= n:
            cmap = np.vstack([cmap[1:, _channels], cmap[0, _channels][np.newaxis, :]])
        else:
            cmap = np.vstack([cmap[-1, _channels][np.newaxis, :], cmap[0:-1, _channels]])

        new_colormap = ListedColormap(cmap, name='tmp')
        im.set_cmap(new_colormap)
        c['cmap'] = cmap

        return im

    if not channels:
        channels = range(3)

    _cmap = plt.get_cmap(cmap, n)
    _cmap = _cmap(np.linspace(0, 1, n))
    c = {'cmap': _cmap}

    k_img = np.uint8(img.astype(float) / 4)

    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(k_img, cmap=cmap)
    ax.axis('off')

    anim = animation.FuncAnimation(fig, animate, 2*n, fargs=(channels,), interval=1)
    plt.show()


def show_image_as_surface(img, cmap='viridis', elev=-120, azim=40, fsize=None, colorbar=False):
    """
    This function display a mesh plot of an image. This is not a 3D visualization.

    Args:
        img:
        cmap:
        elev:
        azim:

    Returns:
        None
    """
    # Create the x and y coordinate arrays
    xx, yy = np.mgrid[0:img.shape[0], 0:img.shape[1]]

    if not fsize:
        fsize = (8, 8)

    # create the figure
    fig = plt.figure(figsize=fsize)
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(xx, yy, img,
                           rstride=1,
                           cstride=1,
                           cmap=cmap,
                           linewidth=0)
    ax.view_init(elev, azim)

    if colorbar:
        fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.01)
    ax.axis('off')
    plt.tight_layout()
    plt.show()


def show_series(image_set, series, img_idx, n=4, scale=0.5, _figsize=(18, 8)):
    """

    Args:
        image_set:
        series:
        img_idx:
        n:
        scale:
        _figsize:

    Returns:

    """
    # Load images
    img_list = []

    for i in img_idx:
        img = image_set.load_image(series, i)
        img = resize(img, None, fx=scale, fy=scale, interpolation=INTER_AREA)
        img_list.append(img.copy())

    # Arrange the images creating a new array with a size large enough
    # to contain all the images.
    n_rows = round((len(img_list) / n) + 0.5)
    img_height, img_width = img_list[0].shape
    total_height = n_rows * img_height
    max_width = n * img_width

    final_image = np.zeros((total_height, max_width), dtype=np.uint8)

    # Keep track of where your current image was last placed in the y coordinate
    current_y = 0
    current_x = 0

    for i, image in enumerate(img_list):
        # Add an image to the final array and increment the y coordinate
        h = image.shape[0] + current_y
        w = image.shape[1] + current_x

        final_image[current_y:h, current_x:w] = image.copy()

        current_x += image.shape[1]
        if (i + 1) % n == 0:
            current_y += image.shape[0]
            current_x = 0

    # TODO: add image number on each image
    fig, ax = plt.subplots(1, 1, figsize=_figsize)
    ax.imshow(final_image, cmap='gray')
    ax.axis('off')
    ax.set_title('Series {:s}{:04}'.format(image_set.group_prefix, series))
    plt.show()


def binview(img, mask, color='r', dilate_pixels=1):
    """
    Displays a gray or color image 'img' overlaid by color pixels determined a by binary image 'mask'. It is useful to
    display the edges of an image.

    Args:
        img: gray scale image (X-ray)
        mask: binary image that works as mask
        color: string to define pixel color.
                'r': red (default)
                'g': green
                'b': blue
                'y': yellow
                'c': cyan
                'm': magenta
                'k': black
                'w': white

        dilate_pixels (int): Number of pixels used for dilate the mask.

    Returns:
        img_color (ndarray): output image with a mask overlaid.
    """

    # Defines colors
    # colors = {
    #     'r': np.array([255, 0, 0]),
    #     'g': np.array([0, 255, 0]),
    #     'b': np.array([0, 0, 255]),
    #     'y': np.array([255, 255, 0]),
    #     'c': np.array([0, 255, 255]),
    #     'm': np.array([255, 0, 255]),
    #     'k': np.array([0, 0, 0]),
    #     'w': np.array([255, 255, 255])
    # }
    #
    colors = {
        'r': np.array([1, 0, 0]),
        'g': np.array([0, 1, 0]),
        'b': np.array([0, 0, 1]),
        'y': np.array([1, 1, 0]),
        'c': np.array([0, 1, 1]),
        'm': np.array([1, 0, 1]),
        'k': np.array([0, 0, 0]),
        'w': np.array([1, 1, 1])
    }
    # Create a RGB image from grayscale image.
    img_color = np.dstack((img, img, img))

    # Ensure do not modify the original color image and the mask
    img_color = img_color.copy()

    mask_ = mask.copy()
    # mask_ = dilate(mask_, np.ones((g, g), np.uint8))
    mask_ = binary_dilation(mask_, np.ones((dilate_pixels, dilate_pixels)))

    # Now black-out the area of the mask
    # img_fg = bitwise_and(img, img, mask=mask_)

    # Defines the pixel color used for the mask in the figure.
    cc = colors[color]
    #
    # for i in range(3):
    #     img_color[:, :, i] = cc[i] * img_fg

    # remove artifacts connected to image border
    cleared = clear_border(mask_)
    if np.all(cleared):
        mask_ = cleared

    # label image regions
    label_image = label(mask_)
    img_color = label2rgb(label_image, image=img_color, colors=[cc], bg_label=0)

    return img_color  # add(img_color, img_color)


def h2nh(m):
    """
    Normalize homogeneous coordinates dividing by element of a vector.

    Args:
        m: vector

    Returns:
        The normalized vector (ndarray)

    """

    return (m / m[-1])


def project_edges_on_chessboard(img, mat_p, square_size, nx=10):
    """
    Projects a the real position of the chessboard edges on the calibration pattern.

    Args:
        img (ndarray): the input image
        mat_p (ndarray): projection matrix
        square_size:
        nx:

    Returns:

    """

    x_max = 10
    y_max = 10

    # Generate a meshgrid with real locations edges on the chessboard
    mat_x = np.arange(nx) * square_size
    mat_y = mat_x.copy()
    mat_x, mat_y = np.meshgrid(mat_x, mat_y)

    # Stack all the edges as a tensor where each cell contains the coordinates
    # of the edge in the real world (chess board).
    edges = np.empty(mat_x.shape + (4,))
    edges[:, :, 0] = mat_x
    edges[:, :, 1] = mat_y
    edges[:, :, 2] = np.zeros(mat_x.shape)
    edges[:, :, 3] = np.ones(mat_y.shape)

    # Using Einstein notation to compute and reduce efficiently the matrices.
    me = np.einsum('ijk, mk->ijm', edges, mat_p).reshape(-1, 3).T
    me = h2nh(me)

    # Plot the results
    fontsize = 14
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img, cmap='gray')
    ax.scatter(me[0, :], me[1, :], marker='o', edgecolor='orange', facecolor='y')

    # Plot the origin of coordinate in the chessboard system
    orig = np.dot(mat_p, np.array([[0], [0], [0], [1]]))
    orig = h2nh(orig)
    ax.plot(orig[0], orig[1], 'ro')
    ax.text(orig[0], orig[1] - 20, 'O', color='w', fontsize=fontsize, weight='bold')

    orig = np.dot(mat_p, np.array([[0], [250], [0], [1]]))
    orig = h2nh(orig)
    ax.plot(orig[0], orig[1], 'bo')
    ax.text(orig[0], orig[1] - 20, 'Y', color='w', fontsize=fontsize, weight='bold')

    orig = np.dot(mat_p, np.array([[250], [0], [0], [1]]))
    orig = h2nh(orig)
    ax.plot(orig[0], orig[1], 'go')
    ax.text(orig[0], orig[1] - 20, 'X', color='w', fontsize=fontsize, weight='bold')

    ax.axis('off')
    plt.show()


def multivariate_gaussian(pos, mu, sigma):
    """
    Return the multivariate Gaussian distribution on array pos.

    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.

    [1] https://scipython.com/blog/visualizing-the-bivariate-gaussian-distribution/

    Args:
        pos:
        mu:
        sigma:

    Returns:

    """
    n = mu.shape[0]

    sigma_det = np.linalg.det(sigma)
    sigma_inv = np.linalg.inv(sigma)

    den = np.sqrt(((2 * np.pi) ** n) * sigma_det)

    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k, kl, ...l->...', pos - mu, sigma_inv, pos - mu)

    return np.exp(-fac / 2) / den


def gaussian_superimposition(img, mat_p, square_size, n_points=30):
    """
    This method projects a Gaussian surface on the calibration pattern.
    """

    x_max = 10
    y_max = 10

    # Generate a meshgrid with real locations edges on the chessboard
    mat_x = np.linspace(0, (x_max - 1) * square_size, n_points)
    mat_y = mat_x.copy()
    mat_x, mat_y = np.meshgrid(mat_x, mat_y)

    # Stack all the edges as a tensor where each cell contains the coordinates
    # of the edge in the real world (chess board).
    edges = np.empty(mat_x.shape + (4,))
    edges[:, :, 0] = mat_x
    edges[:, :, 1] = mat_y
    edges[:, :, 2] = np.zeros(mat_x.shape)
    edges[:, :, 3] = np.ones(mat_y.shape)

    # Mean vector and covariance matrix
    mu = np.array([1125, 1125])
    sigma = np.array([[150000, 0.0], [0.0, 150000]])

    # The distribution on the variables X, Y packed into pos.
    z = multivariate_gaussian(edges[:, :, :2], mu, sigma)
    z = (-1000 * z) / np.max(np.max(z))

    # Add Z coordinates to the edges tensor
    edges[:, :, 2] = z

    me = np.einsum('ijk, mk->ijm', edges, mat_p)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img, cmap='gray')

    # Plot lines on the x-axis first
    for ii in range(me.shape[1]):
        l = me[:, ii]
        l = l / l[:, -1][:, np.newaxis]
        ax.plot(l[:, 0], l[:, 1], '-r')

    # And, finally lines on the y-axis
    for ii in range(me.shape[0]):
        l = me[ii, :]
        l = l / l[:, -1][:, np.newaxis]
        ax.plot(l[:, 0], l[:, 1], '-r')

    ax.axis('off')

    # Plot origin of coordinates
    orig = np.dot(mat_p, np.array([[0], [0], [0], [1]]))
    orig = h2nh(orig)
    ax.plot(orig[0], orig[1], 'ro')
    ax.text(orig[0], orig[1] - 20, 'O', fontsize=14, color='y', weight='bold')

    orig = np.dot(mat_p, np.array([[0], [250], [0], [1]]))
    orig = h2nh(orig)
    ax.plot(orig[0], orig[1], 'bo')
    ax.text(orig[0], orig[1] - 20, 'Y', fontsize=14, color='y', weight='bold')

    orig = np.dot(mat_p, np.array([[250], [0], [0], [1]]))
    orig = h2nh(orig)
    ax.plot(orig[0], orig[1], 'go')
    ax.text(orig[0], orig[1] - 20, 'X', fontsize=14, color='y', weight='bold')

    plt.show()


def plot_bboxes(bounding_boxes, color='lawngreen', linewidth=1.5, ax=None):
    """
    Plot a set of bounding boxes.

    Args:
        bounding_boxes (ndarray): array of bounding boxes (x0, y0, width, height), where x0, y0 is the upper-left corner.
        color (str): Edge color of the bounding box. Use matplotlib color names.
        linewidth (float): Bounding box linewidth.
        ax (object): Figure axes. If None, create a new figure and axes.

    Returns:
        ax (object): Figure axes.
    """
    if not ax:
        fig, ax = plt.subplots(1, 1)

    for b, bbox in enumerate(bounding_boxes):
        rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill=False, edgecolor=color, linewidth=linewidth)
        ax.add_patch(rect)

    return ax


def plot_matches_decorator(_func):
    def wrapper_func(ax, I1, I2, kp1, kp2, matches, **args):
        _func(ax, I1, I2, kp1[:, [1, 0]], kp2[:, [1, 0]], matches, **args)
    return wrapper_func


plot_matches = plot_matches_decorator(_plot_matches)
