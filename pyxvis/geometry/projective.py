"""
This module includes all the functionalities for geometric analysis in
X-Ray images.

"""

__author__ = 'Christian Pieringer'
__created__ = '2019.08.22'

import numpy as np


def rotation_matrix_2d(theta):
    """
    The 2D rotation matrix R given by a rotation angle theta in radians.

    :param theta: rotation angle in radians (float)
    :returns _r_mat: the rotation matrix (ndarray)
    """

    _r_mat = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ]
    )

    return _r_mat


def rotation_matrix_3d(wx, wy, wz):
    """
    The method returns the 3D rotation matrix given by a rotation angle theta in radians.

    :param wx: angular rotation in radians for x-axis
    :param wy: angular rotation in radians for y-axis
    :param wz: angular rotation in radians for z-axis
    :returns _r_mat: the rotation matrix for 3D spaces (ndarray)
    """

    _r_mat = np.array([
        [np.cos(wy) * np.cos(wz), -np.cos(wy) * np.sin(wz), np.sin(wy)],
        [np.sin(wx) * np.sin(wy) * np.cos(wz) + np.cos(wx) * np.sin(wz),
         -np.sin(wx) * np.sin(wy) * np.sin(wz) + np.cos(wx) * np.cos(wz),
         -np.sin(wx) * np.cos(wy)],
        [-np.cos(wx) * np.sin(wy) * np.cos(wz) + np.sin(wx) * np.sin(wz),
         np.cos(wx) * np.sin(wy) * np.sin(wz) + np.sin(wx) * np.cos(wz),
         np.cos(wx) * np.cos(wy)]
    ])

    return _r_mat


def get_matrix_p(f):
    """
    Build the 3x4 perspective projection matrix depending on focal distance f.

    Args:
        f: the focal distance (float)

    Returns: the perspective projection matrix p (ndarray)

    """
    p = np.array([
        [f, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, 1, 0]
    ])
    return p


def hyperproj(m3d, pa, hp):
    """
    Build the 3D->2D transformation using an hyperbolic model (Mery D. and Filbert, D. 2002).

    Mery, D.; Filbert, D. (2002): Automated Flaw Detection in Aluminum Castings Based on The Tracking of Potential
        Defects in a Radioscopic Image Sequence. IEEE Transactions on Robotics and Automation, 18(6):890-901.

    Args:
        m3d:
        pa:
        hp:

    Returns:

    """
    alpha = pa[0]
    u0 = pa[1]
    v0 = pa[2]
    f = pa[3]
    a = pa[4]
    b = pa[5]
    kx = pa[6]
    ky = pa[7]

    mat_a = np.array([
        [kx * np.cos(alpha), ky * np.sin(alpha), u0],
        [-kx * np.sin(alpha), ky * np.cos(alpha), v0],
        [0, 0, 1]]
    )

    mat_b = np.array(
        [
            [f, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, 1, 0]
        ]
    )

    mat_p = np.dot(mat_b, hp)

    m = np.dot(mat_p, m3d)
    m = m / m[-1]

    xp = m.item(0)
    yp = m.item(1)

    # TODO: check this equation
    c = 1 / np.sqrt((1 - ((xp / a) ** 2)) - ((yp / b) ** 2))

    mpp = np.array([c * xp, c * yp, 1])

    w = np.dot(mat_a, mpp[:, np.newaxis])

    return w

