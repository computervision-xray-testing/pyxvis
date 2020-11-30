"""
This module includes all the functionalities for epipolar geometry.

"""

__author__ = 'Christian Pieringer'
__created__ = '2019.10.08'

import numpy as np


def line2points(x, l):
    """
    Transforms a point into a line.

    Args:
        x: array with a point
        l: array with a line in homogeneous coordinates
    
    Returns:
        an array with the line
    """
    y = (-1.0 / l[1]) * (l[2] - l[0] * x)

    return np.array([x, y])


def plot_epipolar_line(mat_f, m, ax, line_color='r'):
    """
    This method plot an epipolar line into an open figure.

    Args:
        mat_f: numpy array with the fundamental matrix
        m: numpy array with a image point (2D)
        ax: a matplotlib figure axe
        line_color: an string indicating the line color
    
    Returns:
        ax: the figure axe

    """
    if m.shape[-1] == 2:
        m = np.hstack([m, np.ones((m.shape[0], 1))])

    ell = np.dot(mat_f, m)
    ell = ell / ell[-1]
    ax_lim = ax.axis()
    x = np.asarray(ax_lim[0:2])
    a = ell[0]
    b = ell[1]
    c = ell[2]

    y = -(c + a * x) / b

    ax.plot(x, y, line_color)
    ax.axis('off')

    return ax


def estimate_trifocal_tensor(mat_a, mat_b, mat_c):
    """

    :param mat_a:
    :param mat_b:
    :param mat_c:
    :return:
    """

    mat_h = np.vstack([mat_a, np.random.rand(1, 4)])
    while np.abs(np.linalg.det(mat_h)) < 0.001:
        mat_h = np.vstack([mat_a, np.random.rand(1, 4)])
        if np.linalg.det(mat_h) < 0.001:
            print('Warning: determinant of H in trifocal estimation is zero')

    a = np.dot(mat_a, np.linalg.pinv(mat_h))
    b = np.dot(mat_b, np.linalg.pinv(mat_h))  # b = B * inv(H)
    c = np.dot(mat_c, np.linalg.pinv(mat_h))  # c = C * inv(H)

    tt = np.zeros((3, 3, 3))

    for i in range(3):
        for j in range(3):
            for k in range(3):
                tt[i, j, k] = b[j, i] * c[k, 3] - b[j, 3] * c[k, i]

    #     for i in range(3):
    #         t1 = np.dot(b[:, i].reshape(-1, 1), c[:, -1].reshape(1, -1))
    #         t2 = np.dot(b[:, -1].reshape(-1, 1), c[:, i].reshape(1, -1))
    #         tt[i, :, :] = t1 - t2
    return tt


def estimate_fundamental_matrix(mat_a, mat_b, method='pseudo'):
    """
    This method computes the 3x3 fundamental matrix from 3x4 projection matrices A and B according methods defined in
    method parameter. We refer the reader to [1] for more details about this implementation.

    Args:
        mat_a: numpy array with projection matrix of view 1
        mat_b: numpy array with projection matrix of view 2
        method: a string defining computation method. If no method is given,
            'pseudo' will be assumed as default.
            'tensor' : uses bifocal tensors with canonic matrix
            'tensor0': uses bifocal tensors without canonic matrix
            'pseudo' : uses pseudoinverse matrix of A (default)

    Returns:
        mat_f: the 3x3 fundamental matrix.

    [1] R. Hartley and A. Zisserman. Multiple View Geometry in Computer Vision.
    Cambridge University Press, 2000.
    """

    if method.lower() == 'tensor':
        mat_h = np.vstack([mat_a, np.random.rand(4)])

        while np.abs(np.linalg.det(mat_h)) < 0.001:
            mat_h = np.vstack([mat_a, np.random.rand(4)])

        mat_bs = np.dot(mat_b, np.linalg.inv(mat_h))
        b = np.vstack([mat_bs, mat_bs])

        mat_f = np.zeros((3, 3))

        for i in range(3):
            for j in range(3):
                mat_f[i, j] = b[i + 1, j] * b[i + 2, 3] - b[i + 2, j] * b[
                    i + 1, 3]

    elif method.lower() == 'tensor0':
        # TODO: Verificar con Domingo
        mat_f = np.zeros((3, 3))

        for i in range(3):
            sb = mat_b.copy()
            sb = np.delete(sb, i, 0)

            for j in range(3):
                sa = mat_a.copy()
                sa = np.delete(sa, j, 0)

                mat_f[i][j] = (-1) ** (i + j) * np.linalg.det(
                    np.vstack([sa, sb]))

    else:
        # Find mat_c1 as the null space of the projection matrix using SVD factorization of mat_a. The null vector is 
        # the last row in the matrix V.T . See numpy documentation for more details.

        _, _, mat_v = np.linalg.svd(mat_a)
        mat_c1 = mat_v[-1, :].reshape(-1, 1)
        mat_u = skew_symmetric(mat_b, mat_c1)

        # Compute the fundamental matrix
        mat_f = np.asmatrix(mat_u) * np.asmatrix(mat_b) * np.asmatrix(
            np.linalg.pinv(mat_a))
        mat_f = np.array(mat_f)

    mat_f = np.array(mat_f / np.linalg.norm(mat_f)[0])  # Normalize as F / norm(F)

    return mat_f


def skew_symmetric(mat, null_space):
    """
    Compute the skew-symmetric matrix.

    :param mat: numpy array with a projection matrix (3x4)
    :param null_space: an array with the null space
    :return: an array with the skew-symmetric matrix.

    """
    # epipole in second view
    e2 = np.dot(mat, null_space)

    # The vector e2 as Skew-symmetric matrix
    mat_u = np.array(
        [
            [0, -e2[2], e2[1]],
            [e2[2], 0, -e2[0]],
            [-e2[1], e2[0], 0]
        ]
    )

    return mat_u


def reproject_trifocal(m1, m2, tensor, method=2):
    """

    Args:
        m1: 2D point in image 1
        m2: 2D point in image 2
        tensor: 
        method:
    
    Returns:
        m3: a 2D points in image 3
    """

    # Assert that m1 and m2 are homogeneous.
    m1 = m1 / m1[-1, :]
    m2 = m2 / m2[-1, :]

    n = m1.shape[-1]
    m3 = np.zeros((3, n))

    for k in range(n):
        temp = tensor[:, 0, :] - m2[0, k] * tensor[:, 2, :]
        m3[:, k] = (temp.T @ m1[:, k][:, np.newaxis]).ravel()

    #         temp = np.array([
    #             tensor[:, 0, 0] - m2[0, k] * tensor[:, 2, 0],
    #             tensor[:, 0, 1] - m2[0, k] * tensor[:, 2, 1],
    #             tensor[:, 0, 2] - m2[0, k] * tensor[:, 2, 2]
    #         ]) @ m1[:, k][:, np.newaxis]

    #         m3[:, k] = temp.ravel()

    m3 = m3 / m3[-1, :]

    if method == 2:
        m32 = np.zeros((3, n))
        for k in range(n):
            temp = tensor[:, 1, :] - m2[1, k] * tensor[:, 2, :]
            m32[:, k] = (temp.T @ m1[:, k][:, np.newaxis]).ravel()

        #             temp = np.array(
        #                 [
        #                     tensor[:, 1, 0] - m2[1, k] * tensor[:, 2, 0],
        #                     tensor[:, 1, 1] - m2[1, k] * tensor[:, 2, 1],
        #                     tensor[:, 1, 2] - m2[1, k] * tensor[:, 2, 2]
        #                 ]) @ m1[:, k][:, np.newaxis]#reshape(-1, 1)).ravel()
        #             m32[:, k] = temp.ravel()
        m32 = m32 / m32[-1, :]
        m3 = (m3 + m32) / 2

    return m3


def recon_3dn(m, mat_p, method=None):
    """
    This method computes the reconstruction of a M point from two dimensional
    points in images.

    Args:
        m:
        mat_p:
        method:
        ls: the function computes the reconstruction using least-square method
    
    Returns:
        mat_mr: Reprojected points
        ms: Estimated points in 2D
        err: reprojection error in pixels
    """

    # n = len(m)  # number of images
    # mat_q = []
    # mat_r = []
    #
    # for xy, p in zip(m, mat_p):
    #     mat_q.append([p[2, 0] * xy[0] - p[0, 0], p[2, 1] * xy[0] - p[0, 1],
    #                   p[2, 2] * xy[0] - p[0, 2]])
    #     mat_q.append([p[2, 0] * xy[1] - p[1, 0], p[2, 1] * xy[1] - p[1, 1],
    #                   p[2, 2] * xy[1] - p[1, 2]])
    #     mat_r.append([p[0, 3] - p[2, 3] * xy[0]])
    #     mat_r.append([p[1, 3] - p[2, 3] * xy[1]])
    #
    # mat_q = np.array(mat_q)
    # mat_r = np.array(mat_r)
    #
    # print(mat_q.shape)
    # print(mat_r.shape)

    n = m.shape[-1]

    mat_q = np.zeros((2 * n, 3))
    mat_r = np.zeros((2 * n, 1))

    for k in range(n):
        x = m[0, k] / m[2, k]
        y = m[1, k] / m[2, k]
        p = mat_p[k * 3:k * 3 + 3, :]
        mat_q[k * 2:k * 2 + 2, :] = np.array(
            [
                [p[2, 0] * x - p[0, 0], p[2, 1] * x - p[0, 1],
                 p[2, 2] * x - p[0, 2]],
                [p[2, 0] * y - p[1, 0], p[2, 1] * y - p[1, 1],
                 p[2, 2] * y - p[1, 2]]
            ]
        )

        mat_r[k * 2:k * 2 + 2, :] = np.array(
            [
                [p[0, 3] - p[2, 3] * x],
                [p[1, 3] - p[2, 3] * y]
            ]
        )

    mat_mr = np.vstack([np.dot(np.linalg.pinv(mat_q), mat_r), 1.0])  # Compute [Q.T * Q] * Q.T as pseudoinverse
    
    ms = mat_p @ mat_mr
    ms = np.reshape(ms, (n, 3)).T
    ms = ms / ms[-1, :]
    d = ms[0:1, :] - m[0:1, :]
    err = np.sqrt(np.sum(d * d, 1)).item()

    return mat_mr, ms, err
