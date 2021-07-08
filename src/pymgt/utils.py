"""Utilities and atomic functions
"""
import math
import numpy as np


def spherical2cartesian(a):
    """Transform the spherical coordinate system `a` to cartesian
    """
    ndim = len(a)
    x = np.empty(ndim+1)

    asin = np.sin(a)
    acos = np.cos(a)

    for i in range(ndim):
        x[i] = acos[i]
        for j in range(i):
            x[i] *= asin[j]

    x[-1] = 1.0
    for j in range(ndim):
        x[-1] *= asin[j]

    return x


def cartesian2spherical(v):
    """Transform the cartesian coordinate system `v` to spherical
    """
    ndim = len(v)
    a = np.empty(ndim-1)

    v2 = np.power(v, 2)

    for i in range(ndim-2):
        a[i] = math.acos(v[i]/np.sqrt(np.sum(v2[i:])))

    d = math.acos(v[-2]/np.sqrt(v2[-1] + v2[-2]))
    if v[-1] >= 0.0:
        a[-1] = d
    else:
        a[-1] = 2*np.pi - d

    return a


def orthonormal_basis(a):
    """Determine an orthonormal basis with the first vector `a`
    """
    # a is the first direction
    d = len(a)

    # orthonormal basis of a
    U = np.eye(d)
    U[:, 0] = a  # the first column is the first

    # Q is the orthonormal component having
    # the first column the direction a (normalised)
    Q, R = np.linalg.qr(U)

    return Q
