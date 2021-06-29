"""Utilities and atomic functions
"""
import math
import numpy as np
from scipy.stats import multivariate_normal

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

def generate_directions(ndim, size=100):
    """Generate `size` uniform directions of dimension `ndim`
    """
    v = np.random.normal(size=(size, ndim))
    s = np.sqrt(np.sum(v**2, axis=1))

    for i in range(ndim):
        v[:, i] /= s
        #v[:,i] /= np.linalg.norm(v[:,i])

    for i in range(size):
        v[i, :] /= np.linalg.norm(v[i, :])

    return v

def uniform_on_surface(ndim, size):
    """Generate `size` uniform directions of dimension `ndim`
    """
    x = np.random.normal(size=(size, ndim))
    x /= np.linalg.norm(x, axis=1)[:, np.newaxis]
    return x

def orthonormal_basis(a):
    """Determine an orthonormal basis with the first vector `a`
    """
    #a is the first direction
    d = len(a)

    #orthonormal basis of a
    U = np.eye(d)
    U[:, 0] = a #the first column is the first

    Q, R = np.linalg.qr(U) #Q is the orthonormal component having the first column the direction a (normalised)

    return Q

def mv_index_distribution(ndata, ndim, pi_func, ndir=100):
    """Calculate the projection index `pi_func` for samples from 
    a standard multivariate Gaussian distribution with unit covariance
    of size (`ndata`,`ndim`) for
    `ndir` directions at random
    """
    #generate random directions
    directions = uniform_on_surface(ndim, ndir)
    ret = []
    x = multivariate_normal.rvs(mean=np.zeros(ndim), cov=np.eye(ndim), size=ndata)
    for i in range(ndir):
        #project
        p = x@directions[i, :]
        pi = pi_func(p)
        ret += [pi]

    return ret
