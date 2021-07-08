'''
This module implements/define several indices for departure from Gaussianity
'''
import scipy.stats
import numpy as np

from scipy.stats import multivariate_normal

from typing import Callable, Optional
from .interface import Array2D
from .interface import Vector

def generate_directions(dim: int, n: int = 100) -> Array2D:
    """Generate `n` directions of dimension `dim`
    uniformely distributed on the n-sphere
    """
    v = np.random.normal(size=(n, dim))
    s = np.sqrt(np.sum(v**2, axis=1))

    for i in range(dim):
        v[:, i] /= s

    for i in range(n):
        v[i, :] /= np.linalg.norm(v[i, :])

    return v

def uniform_on_surface(ndim, size):
    """Generate `size` uniform directions of dimension `ndim`
    """
    x = np.random.normal(size=(size, ndim))
    x /= np.linalg.norm(x, axis=1)[:, np.newaxis]
    return x

def mv_index_distribution(ndata, ndim, pi_func, ndir=100):
    """Calculate the projection index `pi_func` for samples from
    a standard multivariate Gaussian distribution with unit covariance
    of size (`ndata`,`ndim`) for
    `ndir` directions at random
    """
    # generate random directions
    directions = uniform_on_surface(ndim, ndir)
    ret = []
    x = multivariate_normal.rvs(
            mean=np.zeros(ndim),
            cov=np.eye(ndim),
            size=ndata)
    for i in range(ndir):
        # project
        p = x@directions[i, :]
        pi = pi_func(p)
        ret += [pi]

    return ret


def projection_index(x: Array2D,
                     index_func: Callable,
                     nprojections: Optional[int] = 100,
                     reduce: Optional[Callable] = np.mean) -> float:
    """Compute the projection `index_func` to `x` (mutivariate),
    therefore multiple `nprojections` are generated and reduced by the
    operator `reduce`
    """
    x = np.asarray(x)
    _, ndim = x.shape

    directions = generate_directions(ndim, n=nprojections)

    indices = [index_func(x@d) for d in directions]

    return reduce(indices)


class Projectable:
    """Class for defining projectable univariate indices used in mutivariate data
    """
    def __init__(self,
                 index_func: Callable,
                 nprojections: Optional[int] = 100,
                 reduce: Optional[Callable] = np.mean):
        self.__func = index_func
        self.__nprojections = nprojections
        self.__reduce = reduce

    def __call__(self, x: Array2D) -> float:
        return projection_index(x,
                                self.__func, nprojections=self.__nprojections,
                                reduce=self.__reduce)


# normality tests
def jarque_bera_index(x: Vector) -> float:
    return scipy.stats.jarque_bera(x)[0]


def shapiro_index(x: Vector) -> float:
    return scipy.stats.shapiro(x)[0]


def anderson_index(x: Vector) -> float:
    return scipy.stats.anderson(x)[0]


def ks_index(x: Vector) -> float:
    return scipy.stats.kstest(x, "norm")[0]
