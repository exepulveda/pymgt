"""Module with the sphering functions
"""
from typing import Optional
import numpy as np
import scipy.linalg

from .transform import Transform
from .interface import AbstractState
from .interface import Array2D

class SpheringState(AbstractState):
    """The state of a sphering transform
    """
    def __init__(self, sphering_matrix: Array2D, means: Optional[Array2D]):
        self.sphering_matrix = sphering_matrix
        self.means = means


class SpheringTransform(Transform):
    """SpheringTransform: Also called whitening. This Transform transforms an
    array-like `x` into a new array-like `y` where variables are uncorrelated and with
    unit variance. Optionally, it also the new array-like is centered, meaning that
    the mean value is zero.

    Parameters
    ----------
    name : string, optional
        User name of Transform instance
    centered : bool, optional
        If it is True, the original values are assumed centered already. Otherwise,
        the transformed values will be centered. Default value is False.

    Attributes
    ----------
    name : string
        User name of Transform instance.
    __centered : bool
        If centered is False, it will contain the mean values of `x`.
    state : any
        The internal representation of the Transform. It contains the matrix
        S and (optionally the mean values).

    Notes
    -----
    https://en.wikipedia.org/wiki/Whitening_transformation

    """

    def __init__(self, **kargs):
        super(SpheringTransform, self).__init__(**kargs)

        self.__centered = kargs.get("centered", False)
        self.__state = None

    def _set_state(self, state: SpheringState):
        self.__state = state

    def _get_state(self) -> SpheringState:
        return self.__state

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def fit(self, x):
        """This function will sphere the data. This means the data will now have a
        mean of 0 and a covariance matrix of 1.
        """
        if not self.__centered:
            muhat = np.mean(x, axis=0)
        else:
            muhat = None

        # covariance matrix
        cov = np.cov(x, rowvar=False)
        # eigenvalue/eigenvector decomposition
        D, V = np.linalg.eig(cov)
        Dinv = np.linalg.inv(np.diag(D))
        sq = scipy.linalg.sqrtm(Dinv)

        S = V@(sq@V.T)

        self.__state = SpheringState(S, muhat)

    def transform(self, x):
        """Apply to `x` the forward sphering using the `state`
        """
        S, muhat = self.__state.sphering_matrix, self.__state.means

        if muhat is not None:
            Xc = x - muhat
        else:
            Xc = x

        Z = Xc@S

        return Z

    def inverse_transform(self, y):
        """Apply to `y` the backward sphering using the `state`
        """
        S, muhat = self.__state.sphering_matrix, self.__state.means

        # apply inverse of S
        Sinv = np.linalg.inv(S)
        Xc = y@Sinv

        if muhat is not None:
            return Xc + muhat

        return Xc

    def to_hdf5(self, h5d):
        S, xhat = self.__state

        h5d.create_dataset("S", data=S)
        h5d.create_dataset("xhat", data=xhat)

    def from_hdf5(self, h5d):
        S = np.array(h5d["S"])
        xhat = np.array(h5d["xhat"])
        self.__state = (S, xhat)
