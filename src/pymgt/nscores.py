from typing import Union, Type, Optional

import numpy as np
import scipy.stats

from numpy.typing import ArrayLike

from .transform import Transform

def univariate_generic_transform(x: ArrayLike,
                                 dist: Union[ArrayLike, Type[scipy.stats.rv_continuous]],
                                 weights: Optional[ArrayLike] = None,
                                 xminval: Optional[float] = None,
                                 xmaxval: Optional[float] = None,
                                 yminval: Optional[float] = None,
                                 ymaxval: Optional[float] = None):
    """The univariate transform to target distribution 'dist'
    build a bijection using the cumulative distribution which is mapped to 'dist'

    inputs
    ======

    x: data to transform. Each row represents a sample
    weights: weight assigned to each sample. if None, same weight to all samples
    dist: one of the univariate distributions in scipy.stats or a given scores of an experimental distribution
    xminval: For the lower tail extrapolation, this value is the minimum sample value allowed
    xmaxval: For the upper tail extrapolation, this value is the minimum sample value allowed
    yminval: For the lower tail extrapolation, this value is the minimum value of the target distribution
    ymaxval: For the lower tail extrapolation, this value is the minimum value of the target distribution

    return
    ======

    y: the transformed values of x
    score_table: the bijective table for interpolation
    """
    epsilon = 1.0e-7

    assert len(x.shape) == 1, "x must be one-dimensional array"

    add_tails = False

    if xminval is not None:
        assert xmaxval is not None
        assert xminval <= np.min(x)
        add_tails = True

    if xmaxval is not None:
        assert xmaxval is not None
        assert xmaxval >= np.max(x)
        add_tails = True

    ndata = x.shape[0]

    if weights is None:
        weights = np.ones(ndata)
    else:
        #make sure the weights sum up N
        assert np.round(np.sum(weights)) == float(ndata)

    x_unodd = x + np.random.random(ndata) * epsilon

    #sort data and weights

    sorted_indices = x_unodd.argsort()

    sorted_x = x[sorted_indices]
    sorted_weights = weights[sorted_indices]

    #normalize weights
    wsum = np.sum(sorted_weights)
    weights_normalised = sorted_weights/wsum

    # cumulative distribution
    if isinstance(dist, scipy.stats.rv_continuous):
        cumsum = np.cumsum(weights_normalised) - 0.5/wsum #centroids
        scores = dist.ppf(cumsum)
        if add_tails:
            score_table_y = np.empty(ndata+2)
            score_table_y[1:ndata+1] = scores

            #add extreme
            score_table_y[0] = yminval
            score_table_y[-1] = ymaxval
        else:
            score_table_y = np.empty(ndata)
            score_table_y[:] = scores
    else:
        scores = np.asarray(dist)
        if add_tails:
            assert len(scores) == ndata+2
            score_table_y = np.copy(scores)
        else:
            assert len(scores) == ndata, "%d vs. %d"%(len(scores), ndata)
            score_table_y = np.copy(scores)

    #transformation table for x
    if add_tails:
        score_table_x = np.empty(ndata+2)
        score_table_x[1:ndata+1] = sorted_x
        #add extremes
        score_table_x[0] = xminval
        score_table_x[-1] = xmaxval
    else:
        score_table_x = np.empty(ndata)
        score_table_x[:] = sorted_x

    #Interpolate
    y = np.interp(x, score_table_x, score_table_y)

    return y, (score_table_x, score_table_y, weights)

def univariate_nscore(x,
                      weights=None,
                      xminval=None,
                      xmaxval=None,
                      yminval=-6.0,
                      ymaxval=6.0,
                      gaussian_table=None):
    """The univariate normal score transform
    build a bijection using the cummulative distribution which is mapped to the standard
    cummulative distribution

    inputs
    ======

    x: data to transform. Each row represents a sample.
    weights: weight assigned to each sample. if None, same weight to all samples.
    xminval: For the lower tail extrapolation, this value is the minimum sample value allowed.
             default to min(x)
    xmaxval: For the upper tail extrapolation, this value is the minimum sample value allowed.
             default to max(x)
    yminval: For the lower tail extrapolation, this value is the minimum value of the standard
             Gaussian distribution. Default to -10.0
    ymaxval: For the lower tail extrapolation, this value is the minimum value of the standard
             Gaussian distribution. Default to 10.0

    return
    ======

    y: the transformed values of x
    nscore_table: the bijective table for interpolation
    """

    dist = gaussian_table if gaussian_table is not None else scipy.stats.norm

    return univariate_generic_transform(
        x,
        dist,
        weights=weights,
        xminval=xminval,
        xmaxval=xmaxval,
        yminval=yminval,
        ymaxval=ymaxval
    )


class UnivariateGaussianTransform(Transform):
    """UnivariateGaussianTransform: This Transform applies the NormalScore transform
    which construct a bijection between the cumulative distributions of the original
    variable and a standard Gaussian distribution. It applies only for univariate
    random variables.

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
    state : 2d-numpy array of shape (N,2) where N is the number of samples used
        to fit the transform. If tails are given, N can have additional extremes.

    Notes
    -----
    This Transform uses an ordered two-columns table that is used to interpolate.
    The first column of the table contains the original distribution, and the second,
    the Gaussian distribution.

    Examples
    --------
    >>> import numpy as np
    >>> from Transform import UnivariateGaussianTransform
    >>> nscore = UnivariateGaussianTransform()
    >>> x = np.random.random(size=1000)
    >>> print(np.mean(x))
    0.5154211433695641
    >>> print(np.std(x))
    0.2855446410326182
    >>> y = nscore.fit_transform(x)
    >>> print(np.mean(y))
    2.884803507186007e-15
    >>> print(np.std(y))
    0.9993494179950477
    >>> x_back = nscore.inverse_transform(np.random.normal(size=5000))
    >>> print(np.mean(x_back))
    0.5175033866089361
    >>> print(np.std(x_back))
    0.28483278531329337

    """

    def __init__(self, xminval=None, xmaxval=None, yminval=-6.0, ymaxval=6.0, **kargs):
        super(UnivariateGaussianTransform, self).__init__(**kargs)

        self._xminval = xminval
        self._xmaxval = xmaxval
        self._yminval = yminval
        self._ymaxval = ymaxval
        self._raw_table = None
        self._gaussian_table = None

    def _set_state(self, state):
        self._raw_table, self._gaussian_table = state

    def _get_state(self):
        return self._raw_table, self._gaussian_table

    def fit_transform(self, x, weights=None, gaussian_table=None):
        x = np.asarray(x)
        assert len(x.shape) == 1

        if gaussian_table is not None:
            self._gaussian_table = gaussian_table

        # create the transform table and transform `x`
        y, transform_table = univariate_nscore(
            x,
            weights=weights,
            xminval=self._xminval,
            xmaxval=self._xmaxval,
            yminval=self._yminval,
            ymaxval=self._ymaxval,
            gaussian_table=self._gaussian_table
        )

        # update state
        self._raw_table, self._gaussian_table, _ = transform_table

        return y

    def fit(self, x, gaussian_table=None):
        self.fit_transform(x, gaussian_table=gaussian_table)

    def transform(self, x):
        x = np.asarray(x)
        assert len(x.shape) == 1

        # interpolation from original space to Gaussian space
        y = np.interp(x, self._raw_table, self._gaussian_table)

        return y

    def inverse_transform(self, y):
        y = np.asarray(y)
        assert len(y.shape) == 1

        # interpolation from Gaussian space to original space
        x = np.interp(y, self._gaussian_table, self._raw_table)

        return x

    def to_hdf5(self, h5d):
        h5d.create_dataset("transform_table", data=self.__transform_table)

    def from_hdf5(self, h5d):
        self.__transform_table = np.array(h5d["transform_table"])

class MarginalGaussianTransform(Transform):
    """MarginalGaussianTransform: This Transform applies the NormalScore transform
    to each dimension in a multivariate scenario.

    Parameters
    ----------
    name : string, optional
        User name of Transform instance

    Attributes
    ----------
    name : string
        User name of Transform instance.
    ndim : int
        Number of dimensions of the multivariates.
    state : 2d-numpy array of shape (N,ndim+1) where N is the number of samples used
        to fit the transform. If tails are given, N can have additional extremes.

    Notes
    -----
    This Transform uses an ordered two-columns table that is used to interpolate.
    The first column of the table contains the original distribution, and the second,
    the Gaussian distribution.

    Examples
    --------
    >>> import numpy as np
    >>> from Transform import MarginalGaussianTransform
    >>> mvnscore = MarginalGaussianTransform(3)
    >>> x = np.random.random(size=(1000,3))
    >>> print(np.mean(x, axis=0))
    [0.47918057 0.50147569 0.48178349]
    >>> print(np.std(x, axis=0))
    [0.28771136 0.28915519 0.28325781]
    >>> y = mvnscore.fit_transform(x)
    >>> print(np.mean(y, axis=0))
    [2.86937141e-15 2.88458146e-15 2.87615765e-15]
    >>> print(np.std(y, axis=0))
    [0.99934942 0.99934942 0.99934942]
    >>> x_back = mvnscore.inverse_transform(np.random.normal(size=(5000,3)))
    >>> print(np.mean(x_back, axis=0))
    [0.47782644 0.49876823 0.48251466]
    >>> print(np.std(x_back, axis=0))
    [0.28540086 0.29105432 0.28495381]

    """

    def __init__(self, xminval=None, xmaxval=None, yminval=None, ymaxval=None, **kargs):
        super(MarginalGaussianTransform, self).__init__(name=kargs.get("name", None))


        if xminval is not None or xmaxval is not None or yminval is not None or ymaxval is not None:
            assert xminval is not None
            assert xmaxval is not None
            assert yminval is not None
            assert ymaxval is not None
            assert len(xminval) == len(xmaxval)
            assert len(xminval) == len(yminval)
            assert len(xminval) == len(ymaxval)

        self.__xminval = xminval
        self.__xmaxval = xmaxval
        self.__yminval = yminval
        self.__ymaxval = ymaxval

        self.__gaussian_table = None
        self.__weights = None
        self.__raw_tables = None
        self.__ndim = None

    def _set_state(self, state):
        self.__gaussian_table = np.copy(state['gaussian_table'])
        self.__weights =  np.copy(state['weights'])
        self.__raw_tables =  np.copy(state['raw_tables'])
        self.__ndim = state['ndim']

    def _get_state(self):
        return {
            'gaussian_table': self.__gaussian_table.copy(),
            'weights': self.__weights.copy(),
            'raw_tables': self.__raw_tables.copy(),
            'ndim': self.__ndim
        }

    def fit_transform(self, x, weights=None):
        ndata, ndim = x.shape

        self.__ndim = ndim

        y = np.empty_like(x)

        # transform the first variable
        y[:, 0], transform_table_1 = univariate_nscore(
            x[:, 0],
            weights=weights,
            xminval=None if self.__xminval is None else self.__xminval[0],
            xmaxval=None if self.__xmaxval is None else self.__xmaxval[0],
            yminval=None if self.__yminval is None else self.__yminval[0],
            ymaxval=None if self.__ymaxval is None else self.__ymaxval[0]
        )

        # we can reuse the second part of the transform_table from variable 1
        raw_table, gaussian_table, weights = transform_table_1

        self.__raw_tables = np.empty((ndata, self.__ndim))
        self.__raw_tables[:, 0] = raw_table

        self.__gaussian_table = gaussian_table
        self.__weights = weights

        for dim in range(1, ndim):
            y[:, dim], transform_table_ = univariate_nscore(
                x[:, dim],
                weights=self.__weights,
                xminval=None if self.__xminval is None else self.__xminval[dim],
                xmaxval=None if self.__xmaxval is None else self.__xmaxval[dim],
                yminval=None if self.__yminval is None else self.__yminval[dim],
                ymaxval=None if self.__ymaxval is None else self.__ymaxval[dim],
                gaussian_table=self.__gaussian_table
            )

            self.__raw_tables[:, dim] = transform_table_[0]

        return y

    def fit(self, x, weights=None):
        self.fit_transform(x, weights=weights)

    def transform(self, x: ArrayLike):
        _, ndim = x.shape
        assert ndim == self.__ndim

        y = np.empty_like(x)

        for dim in range(ndim):
            y[:, dim] = np.interp(x[:, dim], self.__raw_tables[:, dim], self.__gaussian_table)

        return y

    def inverse_transform(self, y: ArrayLike):
        _, ndim = y.shape
        assert ndim == self.__ndim

        x = np.empty_like(y)

        for dim in range(ndim):
            x[:, dim] = np.interp(y[:, dim], self.__gaussian_table, self.__raw_tables[:, dim])

        return x

    def state_to_dict(self, state=None):
        if state is not None:
            st = np.asarray(state)
        else:
            st = self.__transform_table

        ret = {
            "transform_table": st.tolist()
        }

        return ret

    def dict_to_state(self, content):
        self.__transform_table = np.asarray(content["transform_table"])
