"""Normal score transform for the univariate and marginal cases
"""
import pytest

from typing import Union, Type, Optional, List, Tuple, Dict

import numpy as np
import scipy.stats

from .transform import Transform
from .interface import AbstractState, Vector
from .interface import Array2D


def univariate_generic_transform(
        x: Vector,
        dist: Union[Array2D, Type[scipy.stats.rv_continuous]],
        weights: Optional[Vector] = None,
        minval=None,
        maxval=None,
        lower_extrapolation_mode: Optional[str]=None,
        lower_extrapolation_param: Optional[float]=1.0,
        upper_extrapolation_mode: Optional[str]=None,
        upper_extrapolation_param: Optional[float]=1.0) -> Vector:
    """The univariate transform to target distribution 'dist'
    build a bijection using the cumulative distribution
    which is mapped to 'dist'

    inputs
    ======

    x: data to transform. Each row represents a sample
    weights: weight assigned to each sample.
        If None, same weight to all samples
    dist: one of the univariate distributions in scipy.stats or
        a given scores of an experimental distribution
    minval: For the lower tail extrapolation,
        this value is the minimum sample value allowed.
        default to min(x)
    maxval: For the upper tail extrapolation,
        this value is the maximum sample value allowed.
        default to max(x)
    lower_extrapolation_mode: The extrapolation mode for the lower tail.
        The options are: linear, truncate and power
    lower_extrapolation_param: The extrapolation parameter for the lower tail.
        Only valid for power
    upper_extrapolation_mode: The extrapolation mode for the upper tail.
        The options are: linear, truncate, hyperbolic and power
    upper_extrapolation_param: The extrapolation parameter for the upper tail.
        Only valid for power and hyperbolic

    return
    ======

    y: the transformed values of x
    score_table: the bijective table for interpolation
    """
    epsilon = 1.0e-7

    assert len(x.shape) == 1, "x must be one-dimensional array"

    if minval is None:
        minval = np.min(x)

    if maxval is None:
        maxval = np.max(x)

    ndata = x.shape[0]

    if weights is None:
        weights = np.ones(ndata)
    else:
        # make sure the weights sum up N
        assert np.round(np.sum(weights)) == float(ndata)

    x_unodd = x + np.random.random(ndata) * epsilon

    # sort data and weights

    sorted_indices = x_unodd.argsort()

    sorted_x = x[sorted_indices]
    sorted_weights = weights[sorted_indices]

    # normalize weights
    wsum = np.sum(sorted_weights)
    weights_normalised = sorted_weights/wsum

    # cumulative distribution
    if isinstance(dist, scipy.stats.rv_continuous):
        cumsum = np.cumsum(weights_normalised) - 0.5/wsum  # centroids
        score_table_y = dist.ppf(cumsum)
    else:
        scores = np.asarray(dist)
        assert len(scores) == ndata, "%d vs. %d" % (len(scores), ndata)
        score_table_y = np.copy(scores)

    # transformation table for x
    score_table_x = np.empty(ndata)
    score_table_x[:] = sorted_x

    # Interpolate
    y = forward_interpolation(
            x, score_table_x, score_table_y,
            minval, maxval,
            lower_extrapolation_mode, lower_extrapolation_param,
            upper_extrapolation_mode, upper_extrapolation_param)

    return y, (score_table_x, score_table_y, weights)


def univariate_nscore(x: Vector,
                      weights: Optional[Vector]=None,
                      minval=None,
                      maxval=None,
                      lower_extrapolation_mode: Optional[str]=None,
                      lower_extrapolation_param: Optional[float]=1.0,
                      upper_extrapolation_mode: Optional[str]=None,
                      upper_extrapolation_param: Optional[float]=1.0,
                      gaussian_table=None) -> Vector:
    """The univariate normal score transform
    build a bijection using the cumulative distribution
    which is mapped to the standard cumulative distribution

    inputs
    ======

    x: data to transform. Each row represents a sample.
    weights: weight assigned to each sample.
        If None, same weight to all samples.
    minval: For the lower tail extrapolation,
        this value is the minimum sample value allowed.
        default to min(x)
    maxval: For the upper tail extrapolation,
        this value is the maximum sample value allowed.
        default to max(x)
    lower_extrapolation_mode: The extrapolation mode for the lower tail.
        The options are: linear, truncate and power
    lower_extrapolation_param: The extrapolation parameter for the lower tail.
        Only valid for power
    upper_extrapolation_mode: The extrapolation mode for the upper tail.
        The options are: linear, truncate, hyperbolic and power
    upper_extrapolation_param: The extrapolation parameter for the upper tail.
        Only valid for power and hyperbolic

    return
    ======

    y: the transformed values of x
    nscore_table: the bijective table for interpolation
    """

    dist = gaussian_table if gaussian_table is not None else scipy.stats.norm

    return univariate_generic_transform(
        np.asarray(x),
        dist,
        weights=weights,
        minval=np.min(x) if minval is None else minval,
        maxval=np.max(x) if maxval is None else maxval,
        lower_extrapolation_mode=lower_extrapolation_mode,
        lower_extrapolation_param=lower_extrapolation_param,
        upper_extrapolation_mode=upper_extrapolation_mode,
        upper_extrapolation_param=upper_extrapolation_param
    )


class UnivariateGaussianState(AbstractState):
    """The state of UnivariateGaussianTransform"""
    def __init__(self, raw_table: Vector, gaussian_table: Vector, params: Dict):
        self.raw_table = raw_table
        self.gaussian_table = gaussian_table
        self.params = params


class UnivariateGaussianTransform(Transform):
    """UnivariateGaussianTransform: This Transform applies
    the NormalScore transform which construct a bijection between
    the cumulative distributions of the original variable and
    a standard Gaussian distribution. It applies only for univariate
    random variables.

    Parameters
    ----------
    name : string, optional
        User name of Transform instance
    centered : bool, optional
        If it is True, the original values are assumed centered already.
        Otherwise, the transformed values will be centered.
        Default value is False.

    Attributes
    ----------
    name : string
        User name of Transform instance.
    state : 2d-numpy array of shape (N,2) where N is the number of samples used
        to fit the transform. If tails are given,
        N can have additional extremes.

    Notes
    -----
    This Transform uses an ordered two-columns table that is used to interpolate.
    The first column of the table contains the original distribution,
    and the second, the Gaussian distribution.

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

    def __init__(self, **kargs):
        super(UnivariateGaussianTransform, self).__init__(**kargs)

        self._minval = kargs.get("minval", None)
        self._maxval = kargs.get("maxval", None)
        self._lextra_mode=kargs.get("lower_extrapolation_mode", "linear")
        self._lextra_param=kargs.get("lower_extrapolation_param", None)
        self._uextra_mode=kargs.get("upper_extrapolation_mode", "linear")
        self._uextra_param=kargs.get("upper_extrapolation_param", None)

    def __get_params__(self) -> Dict:
        return {
            "minval": self._minval,
            "maxval": self._maxval,
            "lower_extrapolation_mode": self._lextra_mode,
            "lower_extrapolation_param": self._lextra_param,
            "upper_extrapolation_mode": self._uextra_mode,
            "upper_extrapolation_param": self._uextra_param,
        }

    def __set_params__(self, params: Dict):
        self._minval = params["minval"]
        self._maxval = params["maxval"]
        self._lextra_mode = params["lower_extrapolation_mode"]
        self._lextra_param = params["lower_extrapolation_param"]
        self._uextra_mode = params["upper_extrapolation_mode"]
        self._uextra_param = params["upper_extrapolation_param"]

    def fit_transform(self, x, weights=None, gaussian_table=None):
        x = np.asarray(x)
        assert len(x.shape) == 1

        # create the transform table and transform `x`
        y, transform_table = univariate_nscore(
            x,
            weights=weights,
            minval=self._minval,
            maxval=self._maxval,
            lower_extrapolation_mode=self._lextra_mode,
            lower_extrapolation_param=self._lextra_param,
            upper_extrapolation_mode=self._uextra_mode,
            upper_extrapolation_param=self._uextra_param,
            gaussian_table=gaussian_table
        )

        # update state
        raw_table, gaussian_table, _ = transform_table
        self.state = UnivariateGaussianState(raw_table, gaussian_table, self.__get_params__())

        return y

    def fit(self, x: Array2D, gaussian_table: Optional[Vector]=None):
        self.fit_transform(x, gaussian_table=gaussian_table)

    def transform(self, x: Array2D) -> Array2D:
        x = np.asarray(x)
        assert len(x.shape) == 1

        self.__set_params__(self.state.params)

        # interpolation from original space to Gaussian space
        y = forward_interpolation(
            x,
            self.state.raw_table, self.state.gaussian_table,
            minval=self._minval,
            maxval=self._maxval,
            lower_extrapolation_mode=self._lextra_mode,
            lower_extrapolation_param=self._lextra_param,
            upper_extrapolation_mode=self._uextra_mode,
            upper_extrapolation_param=self._uextra_param
        )

        return y

    def inverse_transform(self, y: Array2D) -> Array2D:
        y = np.asarray(y)
        assert len(y.shape) == 1

        self.__set_params__(self.state.params)

        # interpolation from Gaussian space to original space
        x = backward_interpolation(
            y,
            self.state.raw_table, self.state.gaussian_table,
            minval=self._minval,
            maxval=self._maxval,
            lower_extrapolation_mode=self._lextra_mode,
            lower_extrapolation_param=self._lextra_param,
            upper_extrapolation_mode=self._uextra_mode,
            upper_extrapolation_param=self._uextra_param
        )

        return x


class MarginalGaussianState(AbstractState):
    """The state of a MarginalGaussianTransform
    """
    def __init__(self, minval: Vector, maxval: Vector,
                 raw_tables: List[Vector], gaussian_table: Vector,
                 weights: Vector, ndim: int):

        assert minval is not None
        assert maxval is not None
        assert raw_tables is not None
        assert gaussian_table is not None

        self.minval = minval
        self.maxval = maxval
        self.gaussian_table = gaussian_table
        self.weights = weights
        self.raw_tables = raw_tables
        self.ndim = ndim


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

    def __init__(self, **kargs):
        super(MarginalGaussianTransform, self).__init__(name=kargs.get("name", None))

        self._minval = kargs.get("minval", None)
        self._maxval = kargs.get("maxval", None)

        if self._minval is not None or self._maxval is not None:
            assert len(self._minval) == len(self._maxval)

        self._lextra_mode = kargs.get("lower_extrapolation_mode", "linear")
        self._lextra_param = kargs.get("lower_extrapolation_param", None)
        self._uextra_mode = kargs.get("upper_extrapolation_mode", "linear")
        self._uextra_param = kargs.get("upper_extrapolation_param", None)

    def fit_transform(self, x: Array2D, weights: Optional[Vector]=None) -> Array2D:
        ndata, ndim = x.shape

        y = np.empty_like(x)

        if self._minval is None:
            self._minval = np.min(x, axis=0)

        if self._maxval is None:
            self._maxval = np.max(x, axis=0)


        # transform the first variable
        y[:, 0], transform_table_1 = univariate_nscore(
            x[:, 0],
            weights=weights,
            minval=self._minval[0],
            maxval=self._maxval[0],
            lower_extrapolation_mode=self._lextra_mode,
            lower_extrapolation_param=self._lextra_param,
            upper_extrapolation_mode=self._uextra_mode,
            upper_extrapolation_param=self._uextra_param
        )

        # we can reuse the second part of the transform_table from variable 1
        raw_table, gaussian_table, weights = transform_table_1

        raw_tables = np.empty((ndata, ndim))
        raw_tables[:, 0] = raw_table

        for dim in range(1, ndim):
            y[:, dim], transform_table_ = univariate_nscore(
                x[:, dim],
                weights=weights,
                minval=self._minval[dim],
                maxval=self._maxval[dim],
                lower_extrapolation_mode=self._lextra_mode,
                lower_extrapolation_param=self._lextra_param,
                upper_extrapolation_mode=self._uextra_mode,
                upper_extrapolation_param=self._uextra_param,
                gaussian_table=gaussian_table
            )

            raw_tables[:, dim] = transform_table_[0]

        self.state = MarginalGaussianState(self._minval, self._maxval, raw_tables, gaussian_table, weights, ndim)

        return y

    def fit(self, x:Array2D, weights: Optional[Vector]=None):
        self.fit_transform(x, weights=weights)

    def transform(self, x: Array2D) -> Array2D:
        _, ndim = x.shape
        assert ndim == self.state.ndim

        y = np.empty_like(x)

        for dim in range(ndim):
            y[:, dim] = forward_interpolation(
                x[:, dim],
                self.state.raw_tables[:, dim], self.state.gaussian_table,
                minval=self.state.minval[dim],
                maxval=self.state.maxval[dim],
                lower_extrapolation_mode=self._lextra_mode,
                lower_extrapolation_param=self._lextra_param,
                upper_extrapolation_mode=self._uextra_mode,
                upper_extrapolation_param=self._uextra_param
            )

        return y

    def inverse_transform(self, y: Array2D) -> Array2D:
        _, ndim = y.shape
        assert ndim == self.state.ndim

        x = np.empty_like(y)

        for dim in range(ndim):
            x[:, dim] = backward_interpolation(
                y[:, dim],
                self.state.raw_tables[:, dim], self.state.gaussian_table,
                minval=self.state.minval[dim],
                maxval=self.state.maxval[dim],
                lower_extrapolation_mode=self._lextra_mode,
                lower_extrapolation_param=self._lextra_param,
                upper_extrapolation_mode=self._uextra_mode,
                upper_extrapolation_param=self._uextra_param
            )

        return x

def power_interpolation(x: float, xlower: float, xupper: float, ylower: float, yupper: float, power: float=1.0) -> float:
    if x < xlower:
        return ylower

    if x > xupper:
        return xupper

    if (xupper - xlower) < np.finfo(float).eps:
        return (yupper + ylower) / 2.0

    return ylower + (yupper - ylower) * ((x-xlower)/(xupper-xlower))**power

def forward_interpolation(x: Vector, raw_table: Vector, gaussian_table: Vector, minval: float, maxval: float,
                          lower_extrapolation_mode: Optional[str]=None, lower_extrapolation_param: Optional[float]=1.0,
                          upper_extrapolation_mode: Optional[str]=None, upper_extrapolation_param: Optional[float]=1.0
                         ) -> Vector:
    if lower_extrapolation_mode is None or lower_extrapolation_mode == "linear":
        power_lower = 1.0
    elif lower_extrapolation_mode == "power":
        if lower_extrapolation_param is None or lower_extrapolation_param == 0.0:
            raise ValueError("Lower tail extrapolation param must be not zero real number")
        power_lower = 1.0 / lower_extrapolation_param
    elif lower_extrapolation_mode != "truncate":
        raise ValueError("Invalid lower tail extrapolation mode [%s]"%lower_extrapolation_mode)

    if upper_extrapolation_mode is None or upper_extrapolation_mode == "linear":
        power_upper = 1.0
    elif upper_extrapolation_mode == "power":
        if upper_extrapolation_param is None or upper_extrapolation_param == 0.0:
            raise ValueError("Upper tail extrapolation param must be not zero real number")
        power_upper = 1.0 / upper_extrapolation_param
    else:
        raise ValueError("Invalid upper tail extrapolation mode [%s]"%upper_extrapolation_mode)

    x = np.asarray(x)

    # note that numpy will truncate in extrapolation case
    y = np.interp(x, raw_table, gaussian_table)

    # quick return
    if lower_extrapolation_mode == "truncate" and upper_extrapolation_mode == "truncate":
        return y  # interp will truncate the extremes 

    # check for lower tail
    indices = np.where(x < raw_table[0])[0]
    for i in indices:
        # compute the interpolation of the cdf
        icdf = power_interpolation(x[i],
                                   minval, raw_table[0],
                                   0.0, scipy.stats.norm.cdf(gaussian_table[0]),
                                   power=power_lower)
        # transform back
        y[i] = scipy.stats.norm.ppf(icdf)

    # check for upper tail
    indices = np.where(x > raw_table[-1])[0]
    for i in indices:
        #import pdb; pdb.set_trace()
        icdf = power_interpolation(x[i],
                                   raw_table[-1], maxval,
                                   scipy.stats.norm.cdf(gaussian_table[-1]), 1.0,
                                   power=power_upper)

        y[i] = scipy.stats.norm.ppf(icdf)
    return y


def backward_interpolation(y: Vector, raw_table: Vector, gaussian_table: Vector, minval: float, maxval: float,
                          lower_extrapolation_mode: Optional[str]=None, lower_extrapolation_param: Optional[float]=1.0,
                          upper_extrapolation_mode: Optional[str]=None, upper_extrapolation_param: Optional[float]=1.0
                         ) -> Vector:
    if lower_extrapolation_mode is None or lower_extrapolation_mode == "linear":
        power_lower = 1.0
    elif lower_extrapolation_mode == "power":
        if lower_extrapolation_param is None or lower_extrapolation_param == 0.0:
            raise ValueError("Lower tail extrapolation param must be not zero real number")
        power_lower = 1.0 / lower_extrapolation_param
    elif lower_extrapolation_mode != "truncate":
        raise ValueError("Invalid lower tail extrapolation mode [%s]"%lower_extrapolation_mode)

    if upper_extrapolation_mode is None or upper_extrapolation_mode == "linear":
        power_upper = 1.0
    elif upper_extrapolation_mode == "power":
        if upper_extrapolation_param is None or upper_extrapolation_param == 0.0:
            raise ValueError("Upper tail extrapolation param must be not zero real number")
        power_upper = 1.0 / upper_extrapolation_param
    elif upper_extrapolation_mode == "truncate":
        pass
    elif upper_extrapolation_mode == "hyperbolic":
        factor_upper = (raw_table[-1]**upper_extrapolation_param)*(1.0-scipy.stats.norm.cdf(gaussian_table[-1]))
        power_upper = 1.0 / upper_extrapolation_param
    else:
        raise ValueError("Invalid upper tail extrapolation mode [%s]"%upper_extrapolation_mode)

    y = np.asarray(y)

    # note that numpy will truncate in extrapolation case
    x = np.interp(y, gaussian_table, raw_table)

    # quick return
    if lower_extrapolation_mode == "truncate" and upper_extrapolation_mode == "truncate":
        return x  # interp will truncate the extremes 


    # check for lower tail
    indices = np.where(y < gaussian_table[0])[0]
    for i in indices:
        x[i] = power_interpolation(
                    scipy.stats.norm.cdf(y[i]),
                    0.0, scipy.stats.norm.cdf(gaussian_table[0]),
                    minval, raw_table[0],
                    power=power_lower
                )

    # check for upper tail
    indices = np.where(y > gaussian_table[-1])[0]
    for i in indices:
        if upper_extrapolation_mode == "hyperbolic":
            x[i] = (factor_upper/(1.0 - scipy.stats.norm.cdf(y[i])))**power_upper
        else:
            x[i] = power_interpolation(
                        scipy.stats.norm.cdf(y[i]),
                        scipy.stats.norm.cdf(gaussian_table[-1]), 1.0,
                        raw_table[-1], maxval,
                        power=power_upper
                    )

    return x
