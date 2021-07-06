import numpy as np
import scipy
import scipy.stats
import pytest

from pymgt import *
from pymgt.nscores import univariate_nscore
from pymgt.nscores import forward_interpolation
from pymgt.nscores import backward_interpolation
from test_utils import *


def test_normalscore_default(alpha=0.05):
    decimals = 2

    x = np.random.uniform(10.0, 20.0, size=1000)

    y, state = univariate_nscore(x)

    score_table_x, score_table_y, weights = state

    # check statistical tests
    t, pvalue = scipy.stats.kstest(y, "norm")

    assert t <= 0.005
    assert pvalue >= alpha

    # weights should be ones
    np.testing.assert_array_almost_equal(weights, np.ones_like(weights))

    # score tables should be sorted
    assert np.all(np.diff(score_table_x) >= 0)
    assert np.all(np.diff(score_table_y) >= 0)

    # first and last element in raw data should be minimum and maximum
    assert score_table_x[0] == np.min(x)
    assert score_table_x[-1] == np.max(x)

def test_normalscore_weights(alpha=0.05):
    decimals = 2

    sample_size = 1000

    x = np.random.uniform(10.0, 20.0, size=sample_size)

    weights = np.random.uniform(0.5, 0.8, size=sample_size)
    weights = weights / np.sum(weights) * sample_size
    # weights should sum up the sample_size
    np.testing.assert_almost_equal(np.sum(weights), sample_size)

    y, state = univariate_nscore(x, weights=weights)

    score_table_x, score_table_y, weights = state

    # check statistical tests
    t, pvalue = scipy.stats.kstest(y, "norm")
    assert pvalue >= alpha

    # weights should sum up the sample_size
    np.testing.assert_almost_equal(np.sum(weights), sample_size)

    # score tables should be sorted
    assert_array_sorted(score_table_x)
    assert_array_sorted(score_table_y)

    # first and last element in raw data should be minimum and maximum
    assert score_table_x[0] == np.min(x)
    assert score_table_x[-1] == np.max(x)

def test_normalscore_minmax(alpha=0.05):
    sample_size = 1000

    x = np.random.uniform(10.0, 20.0, size=sample_size)

    weights = np.random.uniform(0.5, 0.8, size=sample_size)
    weights = weights / np.sum(weights) * sample_size
    # weights should sum up the sample_size
    np.testing.assert_almost_equal(np.sum(weights), sample_size)

    xminval = np.min(x)*0.9
    xmaxval = np.max(x)*1.1
    yminval = -8.0
    ymaxval = 8.0

    y, state = univariate_nscore(x, weights=weights, xminval=xminval, xmaxval=xmaxval, yminval=yminval, ymaxval=ymaxval)

    score_table_x, score_table_y, weights = state

    # check statistical tests
    t, pvalue = scipy.stats.kstest(y, "norm")

    pvalue >= alpha

    # weights should sum up the sample_size
    np.testing.assert_almost_equal(np.sum(weights), sample_size)

    # score tables should be sorted
    assert_array_sorted(score_table_x)
    assert_array_sorted(score_table_y)

    # first and last element in tables should be minimum and maximum
    assert score_table_x[0] == xminval
    assert score_table_x[-1] == xmaxval
    assert score_table_y[0] == yminval
    assert score_table_y[-1] == ymaxval

    assert len(score_table_x) == sample_size + 2
    assert len(score_table_y) == sample_size + 2

def test_normalscore_tables():

    x = [1.0, 2.0, 3.0, 4.0, 5.0]

    xminval = 0.0
    xmaxval = 8.0
    yminval = -10.0
    ymaxval = 10.0

    y, state = univariate_nscore(x, xminval=xminval, xmaxval=xmaxval, yminval=yminval, ymaxval=ymaxval)

    score_table_x, score_table_y, weights = state

    np.testing.assert_array_almost_equal(score_table_x, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0], decimal=4)
    # y table should ppf of 0.1 0.3 0.5 0.7 0.9
    ytable = np.empty(len(x)+2)
    ytable[1:-1] = scipy.stats.norm.ppf([0.1, 0.3, 0.5, 0.7, 0.9])
    ytable[0] = yminval
    ytable[-1] = ymaxval
    np.testing.assert_array_almost_equal(score_table_y, ytable, decimal=4)
    np.testing.assert_array_almost_equal(weights, np.ones_like(y), decimal=4)
    np.testing.assert_array_almost_equal(y, ytable[1:-1], decimal=4)


def test_marginal_transform():
    ndim = 4
    ndata = 1000

    np.random.seed(1)

    x = np.random.uniform(10.0, 20.0, size=(ndata, ndim))

    t = MarginalGaussianTransform()

    y, x_back = assert_reversibility(t, x)        

    for dim in range(ndim):
        assert_normality(y[:, dim])

    # forward transform should return the same y

    y_forward = t.transform(x)

    np.testing.assert_array_almost_equal(y, y_forward)

def test_marginal_state():
    ndim = 10
    ndata = 10000

    np.random.seed(1)

    x = np.random.uniform(10.0, 20.0, size=(ndata, ndim))

    t1 = MarginalGaussianTransform()

    y1 = t1.fit_transform(x)
    t_state = t1.state

    t2 = MarginalGaussianTransform()
    t2.state = t_state
    y2 = t2.transform(x)

    np.testing.assert_array_almost_equal(y1, y2)

def test_forward_interpolation():
    raw_table = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    gaussian_table = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    # assertions
    # minval
    with pytest.raises(ValueError):
        y = forward_interpolation([15.0], raw_table, gaussian_table, 40.0, 70.0)
    # maxval
    with pytest.raises(ValueError):
        y = forward_interpolation([15.0], raw_table, gaussian_table, 0.0, 45.0)
    # extrapolation_mode
    with pytest.raises(ValueError):
        y = forward_interpolation([15.0], raw_table, gaussian_table, 0.0, 70.0,
                                  lower_extrapolation_mode="unkown")
    # extrapolation_mode
    with pytest.raises(ValueError):
        y = forward_interpolation([15.0], raw_table, gaussian_table, 0.0, 70.0,
                                  upper_extrapolation_mode="unkown")

    minval = 0.0
    maxval = 100.0
    lower_extrapolation_mode = "linear"
    lower_extrapolation_param = None

    # linear interpolation 15.0 -> -1.5
    y = forward_interpolation([15.0], raw_table, gaussian_table, minval, maxval,
                          lower_extrapolation_mode, lower_extrapolation_param)

    assert 0 <= scipy.stats.norm.cdf(y[0]) <= 1.0
    assert y[0] == -1.5

    minval = 0.0
    maxval = 100.0
    lower_extrapolation_mode = "linear"
    lower_extrapolation_param = None

    # linear interpolation of 5.0 using
    # (0.0, 10.0) and (0.0, ppf(-2.0)) -> -2.2776
    y = forward_interpolation([5.0], raw_table, gaussian_table, minval, maxval,
                          lower_extrapolation_mode, lower_extrapolation_param)

    assert 0 <= scipy.stats.norm.cdf(y[0]) <= 1.0
    np.testing.assert_almost_equal(y, -2.2776048388094594)

    # linear interpolation of 60.0 using   
    # (50.0, 70.0) and (ppf(2.0), 1.0) -> 2.2776
    minval = 0.0
    maxval = 70.0
    upper_extrapolation_mode = "linear"
    upper_extrapolation_param = None

    y = forward_interpolation([60.0], raw_table, gaussian_table, minval, maxval,
                          upper_extrapolation_mode, upper_extrapolation_param)

    assert 0 <= scipy.stats.norm.cdf(y[0]) <= 1.0
    np.testing.assert_almost_equal(y, 2.27760484)

    # linear interpolation (but using power 1.0) of [5.0, 60.0] using
    minval = 0.0
    maxval = 70.0
    lower_extrapolation_mode = "power"
    lower_extrapolation_param = 1.0
    upper_extrapolation_mode = "power"
    upper_extrapolation_param = 1.0

    y = forward_interpolation([5.0, 60.0], raw_table, gaussian_table, minval, maxval,
                          upper_extrapolation_mode, upper_extrapolation_param)

    assert 0 <= scipy.stats.norm.cdf(y[0]) <= 1.0
    assert 0 <= scipy.stats.norm.cdf(y[1]) <= 1.0
    np.testing.assert_array_almost_equal(y, [-2.27760484, 2.27760484])

def test_backward_interpolation():
    raw_table = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    gaussian_table = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    # assertions
    # minval
    with pytest.raises(ValueError):
        x = backward_interpolation([-1.5], raw_table, gaussian_table, 40.0, 70.0)
    # maxval
    with pytest.raises(ValueError):
        x = backward_interpolation([-1.5], raw_table, gaussian_table, 0, 35.0)
    # extrapolation_mode
    with pytest.raises(ValueError):
        x = backward_interpolation([-1.5], raw_table, gaussian_table, 0.0, 60.0,
                                  lower_extrapolation_mode="unkown")
    # extrapolation_mode
    with pytest.raises(ValueError):
        x = backward_interpolation([-1.5], raw_table, gaussian_table, 0.0, 60.0,
                                  upper_extrapolation_mode="unkown")

    # linear interpolation working
    minval = 0.0
    maxval = 60.0
    lower_extrapolation_mode = "linear"
    lower_extrapolation_param = None

    x = backward_interpolation([-1.5], raw_table, gaussian_table, minval, maxval,
                          lower_extrapolation_mode, lower_extrapolation_param)

    assert x[0] == 15.0

    # linear extrapolation of -2.5 using
    # (0.0, cdf(-2.5)) and (0.0, 10.0) -> 2.72950739
    minval = 0.0
    maxval = 60.0
    lower_extrapolation_mode = "linear"
    lower_extrapolation_param = None


    y = backward_interpolation([-2.5], raw_table, gaussian_table, minval, maxval,
                          lower_extrapolation_mode, lower_extrapolation_param)

    assert minval <= y[0] <= maxval
    np.testing.assert_almost_equal(y, [2.72950739])

    # linear extrapolation of 2.5 using   
    # (0.0, cdf(-2.5)) and (50.0, 70.0) -> 64.54098522160602
    minval = 0.0
    maxval = 70.0
    upper_extrapolation_mode = "linear"
    upper_extrapolation_param = None

    y = backward_interpolation([2.5], raw_table, gaussian_table, minval, maxval,
                          upper_extrapolation_mode=upper_extrapolation_mode,
                          upper_extrapolation_param=upper_extrapolation_param)

    assert minval <= y[0] <= maxval
    np.testing.assert_almost_equal(y, [64.54098522160602])

    # hyperbolic extrapolation of 2.5 using   
    # (0.0, cdf(-2.5)) and (50.0, 70.0) -> 97.3100186862884
    minval = 0.0
    maxval = 70.0
    upper_extrapolation_mode = "hyperbolic"
    upper_extrapolation_param = 1.95

    y = backward_interpolation([2.5], raw_table, gaussian_table, minval, maxval,
                          upper_extrapolation_mode=upper_extrapolation_mode,
                          upper_extrapolation_param=upper_extrapolation_param)

    assert minval <= y[0]
    np.testing.assert_almost_equal(y, [97.3100186862884])

    # linear interpolation (but using power 1.0) of [5.0, 60.0] using
    minval = 0.0
    maxval = 70.0
    lower_extrapolation_mode = "power"
    lower_extrapolation_param = 1.0
    upper_extrapolation_mode = "power"
    upper_extrapolation_param = 1.0

    y = backward_interpolation([-2.5, 2.5], raw_table, gaussian_table, minval, maxval,
                          upper_extrapolation_mode=upper_extrapolation_mode,
                          upper_extrapolation_param=upper_extrapolation_param)

    assert minval <= y[0] <= maxval
    assert minval <= y[1] <= maxval
    np.testing.assert_array_almost_equal(y, [2.72950739, 64.54098522160602])
