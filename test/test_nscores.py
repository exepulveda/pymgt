import numpy as np
import scipy

from pymgt import *
from pymgt.nscores import univariate_nscore
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
