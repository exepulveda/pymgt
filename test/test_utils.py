"""Utilities assertion for the package
"""
import numpy as np
import scipy.stats

def assert_reversibility(transform, x, decimal=6):
    x = np.asarray(x)
    y = transform.fit_transform(x)
    x_back = transform.inverse_transform(y)

    assert x.shape == y.shape, "must have the same shape"
    assert x.shape == x_back.shape, "must have the same shape"

    # check reversibility
    np.testing.assert_array_almost_equal(x, x_back, decimal=decimal)

    return y, x_back

def assert_array_sorted(x):
    assert np.all(np.diff(x) >= 0)

def assert_normality(y, alpha=0.05):
    _, pvalue = scipy.stats.kstest(y, "norm")
    assert pvalue >= alpha
