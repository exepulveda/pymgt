import numpy as np

from pymgt import *
from test_utils import *

def test_normalscore_noweights():
    decimals = 2

    t = UnivariateGaussianTransform(name='nscore')
    x = np.random.uniform(10.0, 20.0, size=10000)

    y = t.fit_transform(x)

    assert_normality(y)
    assert_reversibility(t, x)

def test_normalscore_weights():
    decimals = 5
    sample_size = 10000

    weights = np.random.uniform(0.5, 0.8, size=sample_size)
    weights = weights / np.sum(weights) * sample_size

    t = UnivariateGaussianTransform(name='nscore')
    x = np.random.uniform(10.0, 20.0, size=sample_size)

    y = t.fit_transform(x, weights=weights)

    assert_normality(y)
    assert_reversibility(t, x)

def test_states():
    sample_size = 1000
    ndim = 4

    t = UnivariateGaussianTransform(name='marginal_nscore', trace=True)
    x = np.random.uniform(10.0, 20.0, size=sample_size)

    y, x_back = assert_reversibility(t, x)

    state = t.state

    # build new object
    t = UnivariateGaussianTransform(name='marginal_nscore', trace=True)
    t.state = state
    y2 = t.transform(x)
    x_back2 = t.inverse_transform(y2)

    np.testing.assert_array_almost_equal(y, y2)
    np.testing.assert_array_almost_equal(x_back, x_back2)

def test_marginalnormalscore():
    decimals = 2
    sample_size = 1000

    for ndim in range(2, 10):
        t = MarginalGaussianTransform(name='marginal_nscore', trace=True)
        x = np.random.uniform(10.0, 20.0, size=(sample_size, ndim))

        y, x_back = assert_reversibility(t, x)

        for dim in range(ndim):
            #plt.hist(x_back[:, dim], 39)
            #plt.show()
            assert_normality(y[:, dim])

def test_states():
    sample_size = 1000
    ndim = 4

    t = MarginalGaussianTransform(name='marginal_nscore', trace=True)
    x = np.random.uniform(10.0, 20.0, size=(sample_size, ndim))

    y, x_back = assert_reversibility(t, x)

    state = t.state

    # build new object
    t = MarginalGaussianTransform(name='marginal_nscore', trace=True)
    t.state = state
    y2 = t.transform(x)
    x_back2 = t.inverse_transform(y2)

    np.testing.assert_array_almost_equal(y, y2)
    np.testing.assert_array_almost_equal(x_back, x_back2)
