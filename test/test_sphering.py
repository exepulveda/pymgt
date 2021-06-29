import numpy as np

from pymgt import *
from test_utils import *

def test_simple():
    sample_size = 10000
    ndim = 5

    t = SpheringTransform(name='sphering')
    x = np.random.uniform(10.0, 20.0, size=(sample_size, ndim))

    y = t.fit_transform(x)

    assert_reversibility(t, x)

def test_states():
    sample_size = 1000
    ndim = 4

    t = SpheringTransform(name='sphering')
    x = np.random.uniform(-20.0, -10.0, size=(sample_size, ndim))

    y, x_back = assert_reversibility(t, x)

    state = t.state

    # build new object
    t = SpheringTransform(name='sphering')
    t.state = state
    y2 = t.transform(x)
    x_back2 = t.inverse_transform(y2)

    np.testing.assert_array_almost_equal(y, y2)
    np.testing.assert_array_almost_equal(x_back, x_back2)
