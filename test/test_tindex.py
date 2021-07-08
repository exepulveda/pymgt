import numpy as np

from pymgt.tindex import generate_directions
from pymgt.tindex import projection_index
from pymgt.tindex import Projectable
from pymgt.tindex import jarque_bera_index
from pymgt.tindex import shapiro_index
from pymgt.tindex import anderson_index
from pymgt.tindex import ks_index

def test_generate_directions():   
    for dim in range(2, 11):
        dirs = generate_directions(dim, n=100)
        for d in dirs:
            np.testing.assert_almost_equal(np.linalg.norm(d), 1.0)

def test_projection_index():
    ndata, ndim = 1000, 5
    x = np.random.uniform(10.0, 20.0, size=(ndata, ndim))    
    pi = projection_index(x, lambda p: 1.0, nprojections=100, reduce = np.mean)

    assert pi == 1.0

def test_projectable():
    ndata, ndim = 1000, 5

    p = Projectable(lambda p: 2.0, nprojections=100, reduce = np.mean)

    x = np.random.uniform(0.0, 1.0, size=(ndata, ndim))    
    pi = p(x)

    assert pi == 2.0

def test_indices():
    ndata = 100000

    np.random.seed(1)

    for _ in range(10):
        x = np.random.normal(0.0, 1.0, size=ndata)

        pi = jarque_bera_index(x)
        assert 0.0 <= pi <= 3.0

        pi = shapiro_index(x)
        assert 0.99 <= pi <= 1.0

        pi = anderson_index(x)
        assert 0.0 <= pi <= 1.0

        pi = ks_index(x)
        assert 0.0 <= pi <= 0.01
