import numpy as np

from pymgt.utils import spherical2cartesian
from pymgt.utils import cartesian2spherical


def test_cartesian_spherical():
    for ndim in range(2, 50):
        for _ in range(100):
            angles = np.random.uniform(low=0.0, high=np.pi, size=ndim-1)
            angles[-1] *= 2.0

            # convert to cartesian
            direction = spherical2cartesian(angles)
            # convert back to spherical
            angles_back = cartesian2spherical(direction)

            np.testing.assert_array_almost_equal(angles, angles_back)

