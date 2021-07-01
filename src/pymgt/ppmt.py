"""Projection Pursuit Multivariate Transform using moments as projection index
"""

import numpy as np
from scipy.stats import multivariate_normal

from .transform import Transform
from .nscores import UnivariateGaussianTransform
from .nscores import MarginalGaussianTransform
from .sphering import SpheringTransform
from .utils import orthonormal_basis
from .ppmt_utils import friedman_index
from .ppmt_utils import find_next_best_direction_gd
from .ppmt_utils import find_next_best_direction
from .metrics import FRIEDMAN_METRIC


class PPMTransform(Transform):
    """The original projection pursuit mutivariate transform (as closer as possible) with some
    improvements, such as a robust optimisation method, displaying metrics and serialisation.

    Paper:
    Barnett, R. M., Manchuk, J. G., & Deutsch, C. V. (2016).
    The Projection-Pursuit Multivariate Transform for Improved Continuous Variable Modeling.
    Society of Petroleum Engineers, 1(December). https://doi.org/10.2118/184388-PA
    """
    def __init__(self, target=None, optimiser="original", maxiter=100, metrics=None, **kargs):
        super(PPMTransform, self).__init__(metrics=metrics, **kargs)

        assert optimiser in ("original", "de")

        self._opt = optimiser
        self._ugt = UnivariateGaussianTransform()
        self._mvgt = MarginalGaussianTransform()
        self._sph = SpheringTransform()
        self._state = None
        self.maxiter = maxiter
        self.target = target

        self._apply_first_marginal = True
        self._apply_sphering = True
        self._apply_iterations = True
        self._apply_last_marginal = True

    def _set_state(self, state):
        self.__state = state

    def _get_state(self):
        return self.__state

    def fit_transform(self, x, weights=None):
        ndata, ndim = x.shape

        y = x.copy()

        state1 = None
        state2 = None
        gaussian_table = None
        state3_steps = None
        state4 = None

        # step 1: Marginal Gaussianisation
        if self._apply_first_marginal:
            y = self._mvgt.fit_transform(y, weights=weights)
            state1 = self._mvgt.state

            if self.tracing:
                for dim in range(ndim):
                    print("%d marginal pi=%f"%(dim, friedman_index(y[:, dim])))

        # step 2: sphering
        if self._apply_sphering:
            y = self._sph.fit_transform(y)
            state2 = self._sph.state

            if self.tracing:
                for dim in range(ndim):
                    print("%d marginal pi=%f"%(dim, friedman_index(y[:, dim])))

        # step 3: iterative PP step
        imvg = multivariate_normal.rvs(mean=np.zeros(ndim), cov=np.eye(ndim), size=ndata)
        s = self.metrics_text(imvg)
        print(f"Metrics applied to a standard {ndim}-dimensional Gaussian distribution of size={ndata}:\n{s}")

        if self.target is None:
            self.target = self._mdmetric(x, FRIEDMAN_METRIC)

        print(f"Stopping at {self.maxiter} or target={self.target}")

        gaussian_table = None
        state3_steps = []
        if self._apply_iterations:
            for i in range(self.maxiter):
                y, direction, pi, uvgstate = self._step_fit_transform(y, gaussian_table, False)
                rtable, gtable = uvgstate.raw_table, uvgstate.gaussian_table
                if gaussian_table is None:
                    gaussian_table = gtable

                state3_steps += [(direction, rtable)]

                if self.tracing:
                    for dim in range(ndim):
                        print("%d marginal pi=%f"%(dim, friedman_index(y[:, dim])))

                s = self.metrics_text(y)
                if len(s) > 0:
                    print(f"Iteration[{i+1}] index={pi} - metrics: {s}")
                else:
                    print(f"Iteration[{i+1}] index={pi}")

                # check for termination
                if pi < self.target:
                    break

        # step 4: last marginal Gaussianisation no weights
        if self._apply_last_marginal:
            y = self._mvgt.fit_transform(y)
            state4 = self._mvgt.state

        self._state = (state1, state2, gaussian_table, state3_steps, state4)

        return y

    def fit(self, x, y=None):
        self.fit_transform(x)

    def transform(self, x):
        state1, state2, gaussian_table, state3_steps, state4 = self._state

        # step 1: Marginal Gaussianisation
        self._mvgt.state = state1
        y = self._mvgt.transform(x)

        # step 2: sphering
        self._sph.state = state2
        y = self._sph.transform(y)

        # step 3: iterative PP step
        for state3_step in state3_steps:
            direction, raw_table = state3_step

            y = self._step_transform(y, direction, raw_table, gaussian_table)

        # step 4: last marginal Gaussianisation no weights
        self._mvgt.state = state4
        y = self._mvgt.transform(y)

        return y

    def inverse_transform(self, y):
        x = np.copy(y)

        state1, state2, gaussian_table, state3_steps, state4 = self._state

        # step 4: last marginal Gaussianisation no weights
        if self._apply_last_marginal:
            self._mvgt.state = state4
            x = self._mvgt.inverse_transform(x)

        # step 3: iterative PP step
        if self._apply_iterations:
            for _, state3_step in enumerate(reversed(state3_steps)):
                direction, raw_table = state3_step
                x = self._step_inverse_transform(x, direction, raw_table, gaussian_table)

        # step 2: sphering
        if self._apply_sphering:
            self._sph.state = state2
            x = self._sph.inverse_transform(x)

        # step 1: Marginal Gaussianisation
        if self._apply_first_marginal:
            self._mvgt.state = state1
            x = self._mvgt.inverse_transform(x)

        return x

    def _step_fit_transform(self, x, gaussian_table, trace=False):
        _, ndim = x.shape
        cls_name = self.__class__.__name__

        if trace: print("fit_transform at [%s]: x.shape=%s"%(cls_name, str(x.shape)))

        # find the best next direction
        if self._opt == "original":
            direction, pi = find_next_best_direction_gd(x, trace=trace) #, friedman_index)
        else:
            direction, pi = find_next_best_direction(x, friedman_index)

        if trace: print("fit_transform at [%s]: pi=%f"%(cls_name, pi))

        # set the rotation matrix
        Q = orthonormal_basis(direction)
        Q_inv = np.linalg.inv(Q)

        # Rotate data
        z = x@Q

        # normalizing the first projection
        # by building the bijection relationship with gaussian_table
        xp = z[:, 0]

        nscores = self._ugt.fit_transform(xp, gaussian_table=gaussian_table)

        if trace: print("nscores after Guassianisation pi=%f"%friedman_index(nscores))

        # set nscores to the first projection
        z[:, 0] = nscores

        if trace:
            for dim in range(ndim):
                print("after nscore %d marginal pi=%f"%(dim, friedman_index(z[:, dim])))

        # rotate back
        y = z@Q_inv

        if trace:
            for dim in range(ndim):
                print("after rotation %d marginal pi=%f"%(dim, friedman_index(y[:, dim])))

        return y, direction, pi, self._ugt.state

    def _step_transform(self, x, direction, raw_table, gaussian_table):
        # set the rotation matrix
        Q = orthonormal_basis(direction)
        Q_inv = np.linalg.inv(Q)

        # Rotate data
        z = x@Q

        # normalizing the first projection
        # by building the bijection relationship with gaussian_table
        xp = z[:, 0]

        self._ugt.state.raw_table = raw_table
        self._ugt.state.gaussian_table = gaussian_table

        nscores = self._ugt.transform(xp)

        # set nscores to the first projection
        z[:, 0] = nscores

        # rotate back
        y = z@Q_inv

        return y

    def _step_inverse_transform(self, y, direction, raw_table, gaussian_table):
        # set the rotation matrix
        Q = orthonormal_basis(direction)
        Q_inv = np.linalg.inv(Q)

        # Rotate data
        z = y@Q

        # back transform the first projection
        # by building the bijection relationship with raw_table
        yp = z[:, 0]

        self._ugt.state.raw_table = raw_table
        self._ugt.state.gaussian_table = gaussian_table

        rawscores = self._ugt.inverse_transform(yp)

        #print(np.min(rawscores), np.mean(rawscores), np.max(rawscores))

        # set nscores to the first projection
        z[:, 0] = rawscores

        # rotate back
        x = z@Q_inv

        return x
