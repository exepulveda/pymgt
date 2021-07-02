"""Projection Pursuit Multivariate Transform using moments as projection index
"""

import numpy as np
from scipy.stats import multivariate_normal

from .transform import Transform
from .nscores import univariate_nscore
from .nscores import MarginalGaussianTransform
from .sphering import SpheringTransform
from .utils import orthonormal_basis
from .ppmt_utils import friedman_index
from .ppmt_utils import find_next_best_direction_gd
from .ppmt_utils import find_next_best_direction
from .metrics import FRIEDMAN_METRIC

from typing import List
from .nscores import UnivariateGaussianState
from .nscores import MarginalGaussianState
from .sphering import SpheringState
from .interface import Vector
from .interface import Array2D
from .interface import AbstractState

class PPMTStepState(AbstractState):
    """The state of a PPMT step
    """
    def __init__(self, direction: Vector, raw_table: Vector):
        self.direction = direction
        self.raw_table = raw_table


class PPMTStep(AbstractState):
    """The state of a PPMT transform
    """
    def __init__(self, state_initial_marginal: MarginalGaussianState, 
                       state_sphering: SpheringState,
                       gaussian_table: Vector,
                       iteration_steps: List[PPMTStepState],
                       state_final_marginal: MarginalGaussianState):
        self.state_initial_marginal = state_initial_marginal
        self.state_sphering = state_sphering
        self.gaussian_table = gaussian_table
        self.iteration_steps = iteration_steps
        self.state_final_marginal = state_final_marginal


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
        self._mvgt = MarginalGaussianTransform()
        self._sph = SpheringTransform()
        self.maxiter = maxiter
        self.target = target

        self._apply_first_marginal = True
        self._apply_sphering = True
        self._apply_iterations = True
        self._apply_last_marginal = True

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
            # the first gaussianisation does not use weights
            y = self._mvgt.fit_transform(y)
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
                y, direction, pi, raw_table, gaussian_table = self._step_fit_transform(y, gaussian_table, False)

                assert np.all(np.diff(raw_table) >= 0)
                assert np.all(np.diff(gaussian_table) >= 0)

                #print("raw_table stats: ", np.min(raw_table), np.mean(raw_table), np.max(raw_table))
                #print("gaussian_table stats: ", np.min(gaussian_table), np.mean(gaussian_table), np.max(gaussian_table))
                state3_steps += [PPMTStepState(direction, raw_table)]

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
            # this final, uses weights
            y = self._mvgt.fit_transform(y, weights=weights)
            state4 = self._mvgt.state

        self._state = PPMTStep(state1, state2, gaussian_table, state3_steps, state4)

        return y

    def fit(self, x: Array2D, y=None):
        self.fit_transform(x)

    def transform(self, x: Array2D) -> Array2D:
        state1 = self._state.state_initial_marginal
        state2 = self._state.state_sphering
        gaussian_table = self._state.gaussian_table
        state3_steps = self._state.iteration_steps
        state4 = self._state.state_final_marginal

        # step 1: Marginal Gaussianisation
        self._mvgt.state = state1
        y = self._mvgt.transform(x)

        # step 2: sphering
        self._sph.state = state2
        y = self._sph.transform(y)

        # step 3: iterative PP step
        for state3_step in state3_steps:
            direction = state3_step.direction
            raw_table = state3_step.raw_table

            y = self._step_transform(y, direction, raw_table, gaussian_table)

        # step 4: last marginal Gaussianisation no weights
        self._mvgt.state = state4
        y = self._mvgt.transform(y)

        return y

    def inverse_transform(self, y: Array2D) -> Array2D:
        x = np.copy(y)

        state1 = self._state.state_initial_marginal
        state2 = self._state.state_sphering
        gaussian_table = self._state.gaussian_table
        state3_steps = self._state.iteration_steps
        state4 = self._state.state_final_marginal

        # step 4: last marginal Gaussianisation no weights
        if self._apply_last_marginal:
            self._mvgt.state = state4
            x = self._mvgt.inverse_transform(x)

        # step 3: iterative PP step
        if self._apply_iterations:
            for _, state3_step in enumerate(reversed(state3_steps)):
                x = self._step_inverse_transform(x, state3_step.direction, state3_step.raw_table, gaussian_table)

        # step 2: sphering
        if self._apply_sphering:
            self._sph.state = state2
            x = self._sph.inverse_transform(x)

        # step 1: Marginal Gaussianisation
        if self._apply_first_marginal:
            self._mvgt.state = state1
            x = self._mvgt.inverse_transform(x)

        return x

    def _step_fit_transform(self, x: Array2D, gaussian_table: Vector, trace: bool=False):
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

        if gaussian_table is None:
            _, (_, gaussian_table, _) = univariate_nscore(xp)

        # the original implementation just sort the projection to match the gaussian table
        xp_sorted_indices = np.argsort(xp)
        xp_sorted = xp[xp_sorted_indices]

        z[xp_sorted_indices, 0] = gaussian_table

        if trace:
            for dim in range(ndim):
                print("after nscore %d marginal pi=%f"%(dim, friedman_index(z[:, dim])))

        # rotate back
        y = z@Q_inv

        if trace:
            for dim in range(ndim):
                print("after rotation %d marginal pi=%f"%(dim, friedman_index(y[:, dim])))

        return y, direction, pi, xp_sorted, gaussian_table

    def _step_transform(self, x, direction, raw_table, gaussian_table):
        # set the rotation matrix
        Q = orthonormal_basis(direction)
        Q_inv = np.linalg.inv(Q)

        # Rotate data
        z = x@Q

        # normalizing the first projection
        # by building the bijection relationship with gaussian_table
        xp = z[:, 0]

        #self._ugt.state = UnivariateGaussianState(raw_table, gaussian_table)

        nscores = np.interp(xp, raw_table, gaussian_table)


        # nscores = self._ugt.transform(xp)

        # set nscores to the first projection
        z[:, 0] = nscores

        # rotate back
        y = z@Q_inv

        return y

    def _step_inverse_transform(self, y: Array2D, direction: Vector, raw_table: Vector, gaussian_table: Vector):
        # set the rotation matrix
        Q = orthonormal_basis(direction)
        Q_inv = np.linalg.inv(Q)

        # Rotate data
        z = y@Q

        # back transform the first projection
        # by building the bijection relationship with raw_table
        yp = z[:, 0]

        # self._ugt.state = UnivariateGaussianState(raw_table, gaussian_table)
        rawscores = np.interp(yp, gaussian_table, raw_table)

        #rawscores = self._ugt.inverse_transform(yp)

        #print(np.min(rawscores), np.mean(rawscores), np.max(rawscores))

        # set nscores to the first projection
        z[:, 0] = rawscores

        # rotate back
        x = z@Q_inv

        return x
