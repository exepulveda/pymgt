import warnings
import numpy as np
import sklearn.decomposition

from typing import List, Dict, Optional
from .transform import Transform
from .nscores import MarginalGaussianTransform
from .nscores import MarginalGaussianState
from .sphering import SpheringState
from .metrics import FRIEDMAN_METRIC
from .interface import AbstractState
from .interface import Vector
from .interface import Array2D
from .ppmt_utils import friedman_index

from scipy.stats import multivariate_normal

class RBIGStepState(AbstractState):
    """The state of a RBIG step
    """
    def __init__(self, rotation: Dict, marginal: MarginalGaussianState):
        self.rotation = rotation
        self.marginal = marginal


class RBIGState(AbstractState):
    """The state of a RBIG transform
    """
    def __init__(self, marginal: MarginalGaussianState, 
                       iteration_steps: List[RBIGStepState]):
        self.marginal = marginal
        self.iteration_steps = iteration_steps

class RBIGTransform(Transform):
    """Rotation Based Iterative Transform: This transform applies at each
    iteration a rotation transform followed by a marginal Gaussian transform.
    Before the iterations, applies (optionally) a marginal Gaussian transform to allow
    using declustering weights and Sphering.

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
    rotation : string
        Name of the rotation to apply. It can be 'random', 'pca', 'ica', or 'pp' to apply
        a random, principal component analysis, independent component analysis or projection pursuit
        rotation respectively.
    state : 2d-numpy array of shape (N,ndim+1) where N is the number of samples used
        to fit the transform. If tails are given, N can have additional extremes.

    References
    ----------
    Laparra, V., Camps-Valls, G., & Malo, J. (2011). Iterative gaussianization: From ICA to random rotations.
    IEEE Transactions on Neural Networks, 22(4), 537–549. https://doi.org/10.1109/TNN.2011.2106511

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

    def __init__(self, objective=None, target=None, maxiter=100, metrics=None, **kargs):
        super(RBIGTransform, self).__init__(metrics=metrics, **kargs)

        self.__objective = FRIEDMAN_METRIC if objective is None else objective

        # define the iteration sequence which is (a) Rotation followed by (b) marginal Gaussianisation
        self._rot = sklearn.decomposition.FastICA(whiten=True, **kargs)
        self._mvgt = MarginalGaussianTransform()
        self.maxiter = maxiter
        self.target = target

        self._apply_first_marginal = True
        self._apply_iterations = True
        self._apply_internal_rot = True

    def _get_rot_state(self) -> Dict:
        return self._rot.__getstate__()

    def _set_rot_state(self, rot_state: Dict):
        self._rot.__setstate__(rot_state)

    def fit_transform(self, x: Array2D, weights: Optional[Vector]=None) -> Array2D:
        ndata, ndim = x.shape

        y = x.copy()

        state1 = None
        state2_steps = None

        self._mdmetric.init_directions(y, force=True)

        warnings.simplefilter("ignore")

        imvg = multivariate_normal.rvs(mean=np.zeros(ndim), cov=np.eye(ndim), size=ndata)
        s = self.metrics_text(imvg, extra=self.__objective)
        print(f"Metrics applied to a standard {ndim}-dimensional Gaussian distribution of size={ndata}:\n{s}")

        if self.target is None:
            self.target = self._mdmetric(x)

        print(f"Stopping at {self.maxiter} or target={self.target}")

        # step 1: Marginal Gaussianisation
        if self._apply_first_marginal:
            y = self._mvgt.fit_transform(y, weights=weights)
            state1 = self._mvgt.state

            if self.tracing:
                for dim in range(ndim):
                    print("%d marginal pi=%f"%(dim, friedman_index(y[:, dim])))

            p, test_best = self._mdmetric.compute_test_best(y, self.__objective, self.target)

            s = self.metrics_text(y, extra=self.__objective)
            if len(s) > 0: print("Iteration[0] metrics: %s"%s)

        # step 2: iterative rotation + marginal
        state2_steps = []
        if self._apply_iterations:
            for i in range(self.maxiter):
                # rotate
                if self._apply_internal_rot:
                    y = self._rot.fit_transform(y)
                    rot_state = self._get_rot_state()
                else:
                    rot_state = None

                # marginal gaussianisation
                y = self._mvgt.fit_transform(y)

                state2_steps += [RBIGStepState(rot_state, self._mvgt.state)]

                p, test_best = self._mdmetric.compute_test_best(y.copy(), self.__objective, self.target)

                s = self.metrics_text(y, extra=self.__objective)
                if len(s) > 0: print("Iteration[%d] metrics: %s"%(i+1, s))

                # check for termination
                if test_best:
                    break

        self.state = RBIGState(state1, state2_steps)

        self._fitted = True

        return y

    def transform(self, x: Array2D) -> Array2D:
        assert self._fitted

        y = np.copy(x)

        # step 1: Marginal Gaussianisation
        if self._apply_first_marginal:
            self._mvgt.state = self.state.marginal
            y = self._mvgt.transform(y)

        # step 2: iterative rotation + marginal
        if self._apply_iterations:
            for _, state2_step in enumerate(self.state.iteration_steps):
                if self._apply_internal_rot:
                    self._set_rot_state(state2_step.rotation)
                    y = self._rot.transform(y)

                # marginal gaussianisation
                self._mvgt.state = state2_step.marginal
                y = self._mvgt.transform(y)

        return y

    def inverse_transform(self, y: Array2D) -> Array2D:
        assert self._fitted

        x = np.copy(y)

        # step 2: iterative RBIG step
        if self._apply_iterations:
            for _, state2_step in enumerate(reversed(self.state.iteration_steps)):
                self._mvgt.state = state2_step.marginal
                x = self._mvgt.inverse_transform(x)

                if self._apply_internal_rot:
                    self._set_rot_state(state2_step.rotation)
                    x = self._rot.inverse_transform(x)

        # step 1: Marginal Gaussianisation
        if self._apply_first_marginal:
            self._mvgt.state = self.state.marginal
            x = self._mvgt.inverse_transform(x)

        return x
