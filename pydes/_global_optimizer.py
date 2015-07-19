"""
A myopic global optimizer class.

Author:
    Ilias Bilionis

Date:
    5/1/2015

"""


__all__ = ['GlobalOptimizer']


import warnings
warnings.filterwarnings('ignore')
import numpy as np
from collections import Iterable
import math
import GPy
import matplotlib.pyplot as plt 
import seaborn
from . import expected_improvement
from . import ModelEnsemble
from . import LogLogisticPrior
from . import JeffreysPrior


class GlobalOptimizer(object):

    """
    A global optimizer class.

    It is essentially a myopic, sequential, global optimizer.

    :param func: The function you wish to optimize.
    :arapm args: Additional arguments for the function we optimize.
    :param afunc: The acquisition function you wish to use.
    :param afunc_args: Extra arguments to the optimization function.

    """

    # The initial design
    _X_init = None

    _X_masked_init = None

    # The initial observations
    _Y_init = None

    # The total design we have available
    _X_design = None

    _X_masked_design = None

    # The indexes of the observations we have made so far (list of integers)
    _idx_X_obs = None

    # The objectives we have observed so far (list of whatever the observations are)
    _Y_obs = None

    # The function we wish to optimize
    _func = None

    # Extra arguments to func
    _args = None

    # The acquisition function we are going to use
    _acquisition_function = None

    # Extra arguments to the acquisition function
    _af_args = None

    @property 
    def X_init(self):
        """
        :getter: Get the initial design.
        """
        return self._X_init

    @X_init.setter
    def X_init(self, value):
        """
        :setter: Set the initial design.
        """
        assert isinstance(value, Iterable)
        self._X_init = value

    @property
    def Y_init(self):
        """
        :getter: Get the initial observations.
        """
        return self._Y_init

    @Y_init.setter
    def Y_init(self, value):
        """
        :setter: Set the initial observations.
        """
        if value is not None:
            assert isinstance(value, Iterable)
            value = np.array(value)
        self._Y_init = value

    @property 
    def X_design(self):
        """
        :getter: Get the design.
        """
        return self._X_design

    @X_design.setter
    def X_design(self, value):
        """
        :setter: Set the design.
        """
        assert isinstance(value, Iterable)
        self._X_design = value

    @property 
    def idx_X_obs(self):
        """
        :getter: The indexes of currently observed design points.
        """
        return self._idx_X_obs

    @property 
    def Y_obs(self):
        """
        :getter: The values of the currently observed design points.
        """
        return self._Y_obs

    @property 
    def func(self):
        """
        :getter: Get the function we are optimizing.
        """
        return self._func

    @func.setter
    def func(self, value):
        """
        :setter: Set the function we are optimizing.
        """
        assert hasattr(value, '__call__')
        self._func = value

    @property
    def args(self):
        """
        :getter: The extra arguments of func.
        """
        return self._args

    @property 
    def acquisition_function(self):
        """
        :getter: Get the acquisition function.
        """
        return self._acquisition_function

    @acquisition_function.setter
    def acquisition_function(self, value):
        """
        :setter: Set the acquisition function.
        """
        assert hasattr(value, '__call__')
        self._acquisition_function = value

    @property 
    def af_args(self):
        """
        :getter: The arguments of the acquisition function.
        """
        return self._af_args

    @property 
    def X(self):
        """
        :getter: Get all the currently observed points.
        """
        return np.vstack([self.X_init, self.X_design[self.idx_X_obs]])

    @property 
    def X_masked(self):
        """
        :getter: Get all the currently observed points.
        """
        return np.vstack([self._X_masked_init, self._X_masked_design[self.idx_X_obs]])

    @property 
    def Y(self):
        """
        :getter: Get all the currently observed objectives.
        """
        if len(self.Y_obs) == 0:
            return np.array(self.Y_init)
        return np.vstack([self.Y_init, self.Y_obs])

    @property 
    def best_value(self):
        """
        :getter: Get the best value.
        """
        return self.current_best_value[-1]

    @property 
    def best_index(self):
        """
        :getter: Get the current best index.
        """
        return self.current_best_index[-1]

    @property 
    def best_design(self):
        i = np.argmin(self.Y)
        return self.X[i, :]

    @property 
    def best_masked_design(self):
        i = np.argmin(self.Y)
        return self.X_masked[i, :]

    def __init__(self, X_init, X_design, func, args=(), Y_init=None,
                 af=expected_improvement, af_args=(),
                 X_masked_init=None,
                 X_masked_design=None):
        """
        Initialize the object.
        """
        self.X_init = X_init
        self.X_design = X_design
        self.Y_init = Y_init
        self.func = func
        self._args = args
        self.acquisition_function = af
        self._af_args = af_args
        self._idx_X_obs = []
        self._Y_obs = []
        self._X_masked_init = X_masked_init
        self._X_masked_design = X_masked_design

    def optimize(self, max_it=100, tol=1e-1, fixed_noise=1e-6,
                 GPModelClass=GPy.models.GPRegression,
                 verbose=True,
                 add_at_least=10,
                 callback_func=None,
                 callback_func_args=(),
                 **kwargs):
        """
        Optimize the objective.

        :param callback_func:       A function that should be called at each iteration.
        :param callback_func_args:  Arguments to the callback function.
        """
        assert add_at_least >= 1
        if self.Y_init is None:
            X = self.X_init if self._X_masked_init is None else self._X_masked_init
            self.Y_init = [self.func(x, *self.args) for x in X]
        self.tol = tol
        self.max_it = max_it
        self.af_values = []
        self.selected_index = []
        self.current_best_value = []
        self.current_best_index = []
        for it in xrange(max_it):
            kernel = GPy.kern.RBF(self.X_init.shape[1], ARD=True)
            model = GPModelClass(self.X, self.Y, kernel)
            self.model = model
            if fixed_noise is not None:
                model.Gaussian_noise.variance.unconstrain()
                model.Gaussian_noise.variance.constrain_fixed(fixed_noise)
                #model.kern.lengthscale.unconstrain()
                #model.kern.lengthscale.fix(.8)
            model.optimize_restarts(20, verbose=False)
            af, i_n, m_n = self.acquisition_function(self.X_design, model, *self.af_args)
            i = np.argmax(af)
            self.af = af
            self.selected_index.append(i)
            self.af_values.append(af[i])
            self.current_best_value.append(self.Y.min())
            if it >= add_at_least and af[i] / self.af_values[0] < tol:
                if verbose:
                    print '*** Converged (af[i] / afmax0 = {0:1.7f}) < {1:e}'.format(af[i] / self.af_values[0], tol)
                break
            self.idx_X_obs.append(i)
            if self._X_masked_design is None:
                self.Y_obs.append(self.func(self.X_design[i], *self.args))
            else:
                self.Y_obs.append(self.func(self._X_masked_design[i], *self.args))
            if verbose:
                print '> Iter {0:d}, Selected struct.: {1:d}, Max EI = {2:1.4f}, Min seen energy: {4:1.3f}'.format(it, i, af[i] / self.af_values[0], self.Y_obs[-1][0], self.Y.min())
            self.current_best_index.append(np.argmin(self.Y))
            if callback_func is not None:
                callback_func(*callback_func_args)