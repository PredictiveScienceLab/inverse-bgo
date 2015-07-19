"""
A model ensemble built by sampling from the posterior of a model using, presumably,
MCMC.

Author:
    Ilias Bilionis

Date:
    5/1/2015

"""


__all__ = ['ModelEnsemble']


import numpy as np
from GPy import Model
from GPy.inference.mcmc import HMC
from . import expected_improvement


class ModelEnsemble(object):

    """
    A collection of models.

    :param model:       The underlying model.
    :param param_list:  The parameter list representing samples from the posterior
                        of the model. (2D numpy array, rows are samples, columns are
                        parameters)
    :param w:           The weights corresponding to each parameter.
    """

    # The underlying model
    _model = None

    # List of parameters
    _param_list = None

    # Weight corresponding to each model (normalized)
    _w = None

    @property 
    def model(self):
        """
        :getter: Get the model.
        """
        return self._model

    @model.setter
    def model(self, value):
        """
        :setter: The model.
        """
        assert isinstance(value, Model)
        self._model = value

    @property
    def param_list(self):
        """
        :getter: Get the parameter list.
        """
        return self._param_list

    @param_list.setter
    def param_list(self, value):
        """
        :setter: Set the parameter list.
        """
        assert isinstance(value, np.ndarray)
        assert value.ndim == 2
        self._param_list = value

    @property
    def w(self):
        """
        :getter: Get the weights.
        """
        return self._w

    @w.setter
    def w(self, value):
        """
        :setter: Set the weights.
        """
        assert isinstance(value, np.ndarray)
        assert value.ndim == 1
        assert np.all(value >= 0.)
        self._w = value / np.sum(value)

    @property 
    def num_particles(self):
        """
        :getter: Get the number of particles in the ensemble.
        """
        return self.param_list.shape[0]

    def get_model(self, i):
        """
        Get the model with index ``i``.
        """
        self.model.unfixed_param_array[:] = self.param_list[i, :]
        return self.model

    def __init__(self, model, param_list, w=None):
        """
        Initialize the object.
        """
        self.model = model
        self.param_list = param_list
        if w is None:
            w = np.ones(param_list.shape[0])
        self.w = w

    def posterior_mean_samples(self, X):
        """
        Return samples of the posterior mean.
        """
        Y = []
        for i in xrange(self.num_particles):
            model = self.get_model(i)
            y = model.predict(X)[0]
            Y.append(y[:, 0])
        Y = np.array(Y)
        return Y

    def posterior_samples(self, X, size=10):
        """
        Draw samples from the posterior of the ensemble.
        """
        Y = []
        for i in xrange(self.num_particles):
            model = self.get_model(i)
            y = model.posterior_samples(X, size=size).T
            Y.append(y)
        Y = np.vstack(Y)
        idx = np.arange(Y.shape[0])
        return Y[np.random.permutation(idx), :]

    def predict_quantiles(self, X, quantiles=(50, 2.5, 97.5), size=1000):
        """
        Get the predictive quantiles.
        """
        if self.num_particles == 1:
            tmp = self.get_model(0).predict_quantiles(X, quantiles=quantiles)
            return np.array(tmp)[:, :, 0]
        else:
            Y = self.posterior_samples(X, size=size)
        return np.percentile(Y, quantiles, axis=0)

    def raw_predict(self, X):
        """
        Return the prediction of each model at ``X``.
        """
        Y = []
        V = []
        for i in xrange(self.num_particles):
            y, v = self.get_model(i).predict(X)
            Y.append(y)
            V.append(v)
        Y = np.array(Y)
        V = np.array(V)
        return Y, V

    def predict(self, X, **kwargs):
        """
        Predict using the ensemble at ``X``.

        :returns:   tuple containing the media, 0.025 quantile, 0.095 quantile
        """
        return self.predict_quantiles(X)

    def eval_afunc(self, X, func, args=()):
        """
        Evaluate an acquisition function at X using all models.
        """
        res = []
        X_ns = []   # Locations of max/min
        M_ns = []   # Values of max/min
        for i in xrange(self.num_particles):
            r, i_n, m_n = func(X, self.get_model(i), *args)
            res.append(r)
            X_ns.append(i_n)
            M_ns.append(m_n)
        res = np.array(res)
        X_ns = np.array(X_ns)
        M_ns = np.array(M_ns)
        return np.average(res, axis=0, weights=self.w), X_ns, M_ns

    def expected_improvement(self, X, **kwargs):
        """
        Compute the expected improvement.
        """
        return self.eval_afunc(X, expected_improvement, **kwargs)

    @staticmethod
    def train(model, num_samples=0, thin=1, burn=0, num_restarts=10,  **kwargs):
        """
        Train a Gaussian process model.

        :param model:        The model to optimize.
        :param num_restarts: The number of restarts when maximizing the posterior.
        :param num_samples:  The number of samples from the posterior. If zero,
                             then we construct a single particle approximation to
                             the posterior. If greater than zero, then we sample
                             from the posterior of the hyper-parameters using Hybrid MC.
        :param thin:         The number of samples to skip.
        :param burn:         The number of samples to burn.
        :param **kwargs:     Any parameters of GPy.inference.mcmc.HMC

        """
        model.optimize_restarts(num_restarts=num_restarts, verbose=False)
        if num_samples == 0:
            param_list = np.array(model.unfixed_param_array)[None, :]
        else:
            hmc = HMC(model)
            tmp = hmc.sample(num_samples, **kwargs)
            param_list = tmp[burn::thin, :]
        w = np.ones(param_list.shape[0])
        return ModelEnsemble(model, param_list, w=w)