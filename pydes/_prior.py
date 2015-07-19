"""
Some priors for GPy.

Author:
    Ilias Bilionis

Date:
    5/5/2015

"""


__all__ = ['LogLogisticPrior', 'JeffreysPrior']


import GPy
import numpy as np


class LogLogisticPrior(GPy.priors.Prior):

    """
    Log-Logistic prior suitable for lengthscale parameters.
    
    From Conti & O'Hagan (2010)
    """

    domain = GPy.priors._POSITIVE

    def __init__(self):
        """
        Initialize the object.
        """
        pass

    def __str__(self):
        return 'LogLog'

    def lnpdf(self, x):
        return -np.log(1. + x ** 2)

    def lnpdf_grad(self, x):
        return -2. * x / (1. + x ** 2)

    def rvs(self, n):
        return np.exp(np.random.logistic(size=n))


class JeffreysPrior(GPy.priors.Prior):

    """
    The uninformative Jeffrey's prior used for scale parameters.
    """

    domain = GPy.priors._POSITIVE

    def __init__(self):
        """
        Initialize the object.
        """
        pass

    def __str__(self):
        return 'JeffreysPrior()'

    def lnpdf(self, x):
        return -np.log(x)

    def lnpdf_grad(self, x):
        return -1. / x

    def rvs(self, n):
        return np.ones(n)