"""
Forward problem (dimensionless)

Author:
    Panagiotis Tsilifis
Date:
    06/09/2014

"""


import numpy as np
from model_1 import *
from model_2 import *
from _utils import *
import sys



class CatalysisModelDMNLESS():
    """
    A class representing the forward model of the catalysis problem.
    """

    def __init__(self):
        """
        Initialize the object
        """
        self.num_input = 5
        self.num_output = 35

    def __call__(self, z):
        """
        Solves the dynamical system for given parameters x.
        
        """
        z = view_as_column(z)
        x = np.exp(z)
        # Points where the solution will be evaluated
        t = np.array([0., 1./6, 1./3, 1./2, 2./3, 5./6, 1.])
        t = view_as_column(t)
        # Initial condition
        y0 = np.array([1., 0., 0., 0., 0., 0.])
        y0 = view_as_column(y0)
        assert x.shape[0] == 5
        sol = f(x[:,0], y0[:,0], t[:,0])
        J = df(x[:,0], y0[:,0], t[:,0])
        H = df2(x[:,0], y0[:,0], t[:,0])
        y = np.delete(sol.reshape((7,6)), 2, 1).flatten() # The 3rd species is unobservable
        dy = np.array([np.delete(J[:,i].reshape((7,6)), 2, 1).reshape(35) for i in range(J.shape[1])]) # Delete the 3rd species
        d2y = np.zeros((35, H.shape[1], H.shape[2]))
        for i in range(H.shape[1]):
            for j in range(H.shape[2]):
                d2y[:,i,j]= np.delete(H[:,i,j].reshape((7,6)), 2, 1).reshape(35) # Delete the 3rd species
        return y
