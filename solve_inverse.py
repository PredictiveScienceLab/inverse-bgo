"""
Demonstration of how Bayesian Global Optimization can be used to solve
an inverse problem.

Author:
    Ilias Bilionis

Date:
    7/19/2015

"""


from demos.catalysis import CatalysisModelDMNLESS
import numpy as np 
import pydes 
from plots import *
from data import *
import os


def loss_func(x, y, catal_model):
    """
    The loss function we use in the formulation of the inverse problem.
    We want to minimize this.
    """
    d = catal_model(x) - y
    return [np.dot(d, d)]


if __name__ == '__main__':
    # Fix the random seed in order to ensure reproducibility of results:
    np.random.seed(13456)
    # Load the experimental data
    y = load_catalysis_data()
    # Construct the catalysis solver
    # Note that the solver accepts as input the logarithm of 
    # the kinetic coefficients.
    catal_model = CatalysisModelDMNLESS()
    # Number of initial data pool (user)
    n_init = 20
    # Number of candidate test points (user)
    n_design = 10000
    # Maximum iterations for BGO (user)
    max_it = 100
    # Tolerance for BGO (user)
    tol = 1e-4
    # Minimum range for the kinetic coefficients (user)
    kappa_min = 0.2
    # Maximum range for the kinetic coefficients (user)
    kappa_max = 6.
    # Start the algorithm
    print 'SOLVING INVERSE PROBLEMS USING BGO'.center(80)
    print '=' * 80
    print '{0:20s}: {1:d}'.format('Init. pool size', n_init)
    print '{0:20s}: {1:d}'.format('Design. pool size', n_design)
    print '{0:20s}: {1:d}'.format('Max BGO iter.', max_it)
    print '{0:20s}: {1:e}'.format('Tol. of BGO:', tol)
    print '=' * 80
    print '> starting computations'
    # We work with the logarithms of the kinetic coefficients
    log_k_min = np.log(kappa_min)
    log_k_max = np.log(kappa_max)
    # The initial data pool
    X_init = log_k_min + \
             (log_k_max - log_k_min) * np.random.rand(n_init, catal_model.num_input)
    # The design pool
    X_design = log_k_min + \
             (log_k_max - log_k_min) * np.random.rand(n_design, catal_model.num_input)
    # Initialize the Bayesian Global Optimization
    bgo = pydes.GlobalOptimizer(X_init, X_design, loss_func,
                                args=(y, catal_model))
    print '> starting BGO'
    try:
        bgo.optimize(max_it=max_it, tol=tol)
    except KeyboardInterrupt:
        print '> keyboard interruption'
    print '> plotting the results'
    # Make the plots (set ``to_file`` to ``False`` for interactive plots)
    # It writes ``ei.png``, and ``loss.png``.
    # and puts them in the ``results`` directory.
    make_plots(bgo, to_file=True)
    init_log_kappa = X_init[np.argmin(bgo.Y_init), :]
    plot_catalysis_output(os.path.join('results', 'init_fit.png'),
                          catal_model(init_log_kappa), title='Initial fit')
    final_log_kappa = bgo.best_design
    plot_catalysis_output(os.path.join('results', 'final_fit.png'),
                          catal_model(final_log_kappa), title='Final fit')
