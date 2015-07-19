"""
Some functions that make useful plots.

Author:
    Ilias Bilionis

Date:
    7/19/2015

"""

import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import itertools
import os
from data import *


__all__ = ['plot_catalysis_output', 'make_plots', 'plot_1d_callback']


def plot_catalysis_output(fig_name,
                          Y,
                          y=load_catalysis_data(),
                          t=np.array([0.0, 1./6, 1./3, 1./2, 2./3, 5./6, 1.]),
                          colors=['b', 'r', 'g', 'k', 'm'],
                          linestyles=['', '--', '-.', '--+', ':'],
                          markerstyles=['o', 'v', 's', 'D', 'p'],
                          legend=False,
                          title=None):
    """
    Draw the output of the catalysis problem.
    
    :param fig_name:    A name for the figure.
    :param Y:           The samples observed.
    :param y:           The observations.
    :parma t:           The times of observations.
    """
    shape = (t.shape[0], Y.shape[0] / t.shape[0])
    y = y.reshape(shape)
    Y = Y.reshape(shape)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in xrange(5):
        ax.plot(t, Y[:, i], colors[i] + linestyles[i], linewidth=2, markersize=5.)
    for i in xrange(5):
        ax.plot(t, y[:, i], colors[i] + markerstyles[i], markersize=10)
    ax.set_xlabel('Time ($\\tau$)', fontsize=26)
    ax.set_ylabel('Concentration', fontsize=26)
    ax.set_title(title, fontsize=26)
    plt.setp(ax.get_xticklabels(), fontsize=26)
    plt.setp(ax.get_yticklabels(), fontsize=26)
    if legend:
        leg = plt.legend(['$\operatorname{NO}_3^-$', '$\operatorname{NO}_2^-$',
                          '$\operatorname{N}_2$', '$\operatorname{N}_2\operatorname{O}$',
                          '$\operatorname{NH}_3$'], loc='best')
        plt.setp(leg.get_texts(), fontsize=26)
    ax.set_ylim([0., 1.5])
    plt.tight_layout()
    if fig_name:
        print '> writing:', fig_name
        plt.savefig(fig_name)
    else:
        plt.show()


def make_plots(bgo, to_file=False):
    """
    Makes the demonstration plots.
    """
    make_ei_plot(bgo, to_file)
    make_energy_plot(bgo, to_file)
    if not to_file:
        plt.show()

def make_ei_plot(bgo, to_file):
    """
    Plots the evolution of the expected improvement as BGO runs.
    """
    fig, ax = plt.subplots()
    it = np.arange(1, len(bgo.af_values) + 1)
    ax.plot(it, bgo.af_values)
    ax.plot(it, [bgo.tol] * len(it), '--')
    ax.set_title('Evolution of expected improvement', fontsize=16)
    ax.set_xlabel('Iteration', fontsize=16)
    ax.set_ylabel('EI', fontsize=16)
    if to_file:
        figname = os.path.join('results', 'ei.png')
        print '> writing:', figname
        plt.savefig(figname)
        plt.close(fig)


def make_energy_plot(bgo, to_file):
    """
    Plots the evolution of the energy as BGO runs.
    """
    fig, ax = plt.subplots()
    it = np.arange(1, len(bgo.current_best_value) + 1)
    ax.plot(it, bgo.current_best_value)
    ax.set_title('Evolution of minimum observed loss', fontsize=16)
    ax.set_xlabel('Iteration', fontsize=16)
    ax.set_ylabel('Loss', fontsize=16)
    if to_file:
        figname = os.path.join('results', 'loss.png')
        print '> writing:', figname
        plt.savefig(figname)
        plt.close(fig)


__count_callback = 0
def plot_1d_callback(bgo, molecule, interactive):
    """
    Plots the evolution of BGO for the 1D case.
    """
    global __count_callback
    __count_callback += 1
    fig, ax = plt.subplots()
    ax.set_ylabel('$V(r)$', fontsize=16)
    ax.set_xlabel('$r$', fontsize=16)
    ax.set_ylim([0, 20])
    ax2 = ax.twinx()
    ax2.set_ylim([0, 1.5])
    ax2.set_ylabel('EI$(r)$', color='g', fontsize=16)
    for tl in ax2.get_yticklabels():
        tl.set_color('g')
    xx = np.linspace(bgo.X_design.min(), bgo.X_design.max(), 100)
    q = bgo.model.predict_quantiles(xx[:, None], quantiles=(50, 2.5, 97.5))
    ax.plot(bgo.X, bgo.Y, 'kx', markersize=10, markeredgewidth=2)
    ax.plot(bgo.X[-1], bgo.Y[-1], 'go', markersize=10, markeredgewidth=2)
    ax.plot(xx, q[0], 'b', label='Mean prediction')
    ax.fill_between(xx, q[1].flatten(), q[2].flatten(), color='blue', alpha=0.25)
    ax2.plot(bgo.X_design, bgo.af / bgo.af_values[0], 'g.')
    if interactive:
        plt.show(block=True)
    else:
        figname = os.path.join('results', 'bgo_' + 
                                          molecule.get_chemical_formula()+ '_' 
                                          + str(__count_callback).zfill(2) 
                                          + '.png')
        print '> writing:', figname
        plt.savefig(figname)
