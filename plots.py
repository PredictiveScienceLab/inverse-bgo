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


__all__ = ['make_plots', 'plot_1d_callback']


def make_plots(bgo, molecule, to_file=False):
    """
    Makes the demonstration plots.
    """
    make_ei_plot(bgo, molecule, to_file)
    make_energy_plot(bgo, molecule, to_file)
    make_cluster_plot(bgo, molecule, to_file)
    if not to_file:
        plt.show()

def make_ei_plot(bgo, molecule, to_file):
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
        figname = os.path.join('results', 'ei_' + molecule.get_chemical_formula() + '.png')
        print '> writing:', figname
        plt.savefig(figname)
        plt.close(fig)


def make_energy_plot(bgo, molecule, to_file):
    """
    Plots the evolution of the energy as BGO runs.
    """
    fig, ax = plt.subplots()
    it = np.arange(1, len(bgo.current_best_value) + 1)
    ax.plot(it, bgo.current_best_value)
    ax.set_title('Evolution of minimum observed energy', fontsize=16)
    ax.set_xlabel('Iteration', fontsize=16)
    ax.set_ylabel('Energy', fontsize=16)
    if to_file:
        figname = os.path.join('results', 'energy_' + molecule.get_chemical_formula() + '.png')
        print '> writing:', figname
        plt.savefig(figname)
        plt.close(fig)


def draw_sphere(ax, center, radius=0.2, color='r'):
    """
    Draw a sphere centered at ``center`` with radius ``radius``.
    """
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = radius * np.cos(u) * np.sin(v) + center[0]
    y = radius * np.sin(u) * np.sin(v) + center[1]
    z = radius * np.cos(v) + center[2]
    ax.plot_surface(x, y, z, color=color, rstride=1, cstride=1,
        linewidth=0)


def make_cluster_plot(bgo, molecule, to_file):
    """
    Plots minimum energy cluster fond by BGO.
    """
    CPK_COLORS = {'H': 'w',
                  'N': 'b',
                  'O': 'r',
                  'C': 'k',
                  'F': 'g',
                  'Cl': 'g'}
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for x, s in itertools.izip(bgo.best_masked_design,
                               molecule.get_chemical_symbols()):
        draw_sphere(ax, x, color=CPK_COLORS[s])
    ax.set_aspect('equal', 'datalim')
    if to_file:
        figname = os.path.join('results', 'final_cluster_' + molecule.get_chemical_formula() + '.png')
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
