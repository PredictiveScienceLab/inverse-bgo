"""
Global Optimization of Expensive Functions.

Author:
    Ilias Bilionis

Date:
    10/15/2014
    01/29/2015

"""


__all__ = ['expected_improvement',
           'fb_expected_improvement',
           'expected_information_gain',
           'minimize', 'maximize',
           'plot_summary', 'plot_summary_2d']


import GPy
import GPy.inference.mcmc
from GPy.inference.mcmc import HMC
import numpy as np
import math
import scipy
import scipy.stats as stats
from scipy.integrate import quad
#from choldate import choldowndate, cholupdate


from statsmodels.sandbox.distributions.multivariate import mvnormcdf
import math


def remove(mu, S, i):
    """
    Remove i element from mu and S.
    """
    mu_ni = np.hstack([mu[:i], mu[i+1:]])
    S_nini = np.array(np.bmat([[S[:i, :i], S[:i, i+1:]],
                               [S[i+1:, :i], S[i+1:, i+1:]]]))
    return mu_ni, S_nini


def maxpdf(x, mu, S):
    s = np.zeros(x.shape[0])
    d = mu.shape[0]
    for i in xrange(d):
        mu_i = mu[i]
        S_ii = S[i, i]
        mu_ni, S_nini = remove(mu, S, i)
        S_ini = np.array(np.bmat([[S[:i, i], S[i+1:, i]]]))
        mu_nii = mu_ni[:, None] + np.dot(S_ini.T, x[None, :] - mu_i) / S_ii
        S_ninii = S_nini - np.dot(S_ini, S_ini.T) / S_ii
        phi_i = norm.pdf(x, loc=mu_i, scale=np.sqrt(S_ii))
        Phi_i = np.array([mvnormcdf(x[j], mu_nii[:, j], S_ninii) 
                          for j in xrange(x.shape[0])])
        s += phi_i * Phi_i
    return s


def expected_improvement(X_design, model, mode='min'):
    """
    Compute the Expected Improvement criterion at ``x``.
    """
    y = model.Y.flatten()
    m_s, v_s = model.predict(X_design)[:2]
    m_s = m_s.flatten()
    v_s = v_s.flatten()
    s_s = np.sqrt(v_s)
    if mode == 'min':
        m_n = np.min(y)
        u = (m_n - m_s) / s_s
    elif mode == 'max':
        m_n = np.max(y)
        u = (m_s - m_n) / s_s
    else:
        raise NotImplementedError('I do not know what to do with mode %s' %mode)
    ei = s_s * (u * stats.norm.cdf(u) + stats.norm.pdf(u))
    return ei


def fb_expected_improvement(X_design, model, mode='min', stepsize=1e-2,
                            num_samples=100):
    """
    Compute the fully Bayesian expected improvement criterion.
    """
    model.rbf.variance.set_prior(GPy.priors.LogGaussian(0., 1.))
    model.rbf.lengthscale.set_prior(GPy.priors.LogGaussian(0., 0.1))
    mcmc = HMC(model, stepsize=stepsize)
    params = mcmc.sample(num_samples=num_samples)
    ei_all = []
    for i in xrange(params.shape[0]):
        model.rbf.variance = params[i, 0]
        model.rbf.lengthscale = params[i, 1]
        ei = expected_improvement(X_design, model, mode=mode)
        ei_all.append(ei)
    ei_all = np.array(ei_all)
    ei_fb = ei_all.mean(axis=0)
    return ei_fb


def min_qoi(X_design, f):
    """
    A QoI that corresponds to the min of the function.
    """
    return np.argmin(f, axis=0)


def kl_divergence(g1, g2):
    """
    Compute the KL divergence.
    """
    f = lambda(x): g1.evaluate([[x]]) * np.log(g1.evaluate([[x]]) / g2.evaluate([[x]]))
    return quad(f, 0, 6)


def expected_information_gain(X_design, model, num_Omegas=1000,
                              num_y=100,
                              qoi=min_qoi,
                              qoi_bins=None,
                              qoi_num_bins=20):
    """
    Compute the expected information gain criterion at ``x``.
    """
    import matplotlib.pyplot as plt
    m_d, K_d = model.predict(X_design, full_cov=True)[:2]
    U_d = scipy.linalg.cholesky(K_d, lower=False)    
    Omegas = np.random.randn(X_design.shape[0], num_Omegas)
    delta_y_i = np.random.randn(num_y)
    # Find the histogram of Q the current data
    S_d = m_d + np.dot(U_d.T, Omegas)
    Q_d = qoi(X_design, S_d)
    tmp = stats.itemfreq(Q_d)
    yy = model.posterior_samples(X_design, 10)
    plt.plot(X_design, yy, 'm', linewidth=2)
    plt.savefig('examples/samples.png')
    plt.clf()
    p_d = np.zeros((X_design.shape[0],))
    p_d[np.array(tmp[:, 0], dtype='int')] = tmp[:, 1] / np.sum(tmp[:, 1])
    if qoi_bins is None and qoi is min_qoi:
        #qoi_bins = np.linspace(np.min(Q_d), np.max(Q_d), qoi_num_bins)[None, :]
        qoi_bins = np.linspace(X_design[0, 0], X_design[-1, 0], qoi_num_bins)[None, :]
    H_d, e_d = np.histogramdd(Q_d, normed=True, bins=qoi_bins)
    delta_e_d = e_d[0][1] - e_d[0][0]
    #p_d = H_d * delta_e_d
    plt.plot(X_design, p_d)
    plt.plot(X_design, m_d)
    plt.plot(model.X, model.Y, 'ro', markersize=10)
    plt.hist(X_design[Q_d, 0], normed=True, alpha=0.5)
    plt.savefig('examples/kde_Q.png')
    plt.clf()
    print 'Entropy:', stats.entropy(p_d)
    G = np.zeros((X_design.shape[0],))
    p_d += 1e-16
    for i in xrange(X_design.shape[0]):
        u_di = K_d[:, i] / math.sqrt(K_d[i, i])
        u_di = u_di[:, None]
        #K_dd = K_d - np.dot(u_di, u_di.T)
        #K_dd += np.eye(K_d.shape[0]) * 1e-6
        choldowndate(U_d, u_di.flatten().copy())
        #U_d = scipy.linalg.cholesky(K_dd, lower=False)
        # Pick a value for y:
        Omegas = np.random.randn(X_design.shape[0], num_Omegas)
        delta_y_i = np.random.randn(num_y)
        m_dgi = m_d + delta_y_i * u_di
        S_dgi = m_dgi[:, :, None] + np.dot(U_d.T, Omegas)[:, None, :]
        #for j in xrange(num_y):
        #    print S_dgi[:, j, :]
        #    plt.plot(X_design, S_dgi[:, j, :], 'm', linewidth=0.5)
        #    plt.plot(model.X, model.likelihood.Y, 'ro', markersize=10)
        #    plt.savefig('examples/ig_S_' + str(i).zfill(2) + '_' + str(j).zfill(2) + '.png')
        #    plt.clf()
        Q_dgi = qoi(X_design, S_dgi)
        #print Q_dgi
        #quit()
        p_d_i = np.zeros((num_y, X_design.shape[0]))
        for j in xrange(num_y):
            tmp = stats.itemfreq(Q_dgi[j, :])
            p_d_i[j, np.array(tmp[:, 0], dtype='int')] = tmp[:, 1] / np.sum(tmp[:, 1])
        p_d_i += 1e-16
        G[i] = np.mean([stats.entropy(p_d_i[j, :], p_d) for j in xrange(num_y)])
        #G[i] = np.mean([-stats.entropy(p_d_i[j, :]) for j in xrange(num_y)])
        #plt.plot(X_design, S_dgi[:, :, 0], 'm', linewidth=0.5)
        #plt.plot(X_design, m_d, 'r', linewidth=2)
        plt.plot(model.X, np.zeros((model.X.shape[0], 1)), 'ro', markersize=10)
        plt.plot(X_design, np.mean(p_d_i, axis=0), 'g', linewidth=2)
        plt.savefig('examples/ig_S_' + str(i).zfill(2) + '.png')
        plt.clf()
        print X_design[i, 0], G[i]
        cholupdate(U_d, u_di.flatten().copy())
    plt.plot(X_design, G)
    plt.savefig('examples/ig_KL.png')
    plt.clf()
    return G

    
def plot_summary(f, X_design, model, prefix, G, Gamma_name):
    """
    Plot a summary of the current iteration.
    """
    import matplotlib.pyplot as plt
    X = model.X
    y = model.Y
    m_s, k_s = model.predict(X_design, full_cov=True)
    m_05, m_95 = model.predict_quantiles(X_design)
    fig, ax1 = plt.subplots()
    ax1.plot(X_design, f(X_design), 'b', linewidth=2)
    ax1.plot(X, y, 'go', linewidth=2, markersize=10, markeredgewidth=2)
    ax1.plot(X_design, m_s, 'r--', linewidth=2)
    ax1.fill_between(X_design.flatten(), m_05.flatten(), m_95.flatten(),
                     color='grey', alpha=0.5)
    ax1.set_ylabel('$f(x)$', fontsize=16)
    ax2 = ax1.twinx()
    ax2.plot(X_design, G, 'g', linewidth=2)
    ax2.set_ylabel('$%s(x)$' % Gamma_name, fontsize=16, color='g')
    #ax2.set_ylim([0., 3.])
    plt.setp(ax2.get_yticklabels(), color='g')
    png_file = prefix + '.png'
    print 'Writing:', png_file
    plt.savefig(png_file)
    plt.clf()


def plot_summary_2d(f, X_design, model, prefix, G, Gamma_name):
    """
    Plot a summary of the current iteration.
    """
    import matplotlib.pyplot as plt
    n = np.sqrt(X_design.shape[0])
    X1, X2 = (X_design[:, i].reshape((n, n)) for i in range(2))
    GG = G.reshape((n, n))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.contourf(X1, X2, GG)
    fig.colorbar(cax)
    ax.set_xlabel('$x_1$', fontsize=16)
    ax.set_ylabel('$x_2$', fontsize=16)
    plt.savefig(prefix + '_' + Gamma_name + '.png')
    plt.clf()
    X = model.X
    m_s, k_s = model.predict(X_design)
    M_s = m_s.reshape((n, n))
    S_s = np.sqrt(k_s.reshape((n, n)))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.contourf(X1, X2, M_s)
    fig.colorbar(cax)
    ax.plot(X[:, 0], X[:, 1], 'k.', markersize=10)
    ax.set_xlabel('$x_1$', fontsize=16)
    ax.set_ylabel('$x_2$', fontsize=16)
    plt.savefig(prefix + '_mean.png')
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.contourf(X1, X2, S_s)
    fig.colorbar(cax)
    ax.plot(X[:, 0], X[:, 1], 'k.', markersize=10)
    ax.set_xlabel('$x_1$', fontsize=16)
    ax.set_ylabel('$x_2$', fontsize=16)
    plt.savefig(prefix + '_std.png')
    plt.clf()



def minimize(f, X_init, X_design, prefix="minimize", Gamma=expected_improvement,
             Gamma_name='EI', max_it=10, tol=1e-1, callback=None):
    """
    Optimize f using a limited number of evaluations.
    """
    X = X_init
    y = np.array([f(X[i, :]) for i in xrange(X.shape[0])])
    k = GPy.kern.RBF(X.shape[1], ARD=True)
    for count in xrange(max_it):
        model = GPy.models.GPRegression(X, y, k)
        model.Gaussian_noise.variance.constrain_fixed(1e-6)
        model.optimize()
        print str(model)
        G = Gamma(X_design, model)
        if callback is not None:
            callback(f, X_design, model,
                     prefix + '_' + str(count).zfill(2), G, Gamma_name)
        i = np.argmax(G)
        if G[i] < tol:
            print '*** converged'
            break
        print 'I am adding:', X_design[i:(i+1), :]
        print 'which has a G of', G[i]
        X = np.vstack([X, X_design[i:(i+1), :]])
        y = np.vstack([y, f(X_design[i, :])])
        print 'it =', count+1, ', min =', np.min(y), ' arg min =', X[np.argmin(y), :]
    return X, y


def maximize(f, X_init, X_design, prefix='maximize', Gamma=expected_improvement,
             Gamma_name='EI', max_it=10, tol=1e-1, callback=None):
    """
    Maximize the function ``f``.
    """
    f_minus = lambda(x) : -f(x)
    return minimize(f_minus, X_init, X_design, prefix=prefix, Gamma=Gamma,
                    Gamma_name=Gamma_name, max_it=max_it, tol=tol)
