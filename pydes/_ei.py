"""
An implmentation of the expected improvement data acquisition function.

Author:
    Ilias Bilionis

Date:
    5/1/2015

"""


__all__ = ['expected_improvement']



import numpy as np
import scipy.stats as stats


def expected_improvement(Xd, model, mode='min', noise=None):
    """
    Compute the expected improvement at ``Xd``.

    :param Xd:      The design points on which to evaluate the improvement.
    :param model:   The model of which we want to know the improvement. It must
                    have a method called ``predict()`` that accepts a matrix
                    representing the design points and returns a tuple representing
                    the mean predictions and the predictive variance.
    :param noise:   The variance of the measurement noise on each design point. If
                    ``None``, then we attempt to get this noise from
                    ``model.likelihood.noise``, if possible.
    :returns:       The expected improvement on all design points.
    """
    assert hasattr(model, 'predict')
    if noise is None:
        if hasattr(model, 'likelihood') and hasattr(model.likelihood, 'variance'):
            noise = float(model.likelihood.variance)
        else:
            noise = 0.
    X = model.X.copy()
    #m_obs = model.predict(X)[0].flatten()
    m_obs = model.Y.flatten()
    m_s, v_s = model.predict(Xd)[:2]
    m_s = m_s.flatten()
    v_s = v_s.flatten() - noise
    s_s = np.sqrt(v_s)
    idx = np.isnan(s_s)
    #print idx.any()
    s_s[np.isnan(s_s)] = 1e-10
    if mode == 'min':
        i_n = np.argmin(m_obs)
        m_n = m_obs[i_n]
        u = (m_n - m_s) / s_s
    elif mode == 'max':
        i_n = np.argmax(m_obs)
        m_n = m_obs[i_n]
        u = (m_s - m_n) / s_s
    else:
        raise NotImplementedError('I do not know what to do with mode %s' %mode)
    ei = s_s * (u * stats.norm.cdf(u) + stats.norm.pdf(u))
    i0 = np.argmax(ei)
    #print 'EI: ', ei[i0], 'Pred.:', m_s[i0], 'Var.:', s_s[i0]
    return ei, i_n, m_n