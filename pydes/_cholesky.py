"""Everything related to Cholesky decomposition.

Author:
    Ilias Bilionis

Date:
    2/1/2013
"""


__all__ = ['update_cholesky', 'update_cholesky_linear_system']


import numpy as np
import scipy.linalg


def update_cholesky(L, B, C):
    """Updates the Cholesky decomposition of a matrix.

    We assume that L is the lower Cholesky decomposition of a matrix A, and
    we want to calculate the Cholesky decomposition of the matrix:
        A   B
        B.T C.
    It can be easily shown that the new decomposition is given by:
        L   0
        D21 D22,
    where
        L * D21.T = B,
    and
        D22 * D22.T = C - D21 * D21.T.

    Arguments:
        L       ---         The Cholesky decomposition of the original
                            n x n matrix.
        B       ---         The n x m upper right part of the new matrix.
        C       ---         The m x m bottom diagonal part of the new matrix.

    Return:
        The lower Cholesky decomposition of the new matrix.
    """
    assert isinstance(L, np.ndarray)
    assert L.ndim == 2
    assert L.shape[0] == L.shape[1]
    assert isinstance(B, np.ndarray)
    assert B.ndim == 2
    assert B.shape[0] == L.shape[0]
    assert isinstance(C, np.ndarray)
    assert C.ndim == 2
    assert B.shape[1] == C.shape[0]
    assert C.shape[0] == C.shape[1]
    n = L.shape[0]
    m = B.shape[1]
    L_new = np.zeros((n + m, n + m))
    L_new[:n, :n] = L
    D21 = L_new[n:, :n]
    D22 = L_new[n:, n:]
    D21[:] = scipy.linalg.solve_triangular(L, B, lower=True).T
    if m == 1:
        D22[:] = np.sqrt(C[0, 0] - np.dot(D21, D21.T))
    else:
        D22[:] = scipy.linalg.cholesky(C - np.dot(D21, D21.T), lower=True)
    return L_new


def update_cholesky_linear_system(x, L_new, z):
    """Updates the solution of a linear system involving a Cholesky factor.

    Assume that originally we had an n x n matrix lower triangular matrix L
    and that we have already solved the linear system:
        L * x = y.
    We wish now to solve the linear system:
        L_new * x_new = y_new,
    where L_new is again lower triangular but the fist n x n component is
    identical to L and y_new is (y, z). The solution is:
        x_new = (x, x_u),
    where x_u is the solution of the triangular system:
        D22 * x_u = z - D21 * x,
    where D22 is the lower m x m component of L_new and D21 is the m x n
    bottom left component of L_new.

    Arguments:
        x       ---     The original solution. It can be either a vector or a
                        matrix.
        L_new   ---     The new lower Cholesky factor.
        z       ---     The new right hand side as described above.

    Return:
        The solution of the new linear system.
    """
    assert isinstance(x, np.ndarray)
    assert x.ndim <= 2
    regularized_x = False
    if x.ndim == 1:
        regularized_x = True
        x = x.reshape((x.shape[0], 1))
    assert isinstance(L_new, np.ndarray)
    assert L_new.shape[0] == L_new.shape[1]
    assert isinstance(z, np.ndarray)
    assert z.ndim <= 2
    regularized_z = False
    if z.ndim == 1:
        regularized_z = True
        z = z.reshape((z.shape[0], 1))
    assert x.shape[1] == z.shape[1]
    assert L_new.shape[0] == x.shape[0] + z.shape[0]
    n = x.shape[0]
    D22 = L_new[n:, n:]
    D21 = L_new[n:, :n]
    x_u = scipy.linalg.solve_triangular(D22, z - np.dot(D21, x), lower=True)
    y = np.vstack([x, x_u])
    if regularized_x or regularized_z:
        y = y.reshape((y.shape[0],))
    return y
