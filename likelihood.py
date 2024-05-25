import numpy as np
from scipy.spatial.distance import cdist
from sampling import lhs
import matplotlib.pyplot as plt
import random


def likelihood(t, X, y):
    """
    Calculates the negative of the concentrated ln-likelihood.

    Inputs:
        t - vector of log(theta) parameters
        X - n x k matrix of sample locations
        y - n x 1 vector of observed data

    Outputs:
        NegLnLike - concentrated log-likelihood *-1 for minimising
        Psi - correlation matrix
        U - Choleski factorisation of correlation matrix
    """
    
    theta = 10.0**t
    p = 2
    n, k = X.shape
    one = np.ones(n)

    # Build correlation matrix
    dist = cdist(X, X, metric='minkowski', p=p)
    Psi = np.exp(-np.sum(theta * np.abs(dist), axis=2))

    # Add diagonal and small number to reduce ill-conditioning
    Psi = Psi + np.eye(n) + np.eye(n) * np.finfo(float).eps

    # Cholesky factorisation
    try:
        U = np.linalg.cholesky(Psi)
    except np.linalg.LinAlgError:
        NegLnLike = 1e4
        Psi = None
        U = None
        return NegLnLike, Psi, U

    # Sum lns of diagonal to find ln(abs(det(Psi)))
    LnDetPsi = 2 * np.sum(np.log(np.abs(np.diag(U))))

    # Use back-substitution of Cholesky instead of inverse
    mu = (one @ np.linalg.solve(U, np.linalg.solve(U.T, y))) / \
         (one @ np.linalg.solve(U, np.linalg.solve(U.T, one)))
    SigmaSqr = ((y - one * mu) @ np.linalg.solve(U, np.linalg.solve(U.T, y - one * mu))) / n
    NegLnLike = -(-(n / 2) * np.log(SigmaSqr) - 0.5 * LnDetPsi)

    return NegLnLike, Psi, U

d = 2
n = 3*d
X = lhs(d, n)
y = random.randint(0, 9)
print(X)
plt.plot(X[:,0], X[:,1], 'bo')

t = np.random.uniform(-3, 2, d)
print(t)
likelihood(t, X, y)


