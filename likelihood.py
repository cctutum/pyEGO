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
    n, k = X.shape
    one = np.ones(n)

    # Build correlation matrix (Psi)
    Psi = np.zeros((n, n))
    # It is a symmetric matrix, so we only need the upper matrix
    for i in range(n):
        for j in (range(i+1, n)):
            Psi[i,j] = np.exp(-np.sum(theta * np.abs(X[i,:]-X[j,:])**2))

    # Add upper and lower halves, diagonal of ones and small number to reduce 
    # ill-conditioning
    Psi = Psi + Psi.T + np.eye(n) * (1 + np.finfo(float).eps)

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


def likelihood_wrapper(t, X, y):
    outputs = likelihood(t, X, y)
    return outputs[0]



