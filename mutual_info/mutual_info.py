"""
Non-parametric computation of entropy and mutual-information

Adapted by G Varoquaux for code created by R Brette, itself
from several papers (see in the code).

These computations rely on nearest-neighbor statistics
"""

import numpy as np
from numpy import pi
from scipy import ndimage
from scipy.linalg import det
from scipy.special import gamma, psi
from sklearn.neighbors import NearestNeighbors

__all__ = ["entropy", "mutual_information", "entropy_gaussian"]

EPS = np.finfo(float).eps


def nearest_distances(X, k=1):
    """
    X = array(N,M)
    N = number of points
    M = number of dimensions

    returns the distance to the kth nearest neighbor for every point in X
    """
    knn = NearestNeighbors(n_neighbors=k + 1)
    knn.fit(X)
    d, _ = knn.kneighbors(X)  # the first nearest neighbor is itself
    return d[:, -1]  # returns the distance to the kth nearest neighbor


def entropy_gaussian(C):
    """
    Entropy of a gaussian variable with covariance matrix C
    """
    if np.isscalar(C):  # C is the variance
        return 0.5 * (1 + np.log(2 * pi)) + 0.5 * np.log(C)
    else:
        n = C.shape[0]  # dimension
        return 0.5 * n * (1 + np.log(2 * pi)) + 0.5 * np.log(abs(det(C)))


def entropy(X, k=1):
    """Returns the entropy of the X.

    Parameters
    ===========

    X : array-like, shape (n_samples, n_features)
        The data the entropy of which is computed

    k : int, optional
        number of nearest neighbors for density estimation

    Notes
    ======

    Kozachenko, L. F. & Leonenko, N. N. 1987 Sample estimate of entropy
    of a random vector. Probl. Inf. Transm. 23, 95-101.
    See also: Evans, D. 2008 A computationally efficient estimator for
    mutual information, Proc. R. Soc. A 464 (2093), 1203-1215.
    and:
    Kraskov A, Stogbauer H, Grassberger P. (2004). Estimating mutual
    information. Phys Rev E 69(6 Pt 2):066138.
    """

    # Distance to kth nearest neighbor
    r = nearest_distances(X, k)  # squared distances
    n, d = X.shape
    volume_unit_ball = (pi ** (0.5 * d)) / gamma(0.5 * d + 1)
    """
    F. Perez-Cruz, (2008). Estimation of Information Theoretic Measures
    for Continuous Random Variables. Advances in Neural Information
    Processing Systems 21 (NIPS). Vancouver (Canada), December.

    return d*mean(log(r))+log(volume_unit_ball)+log(n-1)-log(k)
    """
    return d * np.mean(np.log(r + np.finfo(X.dtype).eps)) + np.log(volume_unit_ball) + psi(n) - psi(k)


def mutual_information(variables, k=1):
    """
    Returns the mutual information between any number of variables.
    Each variable is a matrix X = array(n_samples, n_features)
    where
      n = number of samples
      dx,dy = number of dimensions

    Optionally, the following keyword argument can be specified:
      k = number of nearest neighbors for density estimation

    Example: mutual_information((X, Y)), mutual_information((X, Y, Z), k=5)
    """
    if len(variables) < 2:
        raise AttributeError("Mutual information must involve at least 2 variables")
    all_vars = np.hstack(variables)
    # check that mi(X, X) = entropy(X)
    check = np.unique(all_vars, axis=1)
    if all_vars.shape[1] != check.shape[1]:
        print(f"WARNING: dropping {all_vars.shape[1] - check.shape[1]} variables as the samples are identical!")
        all_vars = check
    return sum([entropy(X, k=k) for X in variables]) - entropy(all_vars, k=k)


def mutual_information_2d(x, y, sigma=1, normalized=False):
    """
    Computes (normalized) mutual information between two 1D variate from a
    joint histogram.

    Parameters
    ----------
    x : 1D array
        first variable

    y : 1D array
        second variable

    sigma: float
        sigma for Gaussian smoothing of the joint histogram

    Returns
    -------
    nmi: float
        the computed similariy measure

    """
    bins = (256, 256)

    jh = np.histogram2d(x, y, bins=bins)[0]

    # smooth the jh with a gaussian filter of given sigma
    ndimage.gaussian_filter(jh, sigma=sigma, mode="constant", output=jh)

    # compute marginal histograms
    jh = jh + EPS
    sh = np.sum(jh)
    jh = jh / sh
    s1 = np.sum(jh, axis=0).reshape((-1, jh.shape[0]))
    s2 = np.sum(jh, axis=1).reshape((jh.shape[1], -1))

    # Normalised Mutual Information of:
    # Studholme,  jhill & jhawkes (1998).
    # "A normalized entropy measure of 3-D medical image alignment".
    # in Proc. Medical Imaging 1998, vol. 3338, San Diego, CA, pp. 132-143.
    if normalized:
        mi = ((np.sum(s1 * np.log(s1)) + np.sum(s2 * np.log(s2))) / np.sum(jh * np.log(jh))) - 1
    else:
        mi = np.sum(jh * np.log(jh)) - np.sum(s1 * np.log(s1)) - np.sum(s2 * np.log(s2))

    return mi
