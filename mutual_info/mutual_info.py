"""
Non-parametric computation of entropy and mutual-information

Adapted by G Varoquaux for code created by R Brette, itself
from several papers (see in the code).

These computations rely on nearest-neighbor statistics
"""
from functools import lru_cache

import numpy as np
from numpy import pi
from scipy import ndimage
from scipy.linalg import det
from scipy.special import gamma, digamma
from sklearn.neighbors import NearestNeighbors

__all__ = ["entropy", "mutual_information", "entropy_gaussian"]

EPS = np.finfo(float).eps
DEFAULT_TRANFORM = 'rank'

def handle_transform(X, transform):
    allowed_transforms = {
            'rank': rank_transform,
            'standardize': standardize_transform,
            None: lambda x: x
    }
    if transform not in allowed_transforms:
        raise Exception(f'Unknown transform {transform}. Allowed={allowed_transforms}')
    return allowed_transforms[transform](X)


def rank_transform(x, pct=True):
    """
    Basically a pure numpy version of pd.rank
    See https://stackoverflow.com/questions/5284646/rank-items-in-an-array-using-python-numpy-without-sorting-array-twice/59864018#59864018 for more notes
    """
    ind = np.argsort(x, axis=0)
    ranks = np.empty_like(ind)
    if x.ndim == 1:
        values = np.arange(x.shape[0])
    else:
        values = np.repeat(np.arange(x.shape[0])[:, None], x.shape[1], axis=1)
    np.put_along_axis(ranks, ind, values, axis=0)
    if pct:
        ranks = (ranks + 1) / (ranks.shape[0] + 1)
    return ranks


def standardize_transform(x):
    return (x - x.mean(axis=0)) / x.std(axis=0)


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


def covar_to_corr(C):
    assert np.allclose(C, C.T), 'Covariance matrix not symmetric'
    d = 1 / np.sqrt(np.diag(C))
    # same as np.diag(d) @ C @ np.diag(d), but using broadcasting
    return d * (d * C).T


def entropy_gaussian(C):
    """
    Entropy of a gaussian variable with covariance matrix C
    """
    # Remember Covariance is dimensional variable. Corr is dimensionless.
    # You can never take the log of a dimensional variable.
    if np.isscalar(C):  # corr is just 1
        return 0.5 * (1 + np.log(2 * pi)) # + 0.5 * np.log(1)
    else:
        corr = covar_to_corr(C)
        n = corr.shape[0]  # dimension
        return 0.5 * n * (1 + np.log(2 * pi)) + 0.5 * np.log(abs(det(corr)))



def entropy(X, k=1, transform=DEFAULT_TRANFORM):
    """Returns the approximate entropy of X via some knn estimator.

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
    check = np.unique(X, axis=0)
    if check.shape[0] == 1:
        # deterministic variable has entropy 0
        return 0.0
    X = handle_transform(X, transform)

    # Distance to kth nearest neighbor
    R = nearest_distances(X, k)  # squared distances
    N, D = X.shape
    """
    F. Perez-Cruz, (2008). Estimation of Information Theoretic Measures
    for Continuous Random Variables. Advances in Neural Information
    Processing Systems 21 (NIPS). Vancouver (Canada), December.

    See eq (9):

        (9) hhat = - mean(log(phat))
        (3) phat = k / (n - 1) / volume_unit_ball / r ** d

    So phat = d * mean(log(r)) + log(vub) + log(n - 1) - log(k)
    In eq (20) of Kraskov (2003), it is different:

        (20) hhat = -digamma(k) + digamma(n) + log(vub) + d / n * mean(log(2 * r))

    See https://hal.inria.fr/hal-01272527/document probably for best description.

    I think the confusion is that in the L_infinity norm, unitball is 2 ** D.

    See also the dit project on github. Has a similar form.
    """
    R = R + np.finfo(X.dtype).eps
    volume_unit_ball = (pi ** (D / 2)) / gamma(1 + D / 2)
    return digamma(N) - digamma(k) + np.log(volume_unit_ball) + D * np.mean(np.log(R))


def mutual_information(variables, k=1, transform=DEFAULT_TRANFORM):
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
    variables = [handle_transform(x, transform) for x in variables]
    all_vars = np.hstack(variables)
    # # check that mi(X, X) = entropy(X)
    # check = np.unique(all_vars, axis=1)
    # if all_vars.shape[1] != check.shape[1]:
    #     print(f"WARNING: dropping {all_vars.shape[1] - check.shape[1]} variables as the samples are identical!")
    #     all_vars = check
    return sum([entropy(X, k=k, transform=None) for X in variables]) - entropy(all_vars, k=k, transform=None)


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
