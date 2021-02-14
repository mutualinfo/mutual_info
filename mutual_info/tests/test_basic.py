import numpy as np

from ..mutual_info import entropy, entropy_gaussian, mutual_information, mutual_information_2d, DEFAULT_TRANFORM


def get_example_A():
    rng = np.random.RandomState(0)
    n = 50000
    d = 3
    P = np.array([[1, 0, 0], [0, 1, 0.5], [0, 0, 1]])
    C = np.dot(P, P.T)
    Y = rng.randn(d, n)
    X = np.dot(P, Y)
    return n, d, P, C, Y, X

def get_example_B():
    n = 50000
    d = 2
    rng = np.random.RandomState(0)
    P = np.array([[1, 0], [0.5, 1]])
    C = np.dot(P, P.T)
    U = rng.randn(2, n)
    Z = np.dot(P, U).T
    X = Z[:, 0]
    X = X.reshape(len(X), 1)
    Y = Z[:, 1]
    Y = Y.reshape(len(Y), 1)
    return n, d, P, C, Y, X, Z


def test_gaussian_entropy():
    n, d, P, C, Y, X = get_example_A()
    H = entropy_gaussian(C)
    assert np.min(H) > 0
    # test multiplicative invariance
    np.testing.assert_allclose(H, entropy_gaussian(2 * C))


def test_entropy():
    # Testing against correlated Gaussian variables
    # (analytical results are known)
    # Entropy of a 3-dimensional gaussian variable
    n, d, P, C, Y, X = get_example_A()
    H_th = entropy_gaussian(C)
    H_est = entropy(X.T, k=5, transform='rank')

    # mult invariance
    np.testing.assert_allclose(H_est, entropy(2 * X.T, k=5, transform='rank'))

    # shift invariance
    np.testing.assert_allclose(H_est, entropy(2 + X.T, k=5, transform='rank'))

    # deterministic is zero
    np.testing.assert_equal(entropy(np.ones((10, 3))), 0.0)

    # additivity for independent events
    X2 = np.random.randn(2, X.shape[1])

    # Our estimated entropy should always be less that the actual one
    # (entropy estimation undershoots) but not too much
    # TODO: why is 0.9 safe?
    H_est_std = entropy(X.T, k=5, transform='standardize')
    np.testing.assert_array_less(H_est_std, H_th)
    np.testing.assert_array_less(0.9 * H_th, H_est_std)


def test_mutual_information():
    # Mutual information between two correlated gaussian variables
    # Entropy of a 2-dimensional gaussian variable
    n, d, P, C, Y, X, Z = get_example_B()
    # in bits
    MI_est = mutual_information((X, Y), k=5, transform='standardize')
    MI_est_2 = mutual_information((Y, X), k=5, transform='standardize')

    # symmetry
    np.testing.assert_equal(MI_est, MI_est_2)

    MI_th = entropy_gaussian(C[0, 0]) + entropy_gaussian(C[1, 1]) - entropy_gaussian(C)
    # Our estimator should undershoot once again: it will undershoot more
    # for the 2D estimation that for the 1D estimation
    np.testing.assert_array_less(MI_est, MI_th)
    np.testing.assert_array_less(MI_th, MI_est + 0.3)


def test_degenerate():
    # Test that our estimators are well-behaved with regards to
    # degenerate solutions
    rng = np.random.RandomState(0)
    x = rng.randn(50000)
    X = np.c_[x, x]
    assert np.isfinite(entropy(X))
    assert np.isfinite(mutual_information((x[:, np.newaxis], x[:, np.newaxis])))
    assert 2.9 < mutual_information_2d(x, x) < 3.1


def test_mutual_information_2d():
    # Mutual information between two correlated gaussian variables
    # Entropy of a 2-dimensional gaussian variable
    n = 50000
    rng = np.random.RandomState(0)
    # P = np.random.randn(2, 2)
    P = np.array([[1, 0], [0.9, 0.1]])
    C = np.dot(P, P.T)
    U = rng.randn(2, n)
    Z = np.dot(P, U).T
    X = Z[:, 0]
    X = X.reshape(len(X), 1)
    Y = Z[:, 1]
    Y = Y.reshape(len(Y), 1)
    # in bits
    MI_est = mutual_information_2d(X.ravel(), Y.ravel())
    MI_th = entropy_gaussian(C[0, 0]) + entropy_gaussian(C[1, 1]) - entropy_gaussian(C)
    print((MI_est, MI_th))
    # Our estimator should undershoot once again: it will undershoot more
    # for the 2D estimation that for the 1D estimation
    np.testing.assert_array_less(MI_est, MI_th)
    np.testing.assert_array_less(MI_th, MI_est + 0.2)


if __name__ == "__main__":
    # Run our tests
    test_entropy()
    test_mutual_information()
    test_degenerate()
    test_mutual_information_2d()
