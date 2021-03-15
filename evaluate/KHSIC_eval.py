from __future__ import division
import numpy as np
from scipy.stats import gamma


class KHSIC:
    """
    Inputs:
    X 			n by dim_x matrix
    Y 			n by dim_y matrix
    alph 		level of test

    Outputs:
    testStat	test statistics
    thresh		test threshold for level alpha test
    """
    def __init__(self, X, Y):
        super(KHSIC, self).__init__()
        self.X = X
        self.Y = Y

    def linear_kernel(self, matrix):
        return np.dot(matrix, matrix.T)

    def rbf_dot(self, pattern1, pattern2, deg):
        size1 = pattern1.shape
        size2 = pattern2.shape

        G = np.sum(pattern1 * pattern1, 1).reshape(size1[0], 1)
        H = np.sum(pattern2 * pattern2, 1).reshape(size2[0], 1)

        Q = np.tile(G, (1, size2[0]))
        R = np.tile(H.T, (size1[0], 1))

        H = Q + R - 2 * np.dot(pattern1, pattern2.T)

        H = np.exp(-H / 2 / (deg ** 2))

        return H

    def hsic_gam(self, alph=0.5):
        """
        X, Y are numpy vectors with row - sample, col - dim
        alph is the significance level
        auto choose median to be the kernel width
        """
        X = self.X
        Y = self.Y
        n = X.shape[0]

        # ----- width of X -----
        Xmed = X

        G = np.sum(Xmed * Xmed, 1).reshape(n, 1)
        Q = np.tile(G, (1, n))
        R = np.tile(G.T, (n, 1))

        dists = Q + R - 2 * np.dot(Xmed, Xmed.T)
        dists = dists - np.tril(dists)
        dists = dists.reshape(n ** 2, 1)

        width_x = np.sqrt(0.5 * np.median(dists[dists > 0]))
        # ----- -----

        # if using a rbf kernel for Y
        # ----- width of Y -----
        Ymed = Y

        G = np.sum(Ymed * Ymed, 1).reshape(n, 1)
        Q = np.tile(G, (1, n))
        R = np.tile(G.T, (n, 1))

        dists = Q + R - 2 * np.dot(Ymed, Ymed.T)
        dists = dists - np.tril(dists)
        dists = dists.reshape(n ** 2, 1)

        width_y = np.sqrt(0.5 * np.median(dists[dists > 0]))
        # ----- -----

        bone = np.ones((n, 1), dtype=float)
        H = np.identity(n) - np.ones((n, n), dtype=float) / n

        # rbf kernel for X
        K = self.rbf_dot(X, X, width_x)
        # linear kernel for Y
        L = self.linear_kernel(Y)

        Kc = np.dot(np.dot(H, K), H)
        Lc = np.dot(np.dot(H, L), H)

        testStat = np.sum(Kc.T * Lc) / n

        varHSIC = (Kc * Lc / 6) ** 2

        varHSIC = (np.sum(varHSIC) - np.trace(varHSIC)) / n / (n - 1)

        varHSIC = varHSIC * 72 * (n - 4) * (n - 5) / n / (n - 1) / (n - 2) / (n - 3)

        K = K - np.diag(np.diag(K))
        L = L - np.diag(np.diag(L))

        muX = np.dot(np.dot(bone.T, K), bone) / n / (n - 1)
        muY = np.dot(np.dot(bone.T, L), bone) / n / (n - 1)

        mHSIC = (1 + muX * muY - muX - muY) / n

        al = mHSIC ** 2 / varHSIC
        bet = varHSIC * n / mHSIC

        thresh = gamma.ppf(1 - alph, al, scale=bet)[0][0]
        p_val = gamma.cdf(testStat, al, scale=bet)[0][0]

        return (testStat, thresh, p_val)
