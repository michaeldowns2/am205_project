"""
"""
import logging
import random
import time


import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.spatial as spsp


random.seed(205)
np.random.seed(205)

def k_rbf_1d(x1, x2, h):
    return np.exp((-1./h) * (x1 - x2) **2)

def k_rbf_deriv_1d(x1, x2, h):
    return (-1./h) * 2 * (x1 - x2) * k_rbf_1d(x1, x2, h)

def k_rbf(x1, x2, width):
    return np.exp((-1./(2.*width**2)) * np.linalg.norm(x1 - x2, 2)**2 )


class SSGE:
    def __init__(self, X, kernel='rbf'):
        self.K = None
        self.eigvals = None
        self.eigvecs = None

        self.X = X

        self.num_samples = len(X)

        print("Computing gram matrix")
        self.__compute_gram()

        print("Computing eigenvalues / vectors of gram matrix")
        self.__compute_eigenvalues()

        print("Determining mus / psis")
        self.__get_psis_mus()
        

    def __compute_gram(self):
        """
        Computes K, gram matrix
        """

        dists = spsp.distance.pdist(X, 'euclidean')

        width = np.median(dists)

        sq_dists = dists**2

        k_vals = np.exp(-1./(2.*width**2) * sq_dists)

        K = np.zeros((self.num_samples,
                      self.num_samples))

        K[np.triu_indices(self.num_samples, 1)] = k_vals

        K = K + K.T
        np.fill_diagonal(K, 1)

        self.K = K


    def __compute_eigenvalues(self):
        if self.K is not None:
            eigvals, eigvecs = np.linalg.eig(self.K)

            idx_sort = np.argsort(eigvals)[::-1]

            eigvals = eigvals[idx_sort]

            eigvecs = eigvecs[:, idx_sort]

            self.eigvals = eigvals
            self.eigvecs = eigvecs

        else:
            raise Exception("Form gram matrix K first")

    def __get_psis_mus(self):
        if self.eigvecs is not None and self.eigvals is not None:

            self.mus = self.eigvals / self.num_samples
            self.psis = self.eigvecs * np.sqrt(self.num_samples)

        else:
            raise Exception("Compute eigenvalues and eigenvectors of gram matrix first")


    def jth_psi(j, x):
        pass


    def jth_psi_factory(j):
        def jth_psi(x):
            pass



if __name__ == '__main__':
    dim = 5
    num_samples = 100
    X = np.random.multivariate_normal(np.zeros(dim),
                                      np.eye(dim),
                                      100)


    print(len(X))
    ssge = SSGE(X)
