"""
"""
import logging
import random
import time


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import scipy.spatial as spsp

random.seed(205)
np.random.seed(205)





def normal_log_density(x):
    return -1./2 * np.log(2*np.pi) - 1./2 * x**2

def normal_log_density_deriv(x):
    return -1. * x

class SSGE:
    """
    width computation:
    heuristic1: use just samples
    heuristic2: use just samples + single gradient evaluation point
    heuristic3: use samples + all gradient evaluation points
    g: gradient evaluation values
    """
    def __init__(self, X,
                 g=None,
                 J=None,
                 kernel='rbf',
                 width_rule='heuristic3',
                 recompute_width=False,
                 width=None,
                 r=0.99):

        self.K = None
        self.eigvals = None
        self.eigvecs = None
        self.width_rule = width_rule
        self.width = width
        self.g = g
        self.J = J


        self.r = r

        self.X = X

        self.num_samples, self.dim = X.shape

        print("Computing gram matrix")
        self.__compute_gram()

        print("Computing eigenvalues / vectors of gram matrix")
        self.__compute_eigenvalues()

        print("Computing J")
        self.__get_J()
        print(f"J: {self.J}")

        print("Determining mus / psis")
        self.__get_psis_mus()

        print("Determining betas")
        self.__get_betas_vectorized()


    def __get_J(self):
        #print(np.cumsum(self.eigvals)/np.sum(self.eigvals))
        self.J = int(np.argmax(np.cumsum(self.eigvals)/np.sum(self.eigvals) > self.r))

    def __dists_to_kvals(self, dists):
        sq_dists = dists**2

        k_vals = np.exp(-1./(2. * self.width**2) * sq_dists)

        return k_vals


    def __compute_rbf_width(self):
        dists = spsp.distance.pdist(X, 'euclidean')

        if self.width is not None:
            return self.width, dists

        if self.width_rule == 'heuristic1':
            width = np.median(dists)
        elif self.width_rule == 'heuristic2':
            raise NotImplementedError
        elif self.width_rule == 'heuristic3':
            if self.g is not None:
                X_ = np.concatenate([self.X, self.g], axis=0)
                width = np.median(spsp.distance.pdist(X_, 'euclidean'))
            else:
                raise ValueError("Please specify gradient points to compute")

        return width, dists

    def __compute_gram(self):
        """
        Computes K, gram matrix
        """
        self.width, dists = self.__compute_rbf_width()

        k_vals = self.__dists_to_kvals(dists)

        K = np.zeros((self.num_samples,
                      self.num_samples))

        K[np.triu_indices(self.num_samples, 1)] = k_vals

        K = K + K.T
        np.fill_diagonal(K, 1)

        self.K = K


    def __compute_eigenvalues(self):
        if self.K is not None:
            eigvals, eigvecs = sp.linalg.eigh(self.K,
                                              eigvals=(num_samples-self.J, num_samples-1))

            idx_sort = np.argsort(eigvals)[::-1]

            eigvals = eigvals[idx_sort]

            eigvecs = eigvecs[:, idx_sort]

            self.eigvals = eigvals.real
            self.eigvecs = eigvecs.real

        else:
            raise Exception("Form gram matrix K first")

    def __get_psis_mus(self):
        if self.eigvecs is not None and self.eigvals is not None:

            self.mus = self.eigvals / self.num_samples
            self.psis = self.eigvecs * np.sqrt(self.num_samples)

        else:
            raise Exception("Compute eigenvalues and eigenvectors of gram matrix first")



    def __get_k_vec(self, x):
        """
        For a given x, computes all K(x, x_m)
        """

        dists = spsp.distance.cdist(self.X, x.reshape(1, -1), 'euclidean').flatten()
        width = np.median(dists)
        k_vals = self.__dists_to_kvals(dists)

        return k_vals

    def grad_jth_psi(self, j, x):
        if not 1 <= j <= self.num_samples or not isinstance(j, int):
            raise Exception(f"Invalid j. Must be and integer between 1 and {self.num_samples}")

        broadcasted_subtraction = (x - self.X).T

        k_vals = self.__get_k_vec(x)

        broadcasted_multiplication = k_vals * broadcasted_subtraction

        partials_matrix = -1./self.width**2 * broadcasted_multiplication

        eigvec = self.eigvecs[:, j-1].reshape(-1, 1)
        eigvalue = self.eigvals[j-1]

        
        return (np.sqrt(self.num_samples) / eigvalue * (partials_matrix @ eigvec)).flatten()


    def grad_jth_psi_factory(self, j):
        def f(x):
            return self.grad_jth_psi(j, x)

        return f


    def jth_psi(self, j, x):
        if not 1 <= j <= self.num_samples or not isinstance(j, int):
            raise Exception(f"Invalid j. Must be and integer between 1 and {self.num_samples}")


        k_vals = self.__get_k_vec(x)

        eigvec = self.eigvecs[:, j-1].flatten()
        eigvalue = self.eigvals[j-1]

        return np.sqrt(self.num_samples) / eigvalue * np.dot(eigvec, k_vals)
        


    def jth_psi_factory(self, j):
        def f(x):
            return self.jth_psi(j, x)

        return f

    def jth_beta(self, j):
        if not 1 <= j <= self.num_samples or not isinstance(j, int):
            raise Exception(f"Invalid j. Must be and integer between 1 and {self.num_samples}")

        s = 0
        for i in range(self.num_samples):
            s = s + self.grad_jth_psi(j, self.X[i, :])

        b_j = (-1./self.num_samples) * s

        return b_j

    def __get_betas_vectorized(self):
        broadcasted_subtraction = self.X[:, np.newaxis] - self.X

        broadcasted_subtraction = np.rot90(broadcasted_subtraction, axes=(2, 1))
        broadcasted_subtraction = np.rot90(broadcasted_subtraction, axes=(0, 2))

        # result of this is a three dimensional array where
        # each 2d cross section k through the third dimension is the result of
        # broadcasting row-rise the subtraction of all data points from the
        # kth data point
        
        broadcasted_multiplication = self.K.reshape(self.num_samples, 1, self.num_samples) \
                                     * broadcasted_subtraction

        partials_matrix = -1./self.width**2 * broadcasted_multiplication


        # (dim, num_samples, num_samples)
        # (i, j, k)
        # (ith partial, jth sample, kth eigenvector)
        Kgrad_eigvecs_tensor_product = np.tensordot(partials_matrix,
                                                    self.eigvecs, axes=((0),
                                                                        (0)))

        psihats = np.sqrt(self.num_samples) * Kgrad_eigvecs_tensor_product / self.eigvals

        betas = -1 * np.mean(psihats, axis=1)

        self.betas = betas

    def gradient_estimate_vectorized(self, j, x):
        """
        x is an n-dimensional numpy array
        """

        if j is None:
            j = self.J

        if not 1 <= j <= self.num_samples or not isinstance(j, int):
            raise Exception(f"Invalid j. Must be and integer between 1 and {self.num_samples}")

        K = spsp.distance.cdist(x, self.X, 'sqeuclidean')
        K = np.exp(-1./(2. * self.width**2) * K)

        us = self.eigvecs[:, :j]
        lambdas = self.eigvals[:j]

        matmul = K @ us

        matmul = np.sqrt(num_samples) * matmul / lambdas


        b = self.betas[:, :j]

        print(matmul.shape)
        print(b.T.shape)


        g = matmul @ b.T

        return g.flatten()

        



    def gradient_estimate(self, j, x):
        if j is None:
            j = self.J


        if not 1 <= j <= self.num_samples or not isinstance(j, int):
            raise Exception(f"Invalid j. Must be and integer between 1 and {self.num_samples}")

        

        g = 0
        for jj in range(1, j+1):
            b_j = self.jth_beta(jj)
            psi_j = self.jth_psi(jj, x)

            g = g + b_j * psi_j

        return g


if __name__ == '__main__':
    num_eigvecs = 6
    num_samples = 10000

    x = np.linspace(-5, 5, num_samples)

    X = np.random.randn(num_samples, 1)
    #X = pd.read_csv("./data/paper_toy.csv", header=None).values

    print(X.shape)
    ssge = SSGE(X,
                g=x.reshape(-1, 1),
                J=num_eigvecs,
                width_rule='heuristic3',
                r=0.99999)

    g = ssge.gradient_estimate_vectorized(num_eigvecs,
                                          x.reshape(-1, 1))

    # vals = []
    # for xx in x:
    #     vals.append(ssge.gradient_estimate(num_eigvecs, np.array([xx])))

    #print(ssge.eigvals)
    #print(np.cumsum(ssge.eigvals)/np.sum(ssge.eigvals))

    fig, ax = plt.subplots(1,1)


    ax.plot(x, normal_log_density(x))
    ax.plot(x, normal_log_density_deriv(x))
    #ax.plot(x, vals)
    ax.plot(x, g)

    ax.scatter(X.flatten(), np.zeros(num_samples).flatten(), marker='x')


    plt.show()
    plt.close() 


