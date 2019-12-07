"""
"""
import logging
import random
import time


import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.spatial as spsp


#random.seed(205)
#np.random.seed(205)

def k_rbf_1d(x1, x2, h):
    return np.exp((-1./h) * (x1 - x2) **2)

def k_rbf_deriv_1d(x1, x2, h):
    return (-1./h) * 2 * (x1 - x2) * k_rbf_1d(x1, x2, h)

def k_rbf(x1, x2, width):
    return np.exp((-1./(2.*width**2)) * np.linalg.norm(x1 - x2, 2)**2 )


def normal_log_density(x):
    return -1./2 * np.log(2*np.pi) - 1./2 * x**2

def normal_score(x):
    return -1. * x

class SSGE:
    def __init__(self, X, kernel='rbf'):
        self.K = None
        self.eigvals = None
        self.eigvecs = None
        self.width = None

        self.X = X

        self.num_samples = len(X)

        print("Computing gram matrix")
        self.__compute_gram()

        print("Computing eigenvalues / vectors of gram matrix")
        self.__compute_eigenvalues()

        print("Determining mus / psis")
        self.__get_psis_mus()



    def __dists_to_kvals(self, dists):
        sq_dists = dists**2

        k_vals = np.exp(-1./(2. * self.width**2) * sq_dists)

        return k_vals


    def __compute_gram(self):
        """
        Computes K, gram matrix
        """

        dists = spsp.distance.pdist(X, 'euclidean')

        width = np.median(dists)
        self.width = width

        k_vals = self.__dists_to_kvals(dists)

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
            s += self.grad_jth_psi(j, self.X[i, :])

        return (-1./self.num_samples) * s

    def gradient_estimate(self, j, x):
        if not 1 <= j <= self.num_samples or not isinstance(j, int):
            raise Exception(f"Invalid j. Must be and integer between 1 and {self.num_samples}")

        g = 0
        for jj in range(1, j+1):
            b_j = self.jth_beta(jj)
            psi_j = self.jth_psi(jj, x)

            g += b_j * psi_j

        return g


if __name__ == '__main__':
    # dim = 3
    # num_samples = 1000
    # X = np.random.multivariate_normal(np.zeros(dim),
    #                                   np.eye(dim),
    #                                   num_samples)


    # print(len(X))
    # ssge = SSGE(X)

    # x = np.random.randn(dim)

    # print(ssge.jth_psi(1, x))

    # jth_psi = ssge.jth_psi_factory(1)
    
    # print(ssge.grad_jth_psi(1, x))

    # print(ssge.jth_beta(1))

    # print(ssge.gradient_estimate(1, x))
    # print(ssge.gradient_estimate(10, x))
    # print(ssge.gradient_estimate(30, x))
    # print(ssge.gradient_estimate(100, x))

    X = np.random.randn(100).reshape(-1, 1)
    x = np.linspace(-5, 5, 100)

    ssge = SSGE(X)

    vals = []
    for xx in x:
        vals.append(ssge.gradient_estimate(6, np.array([xx])))

    print(np.cumsum(ssge.eigvals)/np.sum(ssge.eigvals))

    fig, ax = plt.subplots(1,1)
    

    ax.plot(x, normal_log_density(x))
    ax.plot(x, normal_score(x))
    ax.plot(x, vals)
    

    plt.show()
    plt.close()
