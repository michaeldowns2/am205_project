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

their_samples = np.array([-1.3978306,
                          -0.1656294,
                          -0.7687699,
                          1.3277539,
                          -1.348662,
                          -0.01482771,
                          1.214632,
                          0.0299258,
                          -0.14295717,
                          -1.1633193,
                          1.116604,
                          0.8392043,
                          -0.6236486,
                          0.4726939,
                          -0.03785022,
                          0.7830375,
                          1.0958755,
                          -0.8629319,
                          -0.36036167,
                          -1.3055072,
                          0.33276156,
                          -1.9405675,
                          0.44604087,
                          -1.9271216,
                          -1.1683466,
                          0.17335449,
                          -0.5707963,
                          0.7628957,
                          1.1169623,
                          -1.0130662,
                          -0.6747647,
                          2.2576284,
                          1.3528007,
                          -0.5253186,
                          -0.57783794,
                          2.711842,
                          -0.7338472,
                          -1.2018663,
                          0.7715226,
                          0.44492066,
                          0.33246025,
                          -0.09360338,
                          -0.40163088,
                          -0.89535093,
                          0.4709173,
                          0.5012277,
                          2.5456474,
                          1.3322248,
                          2.3658423,
                          -0.9944531,
                          0.68742377,
                          0.38139537,
                          0.66727203,
                          1.8851165,
                          -0.40677354,
                          -3.0937572,
                          0.24492185,
                          -0.37867954,
                          -1.7278725,
                          -0.95450586,
                          0.46903977,
                          0.81732416,
                          -1.4967151,
                          0.06762022,
                          0.48230958,
                          -1.5986868,
                          -1.7150725,
                          -0.89085317,
                          -0.0355133,
                          -0.81800085,
                          0.02615927,
                          0.59442765,
                          -0.16194372,
                          -0.02844677,
                          0.2973782,
                          -2.4878414,
                          0.39310828,
                          -0.06973678,
                          2.1025054,
                          -1.7462497,
                          0.2158863,
                          -0.4426719,
                          0.2397488,
                          0.28329995,
                          0.83471346,
                          1.148953,
                          1.3617909,
                          0.8892179,
                          1.0943977,
                          1.5121077,
                          -0.52893436,
                          -0.77820534,
                          0.08577114,
                          0.10160784,
                          -1.1910279,
                          -0.53252566,
                          1.0272863,
                          -0.8172735,
                          0.5415074,
                          1.1359246])


def k_rbf_1d(x1, x2, h):
    return np.exp((-1./h) * (x1 - x2) **2)

def k_rbf_deriv_1d(x1, x2, h):
    return (-1./h) * 2 * (x1 - x2) * k_rbf_1d(x1, x2, h)

def k_rbf(x1, x2, width):
    return np.exp((-1./(2.*width**2)) * np.linalg.norm(x1 - x2, 2)**2 )


def normal_log_density(x):
    return -1./2 * np.log(2*np.pi) - 1./2 * x**2

def normal_log_density_deriv(x):
    return -1. * x

class SSGE:
    def __init__(self, X,
                 g=None,
                 kernel='rbf',
                 width=None, r=0.99):
        self.K = None
        self.eigvals = None
        self.eigvecs = None
        self.width = width
        self.betas = {}
        self.g = g


        self.r = r

        self.X = X

        self.num_samples = len(X)

        print("Computing gram matrix")
        self.__compute_gram()

        print("Computing eigenvalues / vectors of gram matrix")
        self.__compute_eigenvalues()

        print("Computing J")
        self.__get_J()
        print(f"J: {self.J}")

        print("Determining mus / psis")
        self.__get_psis_mus()


    def __get_J(self):
        self.J = np.argmax(np.cumsum(self.eigvals)/np.sum(self.eigvals) > self.r)

    def __dists_to_kvals(self, dists):
        sq_dists = dists**2

        k_vals = np.exp(-1./(2. * self.width**2) * sq_dists)

        return k_vals


    def __compute_gram(self):
        """
        Computes K, gram matrix
        """

        if self.g is not None:
            X_ = np.concatenate([self.X, self.g], axis=0)
            width = np.median(spsp.distance.pdist(X_, 'euclidean'))


        dists = spsp.distance.pdist(X, 'euclidean')

        #width = np.median(dists)
        if self.width is None:
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

        if j in self.betas:
            return self.betas[j]
        else:
            s = 0
            for i in range(self.num_samples):
                s = s + self.grad_jth_psi(j, self.X[i, :])

            b_j = (-1./self.num_samples) * s

            self.betas[j] = b_j

        return b_j

    def gradient_estimate(self, j, x):
        if not 1 <= j <= self.num_samples or not isinstance(j, int):
            raise Exception(f"Invalid j. Must be and integer between 1 and {self.num_samples}")

        g = 0
        for jj in range(1, j+1):
            b_j = self.jth_beta(jj)
            psi_j = self.jth_psi(jj, x)

            g = g + b_j * psi_j

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

    num_eigvecs = 6
    num_samples = 100

    x = np.linspace(-5, 5, num_samples)

    #X = np.random.randn(num_samples).reshape(-1, 1)
    X = their_samples.reshape(-1, 1)
    
    ssge = SSGE(X, g=x.reshape(-1, 1))

    vals = []
    for xx in x:
        vals.append(ssge.gradient_estimate(num_eigvecs, np.array([xx])))

    # print(ssge.eigvals)
    # print(np.cumsum(ssge.eigvals)/np.sum(ssge.eigvals))

    fig, ax = plt.subplots(1,1)


    ax.plot(x, normal_log_density(x))
    ax.plot(x, normal_log_density_deriv(x))
    ax.plot(x, vals)

    ax.scatter(X.flatten(), np.zeros(num_samples).flatten(), marker='x')


    plt.show()
    plt.close()

