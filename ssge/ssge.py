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
                 r=0.99,
                 verbose=0):

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

        if verbose > 0: print("Computing gram matrix")
        self.__compute_gram()

        if verbose > 0: print("Computing eigenvalues / vectors of gram matrix")
        self.__compute_eigenvalues()

        if verbose > 0: print("Computing J")
        self.__get_J()
        if verbose > 0: print(f"J: {self.J}")

        if verbose > 0: print("Determining mus / psis")
        self.__get_psis_mus()

        if verbose > 0: print("Determining betas")
        self.__get_betas_vectorized()


    def __get_J(self):
        #print(np.cumsum(self.eigvals)/np.sum(self.eigvals))
        self.J = int(np.argmax(np.cumsum(self.eigvals)/np.sum(self.eigvals) > self.r))

    def __dists_to_kvals(self, dists):
        sq_dists = dists**2

        k_vals = np.exp(-1./(2. * self.width**2) * sq_dists)

        return k_vals


    def __compute_rbf_width(self):
        dists = spsp.distance.pdist(self.X, 'euclidean')

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
                                              eigvals=(self.num_samples-self.J, self.num_samples-1))

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
        """
        Suppose that X is of dimensionality Mxn (M data points of
        dimensionality n)
        The result of this function is an i x j matrix of the SSGE betas
        where the ijth element corresponds to the beta for the ith partial
        derivative for the jth psi (eigenfunction estimate)

        """

        # The computation for the ijth beta involves averaging
        # the ith partial derivative of the estimated jth psi over all M
        # data points. The derivative of the jth psi estimate is taken
        # on the kernel function (see (6)). The partial derivative of
        # the ith x for the rbf kernel is
        # (-1./self.width**2 * (x_i - xm_i)) * rbf(x, xm)
        # the line below simultaneously computes all (x_i - xm_i)
        # for all sampled data points used in constructing the
        # SSGE estimate.

        # self.X[:, np.newaxis] yields a 3d numpy array of dimensionality
        # (Mx1xn)
        # self.X[:, np.newaxis] - 1 yields a 3d numpy array of dimensionality
        # (MxMx2) where the cross section [k, :, :] yields
        # the result of broadcasting subtraction of X from
        # the kth data point. Imagining this array as a
        # 3D rectangle This cross section corresponds to  the
        # first square on the vertical stack of squares

        # the 1D cross section [0, 0, :] (should be all zeros) because
        # this corresponds to the first data point subtracted from
        # itself. imagine taking your left pointer finger and placing it
        # on the left side of the top part of the 3d rectangle. that's what
        # this cross section would correspond to. 
        broadcasted_subtraction = self.X[:, np.newaxis] - self.X

        # we would like the cross section [:, :, k] to correspond to
        # the broadcasted subtraction above and this can be achieved through
        # rotating the 3D numpy array twice 90 degrees around two
        # different axes

        # the first rotation involves taking the 3D rectangle and rotating
        # it clockwise around the axis normal to the XY plane
        # (imagine rotating the y unit vector into the x unit vector)
        # we then want to rotate it 90 degrees counterclockwise around the
        # axis normal to the YZ plane (imagine rotating tbe y unit vector into
        # the z unit vector)
        # that's what the two rotations below are doing

        broadcasted_subtraction = np.rot90(broadcasted_subtraction, axes=(2, 1))
        broadcasted_subtraction = np.rot90(broadcasted_subtraction, axes=(0, 2))

        # we now have a 3D numpy array where the cross section [:, :, k]
        # is the broadcasted operation of subtracting X from the
        # kth data point.


        # we then need to broadcast the kernel matrix to the above matrix
        # this can be achieved by rotating the 2D kernel matrix
        # counterclockwise around an axis normal to the XY plane
        # and then broadcasting multiplicating of each
        # 1D cross section oF the rotated K into each 2D cross section of
        # the matrix with the broadcasted subtractions.
        # element (i, 1, j) of the now rotated K contains
        # the kernel computation K(x_i, x_j)
        
        broadcasted_multiplication = self.K.reshape(self.num_samples, 1, self.num_samples) \
                                     * broadcasted_subtraction

        # we then broadcast multiply the scalar -1. / self.width**2
        # to obtain the matrix that has all partial derivatives for the kernel
        # across all data points.
        partials_matrix = -1./self.width**2 * broadcasted_multiplication


        

        # the equation below corresponds to performing the sum product
        # of the eigvenvector u with the vector where each element is
        # a kernel computation between an arbitrary x and the
        # mth data point. reference equation 6.

        # the result is a 3D numpy array where the i j kth element corresponds
        # to the ith partial derivative of the kth eigenvector estimate
        # of the jth sample point in the data

        # (dim, num_samples, num_samples)
        # (i, j, k)
        # (ith partial, jth sample, kth eigenvector)
        Kgrad_eigvecs_tensor_product = np.tensordot(partials_matrix,
                                                    self.eigvecs, axes=((0),
                                                                        (0)))

        # this is just broadcasting the appropriate coefficients in equation 6
        psihats = np.sqrt(self.num_samples) * Kgrad_eigvecs_tensor_product / self.eigvals

        # this is taking the mean across the samples to yield the beta matrix
        # see euqation (17)
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

        # this is computing the gradient estimate in (16) after
        # having determined the betas matrix
        # this computation involves determining
        # the jth psi function for j for all input data points
        # which involves forming a matrix of kernel computations

        # see the documentation for cdist.
        # basically, takes a collection of points and returns a matrix
        # where the ijth element is the distance from the
        # ith point in the first set to the jth point in the second set
        K = spsp.distance.cdist(x, self.X, 'sqeuclidean')

        # perform the RBF kernel computations
        K = np.exp(-1./(2. * self.width**2) * K)

        # we now have a matrix where the ijth element is the
        # kernel computation of the ith input data point with the
        # jth sample used to make the SSGE estimate 


        us = self.eigvecs[:, :j]
        lambdas = self.eigvals[:j]

        # this is performing the sum product in equation (6)
        # simultaneously for all input samples
        matmul = K @ us

        matmul = np.sqrt(self.num_samples) * matmul / lambdas


        b = self.betas[:, :j]

        # this is doing the computation for equation 16 simultaneously
        # on all input data points
        g = matmul @ b.T

        # the result is a matrix where each row is the estiamte of the
        # gradient log density at the input point for that row
        # 

        return g

        



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
    num_samples = 1000

    x = np.linspace(-5, 5, num_samples)

    X = np.random.randn(num_samples, 1)
    print("X shape", X.shape)
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
    #
    # # print(ssge.eigvals)
    # # print(np.cumsum(ssge.eigvals)/np.sum(ssge.eigvals))
    #
    # total_squared_error_vec = (np.array(g) - np.array([normal_log_density_deriv(x)]).T) ** 2
    # MSE = np.mean(total_squared_error_vec)
    # print(f'MSE: {MSE}')


    fig, ax = plt.subplots(1,1)


    ax.plot(x, normal_log_density(x), label="Log density")
    ax.plot(x, normal_log_density_deriv(x), label="Log density deriv")
    #ax.plot(x, vals)
    ax.plot(x, g, label="Our SSGE")

    ax.scatter(X.flatten(), np.zeros(num_samples).flatten(), marker='x')

    plt.legend()
    plt.show()
    plt.close() 


