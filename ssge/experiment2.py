"""
In this experiment we test how well SSGE is able to
approximate the GLD in a distribution with correlation

Kolmogorov-Smirnov
Fréchet distance
Hausdorff distance
"""
import random

import jax.numpy as jnp
from jax.scipy import stats as jsps
from jax import grad, vmap
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from gmm import (sample_gmm,
                 gmm_pdf_vectorized,
                 gmm_gld)

from ssge import SSGE


random.seed(205)
np.random.seed(205)



def pdf2d(pdf, num_points, x_min, x_max, y_min, y_max):
    x = np.linspace(x_min, x_max, num_points)
    y = np.linspace(y_min, y_max, num_points)

    X, Y = np.meshgrid(x, y)

    x = X.flatten()
    y = Y.flatten()

    xy = np.concatenate((x.reshape(-1, 1),
                         y.reshape(-1, 1)), axis=1)


    z = pdf(xy)

    Z = z.reshape(num_points, num_points)

    return X, Y, Z, xy

def grad2d(gld, num_points, x_min, x_max, y_min, y_max):
    x = np.linspace(x_min, x_max, num_points)
    y = np.linspace(y_min, y_max, num_points)

    X, Y = np.meshgrid(x, y)

    x = X.flatten()
    y = Y.flatten()

    xy = np.concatenate((x.reshape(-1, 1),
                         y.reshape(-1, 1)), axis=1)

    UV = gld(xy)

    u = UV[:, 0].flatten()
    v = UV[:, 1].flatten()

    return x, y, u, v, xy


def mse(U, V, U_esimate, V_estimate):
    UV = np.concatenate((U,V), axis=0)
    UV_estimate = np.concatenate((U_esimate, V_estimate), axis=0)

    return np.mean((UV - UV_estimate)**2)

if __name__ == '__main__':
    num_samples = 1000
    weights = [1.]
    mus = [jnp.array([0, 0])]
    sigmas = [jnp.eye(2)]

    samples = sample_gmm(num_samples=num_samples,
                         weights=weights,
                         mus=mus,
                         sigmas=sigmas)

    print(samples)
    pdf = gmm_pdf_vectorized(weights, mus, sigmas)
    gld = gmm_gld(weights, mus, sigmas)

    X_pdf, Y_pdf, Z_pdf, xy_pdf = pdf2d(pdf, 100, -3, 3, -3, 3)
    X_gld, Y_gld, U_gld, V_gld, xy_grad = grad2d(gld,
                                        10,
                                        -3,
                                        3,
                                        -3,
                                        3)


    print(samples)
    print(xy_grad)

    num_eigvecs=40
    ssge = SSGE(samples,
                g=xy_grad,
                J=num_eigvecs,
                width_rule='heuristic3',
                r=0.99999)

    gld_estimate = ssge.gradient_estimate_vectorized(num_eigvecs,
                                                     xy_grad)

    U_gld_estimate = gld_estimate[:, 0]
    V_gld_estimate = gld_estimate[:, 1]


    print(mse(U_gld,
              V_gld,
              U_gld_estimate,
              V_gld_estimate))

    fig, ax = plt.subplots(1,1)

    ax.quiver(X_gld, Y_gld, U_gld, V_gld,
              units='xy',
              scale=10,
              headwidth=2,
              alpha=0.5,
              color='green',
              label='Analytic Gradient')

    ax.quiver(X_gld, Y_gld,
              U_gld_estimate, V_gld_estimate,
              units='xy',
              scale=10,
              headwidth=2,
              alpha=0.5,
              color='purple',
              label='Estimated Gradient')


    ax.contour(X_pdf, Y_pdf, Z_pdf, color='blue', label='Analytic PDF')

    ax.scatter(samples[:, 0].flatten(),
               samples[:, 1].flatten(),
               color='red',
               marker='x',
               label='Samples')

    ax.legend()

    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    

    plt.show()
    plt.close()


