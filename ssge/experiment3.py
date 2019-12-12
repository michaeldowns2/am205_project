"""
In this experiment we test how well SSGE is able to
approximate the GLD in a distribution with correlation.

We assess the simplest case which is a 2D Gaussian

1. Plot: Rsquared of best SSGE vs correlation for unimodal 2D Gaussian

Result: SSGE is able to deal with correlation reasonably well
in this simple case. Future work would include more 


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

    UV_mean = np.mean(UV)


    UV_estimate = np.concatenate((U_esimate, V_estimate), axis=0)


    mean_sq_error_ssge = np.mean((UV - UV_estimate)**2)
    mean_sq_error_avg = np.mean((UV - UV_mean)**2)


    rsq = 1 - mean_sq_error_ssge/mean_sq_error_avg

    return mean_sq_error_ssge, rsq


def do_gaussian_plot(samples, X_gld, Y_gld, U_gld, V_gld,
                     U_gld_estimate, V_gld_estimate, X_pdf, Y_pdf,
                     Z_pdf, num_eigvecs, corr):

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


    C = ax.contour(X_pdf,
                   Y_pdf,
                   Z_pdf,
                   label='Analytic PDF',
                   levels=20)

    ax.scatter(samples[:, 0].flatten(),
               samples[:, 1].flatten(),
               color='red',
               marker='x',
               label='Samples')

    ax.legend()

    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_title("""Comparison of SSGE({}) estimate of GLD vs analytic GLD
       for bivariate guassian with corr={}
                    """.format(num_eigvecs, corr))
    

    plt.savefig('./plots/experiment2/corr{}_eigvecs{}.png'.format(corr, num_eigvecs))
    plt.close()



def train_ssge_and_get_mse(samples, num_eigvecs, pdf, gld):
    X_pdf, Y_pdf, Z_pdf, xy_pdf = pdf2d(pdf, 100, -3, 3, -3, 3)
    X_gld, Y_gld, U_gld, V_gld, xy_grad = grad2d(gld,
                                        10,
                                        -3,
                                        3,
                                        -3,
                                        3)

    ssge = SSGE(samples,
                g=xy_grad,
                J=num_eigvecs,
                width_rule='heuristic3',
                r=0.99999)

    gld_estimate = ssge.gradient_estimate_vectorized(num_eigvecs,
                                                     xy_grad)

    U_gld_estimate = gld_estimate[:, 0]
    V_gld_estimate = gld_estimate[:, 1]

    estimate_mse, rsq = mse(U_gld,
              V_gld,
              U_gld_estimate,
              V_gld_estimate)

    #print("MSE: {}".format(estimate_mse))

    return ssge, estimate_mse, rsq, X_pdf, Y_pdf, Z_pdf, xy_pdf, X_gld, Y_gld, U_gld, V_gld, xy_grad, U_gld_estimate, V_gld_estimate



def test_isotropic_gaussian():
    num_samples = 100
    weights = [1.]
    mus = [jnp.array([0, 0])]
    sigmas = [jnp.eye(2)]

    samples = sample_gmm(num_samples=num_samples,
                         weights=weights,
                         mus=mus,
                         sigmas=sigmas)

    pdf = gmm_pdf_vectorized(weights, mus, sigmas)
    gld = gmm_gld(weights, mus, sigmas)

    num_eigvecs = 3
    ssge, \
        ssge_mse, \
        rsq, \
        X_pdf, \
        Y_pdf, \
        Z_pdf, \
        xy_pdf, \
        X_gld, \
        Y_gld, \
        U_gld, \
        V_gld, \
        xy_grad, \
        U_gld_estimate, \
        V_gld_estimate = train_ssge_and_get_mse(samples,
                                                num_eigvecs,
                                                pdf,
                                                gld)


    do_gaussian_plot(samples,
                     X_gld,
                     Y_gld,
                     U_gld,
                     V_gld,
                     U_gld_estimate,
                     V_gld_estimate,
                     X_pdf,
                     Y_pdf,
                     Z_pdf, 3, 0)





def experiment2_unimodal_bivariate(num_samples=100,
                                   patience=5,
                                   mu_x=0,
                                   mu_y=0,
                                   sigma_x=1,
                                   sigma_y=1):

    correlations = jnp.linspace(-0.99, 0.99, 100)

    weights = [1.]
    mus = [jnp.array([mu_x, mu_y])]
    sigma_x = sigma_x
    sigma_y = sigma_y

    best_num_eigvecss = []
    best_mses = []
    best_ssges = []
    best_rsqs = []

    for corr in correlations:
        #print("Corr: {}".format(corr))
        sigmas = [jnp.array([
            [sigma_x**2, corr*sigma_x * sigma_y],
            [corr*sigma_x * sigma_y, sigma_y**2],
        ])]

        pdf = gmm_pdf_vectorized(weights, mus, sigmas)
        gld = gmm_gld(weights, mus, sigmas)

        samples = sample_gmm(num_samples=num_samples,
                         weights=weights,
                         mus=mus,
                         sigmas=sigmas)


    
        best_rsq = -np.inf
        best_num_eigvecs = 0
        best_mse = np.inf
        best_ssge = None
        patience_counter = patience
        
        for num_eigvecs in range(1, num_samples + 1):
            #print("num_eigvecs: {}".format(num_eigvecs))
            
            ssge, ssge_mse, rsq, X_pdf,Y_pdf,Z_pdf,xy_pdf,X_gld,Y_gld,U_gld,V_gld,xy_grad, U_gld_estimate, V_gld_estimate = train_ssge_and_get_mse(samples, num_eigvecs, pdf,gld)


            #if ssge_mse < best_mse:
            if rsq > best_rsq:
                best_rsq = rsq
                best_mse = ssge_mse
                best_ssge = ssge
                best_num_eigvecs = num_eigvecs

                patience_counter = patience
            elif ssge_mse > best_mse:
                patience_counter -= 1 


            if patience_counter == 0:
                break

        print("Corr: {} | # Eigvecs {}| MSE {} | RSQ {}" .format(
            corr, best_num_eigvecs, best_mse, best_rsq))

        best_num_eigvecss.append(best_num_eigvecs)
        best_mses.append(best_mse)
        best_ssges.append(best_ssge)
        best_rsqs.append(best_rsq)

        ssge, ssge_mse, rsq, X_pdf,Y_pdf,Z_pdf,xy_pdf,X_gld,Y_gld,U_gld,V_gld,xy_grad, U_gld_estimate, V_gld_estimate = train_ssge_and_get_mse(samples, best_num_eigvecs, pdf,gld)

        do_gaussian_plot(samples,
                     X_gld,
                     Y_gld,
                     U_gld,
                     V_gld,
                     U_gld_estimate,
                     V_gld_estimate,
                     X_pdf,
                     Y_pdf,
                             Z_pdf,
                             best_num_eigvecs, corr)


    fig, ax = plt.subplots(1,1)

    ax.plot(correlations, best_rsqs)

    ax.set_ylabel(r"SSGE $R^2$")
    ax.set_xlabel("Bivariate Normal Correlation")

    ax.set_title(r"SSGE $R^2$ vs Bivariate Normal Correlation")

    plt.savefig('./plots/experiment2/mse_vs_corr.png')
    plt.close()
    print(best_mses)

if __name__ == '__main__':
    test_isotropic_gaussian()
    experiment2_unimodal_bivariate(num_samples=100,
                                   mu_x=0,
                                   mu_y=0,
                                   sigma_x=1,
                                   sigma_y=1)
    
