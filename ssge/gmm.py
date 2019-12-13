import random

import jax.numpy as jnp
from jax.scipy import stats as jsps
from jax import grad, vmap
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp



#np.random.seed(205)
#random.seed(205)

def mvn_pdf(x, mu, sigma):
    k = len(mu)
    

    term1 = (2*jnp.pi)**(-k/2)
    term2 = 1./jnp.sqrt(jnp.linalg.det(sigma))
    term3 = jnp.exp(-1./2 * (x - mu).T @ jnp.linalg.inv(sigma) @ (x - mu))

    return term1 * term2 * term3


def mvn_pdf_vectorized(x, mu, sigma):
    """
    assumes x is a num_samples x num_dimensions array.
    performs the density computations in a vectorized manner
    """
    k = len(mu)

    demeaned_x = x - mu

    first_prod = jnp.linalg.inv(sigma) @ demeaned_x.T

    second_prod = demeaned_x * first_prod.T

    reduction = np.sum(second_prod, axis=1).flatten()

    term1 = (2*jnp.pi)**(-k/2)
    term2 = 1./jnp.sqrt(jnp.linalg.det(sigma))
    term3 = jnp.exp(-1./2 * reduction)

    return term1 * term2 * term3 


def gmm_pdf(weights, mus, sigmas):
    def p(x):

        pdfval = 0

        for weight, mu, sigma in zip(weights, mus, sigmas):
            pdfval = pdfval + weight * mvn_pdf(x, mu, sigma)

        return pdfval

    return p

def gmm_pdf_vectorized(weights, mus, sigmas):
    def p(x):

        pdfval = 0

        for weight, mu, sigma in zip(weights, mus, sigmas):
            pdfval = pdfval + weight * mvn_pdf_vectorized(x, mu, sigma)

        return pdfval

    return p

def finite_diff(g, h, weights, mus, sigmas):
    def f(x):
        return (g(x + h, weights, mus, sigmas) - g(x, weights, mus, sigmas))/h

    return f


def log_gmm_pdf(weights, mus, sigmas):
    p = gmm_pdf(weights, mus, sigmas)

    def logp(x):
        return jnp.log(p(x))

    return logp

def log_gmm_pdf_vectorzed(weights, mus, sigmas):
    p = gmm_pdf_vectorized(weights, mus, sigmas)

    def logp(x):
        return jnp.log(p(x))

    return logp

def log_lik_gmm(weights, mus, sigmas ):
    p = gmm_pdf_vectorized(weights, mus, sigmas)

    def log_lik(x):
        return jnp.sum(jnp.log(p(x)))

    return log_lik


def gmm_gld(weights, mus, sigmas):
    logp = log_gmm_pdf(weights, mus, sigmas)

    return vmap(grad(logp))


def sample_gmm(num_samples, weights, mus, sigmas):
    """
    mus: list of means
    sigmas: list of covariance matrices
    """
    num_pdfs = len(weights)

    result = []
    for i in range(num_samples):
        # choose distribution
        idx = np.random.choice(range(num_pdfs), p=weights)

        mu = mus[idx]
        sigma = sigmas[idx]

        # sample from distribution
        sample = np.random.multivariate_normal(mu, sigma)

        result.append(sample)

    return np.array(result)


if __name__ == '__main__':
    num_samples = 100
    weights = [0.3, 0.7]
    mus = [jnp.array([-4]),
           jnp.array([4])]

    sigmas = [jnp.eye(1),
              jnp.eye(1)]


    samples = sample_gmm(num_samples=100,
                         weights=weights,
                         mus=mus,
                         sigmas=sigmas)

    print(samples)

    xs = jnp.linspace(-10, 10, 100)

    pdf = gmm_pdf_vectorized(weights, mus, sigmas)
    gld = gmm_gld(weights, mus, sigmas)

    print(gld(xs))

    fig, ax = plt.subplots(1,1)


    ax.plot(xs, pdf(xs.reshape(-1, 1)),
            label='Analytic PDF')

    # ax.plot(x, log_pdf(x),
    #         label='Analytic Log PDF')

    ax.plot(xs, gld(xs), label='Analytic GLD')

    ax.scatter(samples.flatten(), np.zeros(num_samples),
               color='red',
               marker='x',
               label='Samples')

    ax.legend()

    plt.show()
    plt.close()
