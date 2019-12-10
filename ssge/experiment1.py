"""
1D multimodal distribution
"""

from ssge import SSGE

import jax.numpy as jnp
from jax.scipy import stats as jsps
from jax import grad, vmap

import matplotlib.pyplot as plt
import numpy as np



def gmm_pdf(x, weights, mus, sigmas):

    pdfval = 0

    for weight, mu, sigma in zip(weights, mus, sigmas):
        pdfval = pdfval + weight * jsps.norm.pdf(x, mu, sigma)

    return pdfval


def log_gmm_pdf(x, weights, mus, sigmas):
    return jnp.log(gmm_pdf(x, weights, mus, sigmas))

def sample_gmm(num_samples, weights, mus, sigmas):
    num_pdfs = len(weights)

    mask = np.random.multinomial(1, weights, size=num_samples)

    all_samples = np.random.normal(loc=mus,
                                   scale=sigmas,
                                   size=(num_samples, num_pdfs))

    masked_samples = mask * all_samples

    return masked_samples.sum(axis=1).flatten()


def main():
    num_samples = 100
    weights = [0.3, 0.7]
    mus = [-4, 4]
    sigmas = [1, 1]


    samples = sample_gmm(num_samples=100,
                         weights=weights,
                         mus=mus,
                         sigmas=sigmas)

    deriv_grad_log_density = grad(log_gmm_pdf, 0)

    fig, ax = plt.subplots(1,1)

    

    x = jnp.linspace(-10, 10, 1000)
    ax.plot(x, gmm_pdf(x, weights, mus, sigmas),
            label='Analytic PDF')

    deriv_func = lambda x: deriv_grad_log_density(x, weights, mus, sigmas)
    ax.plot(x, list(map(deriv_func, x)), label='Analytic Density')

    ax.scatter(samples, np.zeros(num_samples),
               color='red',
               marker='x',
               label='Samples')

    ax.legend()

    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
