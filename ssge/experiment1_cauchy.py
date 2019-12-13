"""
In this experiment, we test the SSGE's
performance on the pathological case of
the Cauchy distribution.
"""

import numpy as np
import random
import jax.numpy as jnp
from ssge import *
from jax import grad, vmap
import matplotlib.pyplot as plt
from scipy.stats import cauchy
import jax.random as jrand

# Set random seed
np.random.seed(200)
random.seed(200)

# Function to sample from a Cauchy distribution
def sample_cauchy(num_samples):
    """
    """
    result = np.random.standard_cauchy(num_samples)

    return np.array([result]).T

# Function for the analytic solution of the Cauchy GLD
def cauchy_gld(x_vec):

    output_vec = (-2*x_vec) / (1 + x_vec**2)

    return output_vec


if __name__ == '__main__':

    # Generate samples
    num_samples = 500
    samples = sample_cauchy(num_samples=num_samples)

    # Generate plot x-axis
    xs = jnp.linspace(-10, 10, num_samples)

    # Generate SSGE (with selected number of eigenfunctions)
    num_eigvecs = 25
    ssge = SSGE(samples,
                g=xs.reshape(-1, 1),
                J=num_eigvecs,
                width_rule='heuristic3',
                r=0.99999)
    g = ssge.gradient_estimate_vectorized(num_eigvecs, xs.reshape(-1, 1))

    # Generate PDF and GLD
    cauchy_pdf = cauchy.pdf(xs.reshape(-1, 1))
    cauchy_gld = cauchy_gld(xs)

    # Find MSE
    MSE = np.mean((g - cauchy_gld) ** 2)
    print(f'MSE: {MSE}')

    # Plot
    plt.style.use('seaborn-darkgrid')
    fig, ax = plt.subplots(1,1)

    ax.plot(xs, cauchy_pdf,
            label='PDF')
    ax.plot(xs, cauchy_gld, label='Analytic GLD')
    ax.plot(xs, g, label="SSGE GLD")
    ax.scatter(samples.flatten(), np.zeros(num_samples),
               color='C4',
               marker='|',
               label='Samples')

    ax.set(xlim=(-10, 10), ylim=(-2, 2))
    ax.legend()
    plt.show()
    plt.close()