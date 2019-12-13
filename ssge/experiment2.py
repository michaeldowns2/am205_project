"""
In this experiment, we assess how well the
SSGE performs for a 1D Gaussian Mixture as we vary different
Gaussian Mixture parameters
"""

from ssge import SSGE
from gmm import *
import jax.numpy as jnp
from jax.scipy import stats as jsps
from jax import grad, vmap
import matplotlib.pyplot as plt
import numpy as np
import random


def main():

    # Set random seed
    np.random.seed(207)
    random.seed(207)

    # Generate samples
    num_samples = 1000

    # weights = [1/2, 1/2]
    # mus = [jnp.array([-6]),
    #        jnp.array([6])]
    # sigmas = [jnp.eye(1),
    #           jnp.eye(1)]

    # weights = [1/3, 1/3, 1/3]
    # mus = [jnp.array([-6]),
    #        jnp.array([0]),
    #        jnp.array([6])]
    # sigmas = [jnp.eye(1),
    #           jnp.eye(1),
    #           jnp.eye(1)]

    # weights = [1/5, 1/5, 1/5, 1/5, 1/5]
    # mus = [jnp.array([-6]),
    #        jnp.array([-3]),
    #        jnp.array([0]),
    #        jnp.array([3]),
    #        jnp.array([6])]
    # sigmas = [jnp.eye(1),
    #           jnp.eye(1),
    #           jnp.eye(1),
    #           jnp.eye(1),
    #           jnp.eye(1)]

    weights = [1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7]
    mus = [jnp.array([-7.5]),
           jnp.array([-5]),
           jnp.array([-2.5]),
           jnp.array([0]),
           jnp.array([2.5]),
           jnp.array([5]),
           jnp.array([7.5]),]
    sigmas = [jnp.eye(1),
              jnp.eye(1),
              jnp.eye(1),
              jnp.eye(1),
              jnp.eye(1),
              jnp.eye(1),
              jnp.eye(1)]

    # Draw samples from the distribution
    samples = sample_gmm(num_samples=num_samples,
                         weights=weights,
                         mus=mus,
                         sigmas=sigmas)


    # Define x-axis points
    xs = jnp.linspace(-10, 10, num_samples)

    # Generate SSGE (with selected number of eigenfunctions)
    num_eigvecs = 15
    ssge = SSGE(samples,
                g=xs.reshape(-1, 1),
                J=num_eigvecs,
                width_rule='heuristic3',
                r=0.99999)

    # Generate SSGE GLD estimate
    g = ssge.gradient_estimate_vectorized(num_eigvecs, xs.reshape(-1, 1))

    # Generate PDF and analytic GLD
    pdf = gmm_pdf_vectorized(weights, mus, sigmas)
    gld = gmm_gld(weights, mus, sigmas)
    gld_vals = gld(xs)

    # Find MSE
    MSE = np.mean((g - gld_vals)**2)
    print(f'MSE: {MSE}')

    # Plot
    plt.style.use('seaborn-darkgrid')
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(xs, pdf(xs.reshape(-1, 1)),
            label='PDF')
    ax.plot(xs, gld_vals, label='Analytic GLD')
    ax.plot(xs, g, label="SSGE GLD")
    ax.scatter(samples.flatten(), np.zeros(num_samples),
               marker='|',
               label='Samples',
               color='C4')
    ax.set(xlim=(-10, 10), ylim=(-1, 1))
    fig.tight_layout()
    plt.legend(loc='upper right', fontsize=14)

    # Uncomment this to plot PDF on its own mini-plot within larger plot
    # inner_ax = fig.add_axes([0.15, 0.1, 0.33, 0.15])
    # inner_ax.plot(xs, pdf(xs.reshape(-1, 1)), label='PDF')
    # inner_ax.scatter(samples.flatten(), np.zeros(num_samples),
    #            color='C4',
    #            marker='|',
    #            label='Samples')
    # inner_ax.set_title("Scaled PDF", fontsize=14)
    # inner_ax.set(xlim=(-10, 10), ylim=(-0.1, 0.5),
    #              yticks=[0, 0.5], xticks=[-10, -5, 0, 5, 10], frame_on=True)


    plt.show()
    plt.close()



if __name__ == '__main__':
    main()
