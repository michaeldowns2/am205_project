"""
In this experiment, we test the SSGE's
performance on the pathological case of
the Triangle distribution.
"""

import numpy as np
import random
import jax.numpy as jnp
from ssge import *
from jax import grad, vmap
import matplotlib.pyplot as plt
from scipy.stats import triang
import jax.random as jrand

# Set random seed
np.random.seed(200)
random.seed(200)

# Function to generate samples from a triangular distribution
# centered at 0, with the left vertex at -1 and the right vertex at 1.

# *Ensure the parameters given here produce the same distribution as
# the PDF generated using the parameters given
# to the  triang.pdf and triang.logpdf functions below

# **the parameters in each case refer to different values, so they will be different
def sample_triangle(num_samples):

    result = np.random.triangular(-1, 0, 1, num_samples)

    return np.array([result]).T

def gld_triang(xs):
    res = []
    for x in xs.flatten():
        print(x)
        if x <= 0:
            res.append(1/(1+x))
        else:
            res.append(-1/(1-x))

    return res

    



if __name__ == '__main__':

    # Generate samples
    num_samples = 500
    samples = sample_triangle(num_samples=num_samples)

    # Generate plot x-axis
    # Using (-1, 1) here will result in an MSE of inf
    xs = jnp.linspace(-0.9999, 0.9999, num_samples)

    # Generate SSGE (with selected number of eigenfunctions)
    num_eigvecs = 5
    ssge = SSGE(samples,
                g=xs.reshape(-1, 1),
                J=num_eigvecs,
                width_rule='heuristic3',
                r=0.99999)
    g = ssge.gradient_estimate_vectorized(num_eigvecs, xs.reshape(-1, 1))


    # Generate PDF and GLD
    # (Ensure PDF here correspond to same as sampling from above)
    triang_pdf = triang.pdf(xs.reshape(-1, 1), c=0.5, loc=-1, scale=2)
    #triang_gld = triang.logpdf(xs, c=0.5, loc=-1, scale=2)
    triang_gld = gld_triang(xs)

    # Find MSE
    MSE = np.mean((g - triang_gld) ** 2)
    print(f'MSE: {MSE}')

    # Plot
    plt.style.use('seaborn-darkgrid')
    fig, ax = plt.subplots(1,1)

    ax.plot(xs, triang_pdf,
            label='PDF')
    ax.plot(xs, triang_gld, label='Analytic GLD')
    ax.plot(xs, g, label="SSGE GLD")
    ax.scatter(samples.flatten(), np.zeros(num_samples),
               color='C4',
               marker='|',
               label='Samples')

    ax.set(xlim=(-10, 10), ylim=(-2, 2))
    ax.legend()
    plt.show()
    plt.close()
