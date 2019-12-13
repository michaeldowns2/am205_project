
import numpy as np
import random
import jax.numpy as jnp
from ssge import *
from jax import grad, vmap
import matplotlib.pyplot as plt
from scipy.stats import triang
import jax.random as jrand

np.random.seed(200)
random.seed(200)

def sample_triangle(num_samples):
    """
    ======
    CHECK THE PARAMS of the SAMPLING fn and the PDF/GLD PLOTTING fn!
    (They have diff params so need to make sure all is in order)
    ======
    """
    result = np.random.triangular(-1, 0, 1, num_samples)

    return np.array([result]).T


if __name__ == '__main__':

    # Generate samples
    num_samples = 500
    samples = sample_triangle(num_samples=num_samples)

    xs = jnp.linspace(-0.99999999, 0.99999999, num_samples)

    print("INPUT SAMPLES SHAPE", samples.shape)
    # Generate SSGE
    num_eigvecs = 5
    ssge = SSGE(samples,
                g=xs.reshape(-1, 1),
                J=num_eigvecs,
                width_rule='heuristic3',
                r=0.99999)
    g = ssge.gradient_estimate_vectorized(num_eigvecs, xs.reshape(-1, 1))


    # Generate PDF and GLD
    triang_pdf = triang.pdf(xs.reshape(-1, 1), c=0.5, loc=-1, scale=2)
    triang_gld = triang.logpdf(xs, c=0.5, loc=-1, scale=2)

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