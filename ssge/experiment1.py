"""
In this experiment, we assess how well the
SSGE performs for a 1D Gaussian Mixture as we vary different
Gaussian Mixture parameters
"""

from ssge import SSGE

import matplotlib.pyplot as plt



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

    print("jax")
    deriv_func = lambda x: deriv_grad_log_density(x, weights, mus, sigmas)
    ax.plot(x, list(map(deriv_func, x)), label='Analytic GLD')

    ax.scatter(samples, np.zeros(num_samples),
               color='red',
               marker='x',
               label='Samples')

    ax.legend()

    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
