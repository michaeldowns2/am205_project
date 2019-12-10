"""
1D multimodal distribution
"""

from ssge import SSGE

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps

def gmm_pdf(x, weights, mus, sigmas):

    pdfval = 0

    for weight, mu, sigma in zip(weights, mus, sigmas):
        pdfval += weight * sps.norm(mu, sigma).pdf(x)


    return pdfval

def gmm_grad_log_density(x, weights, mus, sigmas):
    pass

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
    mus = [-3, 4]
    sigmas = [1, 1]


    samples = sample_gmm(num_samples=100,
                         weights=weights,
                         mus=mus,
                         sigmas=sigmas)

    print(samples)

    fig, ax = plt.subplots(1,1)

    x = np.linspace(-10, 10, 1000)
    ax.plot(x, gmm_pdf(x, weights, mus, sigmas), label='pdf')
    ax.scatter(samples, np.zeros(num_samples))


    plt.show()
    plt.close()




    # num_eigvecs = 6
    # num_samples = 1000

    # x = np.linspace(-5, 5, num_samples)

    # X = np.random.randn(num_samples, 1)
    # #X = pd.read_csv("./data/paper_toy.csv", header=None).values

    # print(X.shape)
    # ssge = SSGE(X,
    #             g=x.reshape(-1, 1),
    #             J=num_eigvecs,
    #             width_rule='heuristic3',
    #             r=0.99999)

    # g = ssge.gradient_estimate_vectorized(num_eigvecs,
    #                                       x.reshape(-1, 1))

    # # vals = []
    # # for xx in x:
    # #     vals.append(ssge.gradient_estimate(num_eigvecs, np.array([xx])))

    # #print(ssge.eigvals)
    # #print(np.cumsum(ssge.eigvals)/np.sum(ssge.eigvals))

    # fig, ax = plt.subplots(1,1)


    # ax.plot(x, normal_log_density(x))
    # ax.plot(x, normal_log_density_deriv(x))
    # #ax.plot(x, vals)
    # ax.plot(x, g)

    # ax.scatter(X.flatten(), np.zeros(num_samples).flatten(), marker='x')


    # plt.show()
    # plt.close() 

# if __name__== '__main__':
#     main()
#
if __name__ == '__main__':
    num_eigvecs = 6
    num_samples = 1000

    x = np.linspace(-5, 5, num_samples)

    X = np.random.randn(num_samples, 1)
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

    #print(ssge.eigvals)
    #print(np.cumsum(ssge.eigvals)/np.sum(ssge.eigvals))

    # total_squared_error_vec = (np.array(vals) - np.array([normal_log_density_deriv(x)]).T) ** 2
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
