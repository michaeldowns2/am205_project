"""
1D multimodal distribution
"""

from ssge import SSGE

import numpy as np

def gmm_pdf(x, weights, mus, sigmas):
    pass

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

    samples = sample_gmm(num_samples=100,
                         weights=[0.3, 0.7],
                         mus=[-3, 4],
                         sigmas=[0.5, 2])

    print(samples)




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


if __name__ == '__main__':
    main()
