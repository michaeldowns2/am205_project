"""
In this experiment, we recreate the SSGE
and assess our implementation, as well as
vary parameters to 'stress-test' the
SSGE
"""

from ssge import *
import random
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for our sampling method
random.seed(205)
np.random.seed(205)


if __name__ == '__main__':
    # Select the number of eigenfunctions and samples
    # For their data, the optimal number of eigenfunctions is 6
    num_eigvecs = 6
    num_samples = 100

    # Select the range over which to find the PDF, analytic GLD, and SSGE GLD
    x = np.linspace(-5, 5, num_samples)


    # Here we set the sample data. If using their implementation, use 'their_samples'
    # If using a random distribution, use the np.random.randn function
    # X = np.random.randn(num_samples, 1)
    their_samples = np.array([[-3.0937572,  -2.4878414,  -1.9405675,  -1.9271216,  -1.7462497,  -1.7278725,
         -1.7150725,  -1.5986868,  -1.4967151,  -1.3978306,  -1.348662,   -1.3055072,
         -1.2018663,  -1.1910279,  -1.1683466,  -1.1633193,  -1.0130662,  -0.9944531,
         -0.95450586, -0.89535093, -0.89085317, -0.8629319,  -0.81800085, -0.8172735,
         -0.77820534, -0.7687699,  -0.7338472,  -0.6747647,  -0.6236486,  -0.57783794,
         -0.5707963,  -0.53252566, -0.52893436, -0.5253186,  -0.4426719,  -0.40677354,
         -0.40163088, -0.37867954, -0.36036167, -0.1656294,  -0.16194372, -0.14295717,
         -0.09360338, -0.06973678, -0.03785022, -0.0355133,  -0.02844677, -0.01482771,
         0.02615927,  0.0299258,   0.06762022,  0.08577114,  0.10160784,  0.17335449,
         0.2158863,   0.2397488,   0.24492185,  0.28329995,  0.2973782,   0.33246025,
         0.33276156,  0.38139537,  0.39310828,  0.44492066,  0.44604087,  0.46903977,
         0.4709173,   0.4726939,   0.48230958,  0.5012277,   0.5415074,   0.59442765,
         0.66727203,  0.68742377,  0.7628957,   0.7715226,   0.7830375,   0.81732416,
         0.83471346,  0.8392043,   0.8892179,   1.0272863,   1.0943977,   1.0958755,
         1.116604,    1.1169623,   1.1359246,   1.148953,    1.214632,   1.3277539,
         1.3322248,   1.3528007,   1.3617909,   1.5121077,   1.8851165,   2.1025054,
         2.2576284,   2.3658423,   2.5456474,   2.711842]]).T


    # Use this to calculate the MSE between our SSGE implementation and theirs
    their_stein_result = np.array([4.84420156e+00,  4.84092236e+00,  4.81947899e+00,  4.78003168e+00,
                           4.72299194e+00,  4.64898872e+00,  4.55888891e+00,  4.45376396e+00,
                           4.33489370e+00,  4.20373869e+00,  4.06189299e+00,  3.91110778e+00,
                           3.75320983e+00,  3.59008622e+00,  3.42365265e+00,  3.25580382e+00,
                           3.08838415e+00,  2.92315412e+00,  2.76172471e+00,  2.60558367e+00,
                           2.45600510e+00,  2.31406641e+00,  2.18060207e+00,  2.05622053e+00,
                           1.94124746e+00,  1.83579183e+00,  1.73968267e+00,  1.65253830e+00,
                           1.57373917e+00,  1.50248623e+00,  1.43780446e+00,  1.37859845e+00,
                           1.32365727e+00,  1.27172017e+00,  1.22150624e+00,  1.17175663e+00,
                           1.12125373e+00,  1.06887412e+00,  1.01362956e+00,  9.54662800e-01,
                           8.91287506e-01,  8.23008358e-01,  7.49529481e-01,  6.70732021e-01,
                           5.86716831e-01,  4.97752637e-01,  4.04290527e-01,  3.06919366e-01,
                           2.06335217e-01,  1.03381924e-01, -1.09687936e-03, -1.06174782e-01,
                           -2.10966632e-01, -3.14590931e-01, -4.16256040e-01, -5.15245557e-01,
                           -6.10964119e-01, -7.02967882e-01, -7.90929437e-01, -8.74710500e-01,
                           -9.54314172e-01, -1.02990770e+00, -1.10180819e+00, -1.17046261e+00,
                           -1.23644650e+00, -1.30041528e+00, -1.36310899e+00, -1.42529285e+00,
                           -1.48775446e+00, -1.55126655e+00, -1.61654353e+00, -1.68423879e+00,
                           -1.75488698e+00, -1.82894039e+00, -1.90667677e+00, -1.98823249e+00,
                           -2.07357860e+00, -2.16254187e+00, -2.25475979e+00, -2.34971571e+00,
                           -2.44675398e+00, -2.54506254e+00, -2.64374709e+00, -2.74178696e+00,
                           -2.83808637e+00, -2.93152380e+00, -3.02093148e+00, -3.10516286e+00,
                           -3.18308663e+00, -3.25362897e+00, -3.31579542e+00, -3.36869168e+00,
                           -3.41152787e+00, -3.44364524e+00, -3.46452522e+00, -3.47380543e+00,
                           -3.47126341e+00, -3.45684505e+00, -3.43063116e+00, -3.39286232e+00])

    # create an instantiation of the ssge class
    ssge = SSGE(their_samples,
                g=x.reshape(-1, 1),
                J=num_eigvecs,
                width_rule='heuristic3',
                r=0.99999)

    # get the SSGE estimate for the gradient log density (GLD)
    g = ssge.gradient_estimate_vectorized(num_eigvecs,
                                          x.reshape(-1, 1))


    # Calculate the MSE between the analytic solution and our SSGE estimate
    sq_error = (g - normal_log_density_deriv(x))**2
    MSE = np.mean(sq_error)
    print(f'MSE: {MSE}')

    # Calculate the MSE between the two SSGE implementations
    result_sq_error = (g - their_stein_result)**2
    result_MSE = np.mean(result_sq_error)
    print(f'MSE between our SSGE and theirs: {result_MSE}')


    # Plot results

    plt.style.use('seaborn-darkgrid')
    fig, ax = plt.subplots(1,1)


    ax.plot(x, normal_log_density(x), label=r"$\log\:q(x)$")
    ax.plot(x, normal_log_density_deriv(x), label=r"$\nabla_x\log\:q(x)$")
    #ax.plot(x, vals)
    ax.plot(x, g, label=r"$\hat{\nabla}_x\log\:q(x)$, Spectral")

    ax.scatter(their_samples.flatten(), np.zeros(num_samples).flatten(), marker='|', label="Samples")

    plt.legend(loc='lower center')
    plt.show()
    plt.close()

