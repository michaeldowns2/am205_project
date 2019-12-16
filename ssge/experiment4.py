"""
In thia experiment we assess the impact of using the SSGE estimate of the
GLD in a sampling procedure that makes use of that quantity
(Hamiltonian Monte Carlo).

We use log likelihood to assess how likely a set of samples are to come
from a distribution.

Not discussed in the writeup because we had enough content as-is.
"""

import time

from jax import grad, vmap
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from gmm import sample_gmm, log_lik_gmm
from ssge import SSGE


def K(p):
    D = len(p)
   # I = np.eye(len(p))
   # term1 = 1/2 * p.T @ ( I) @ p
    term1 = 1/2 * np.sum(p**2)
    
   # term2 = 1/2 * D * np.log(m)
    #term2 = 0
    
    term3 = D/2 * np.log(2*np.pi)
    
   # total = term1 + term2 + term3

    total = term1 + term3
    
    
    return total


def H(U, q, p):
    u = U(q)
    k = K(p)
    
    return u + k



m = 1

def do_hmc(position_init, du, total_samples=10000, step_size=1e-3, leapfrog_steps=50, burn_in=.1, 
           thinning_factor=2, verbose=0):
    """
    Performs hamiltonian monte carlo using the Euclidean-Gaussian Kinetic Energy when m=1
    
    In this case,
    
    dk/dp = p
    du/dq = - (d/dq pi(q)/pi(q))
    """
    t1 = time.time()
    try:
        position_init = float(position_init)
    except:
        pass
    
    if verbose >= 1:
        print("position_init")
        print(position_init)
        print("u(position_init)")
        print(u(position_init))
        print("du(position_init)")
        print(du(position_init))

    
    try:
        dim = len(position_init)
    except TypeError:
        dim = 1
        

    mu = jnp.zeros((dim))
    sigma = jnp.eye(dim)
    

    
    q_current = position_init
   
    qs = []
    num_accepts = 0
    for i in range(total_samples):
        if i % (total_samples/100) == 0:
            #clear_output(wait=True)
            if verbose > 0: 
                print("{}% finished".format(round(100*i/total_samples)), flush=True)
        
        # sample random momentum
        p_current = np.random.multivariate_normal(mu, sigma)
        

        
        if verbose >= 1:
            print("p_current")
            print(p_current)
        
        # do leap-frog integration
        p_old = p_current
        q_old = q_current
        
        if verbose >= 1:
            print("p_old")
            print(p_old)
            print("q_old")
            print(q_old)
        
        
        for j in range(leapfrog_steps):      
            # momentum half-step update
            p_new_half = p_old - step_size/2 * du(q_old)


            
            # position full-step update
            q_new = q_old + step_size * p_new_half

            
            # momentum half-step update
            p_new = p_new_half - step_size/2 * du(q_new)
            
            if verbose >= 2:
                print("du(q_old)")
                print(du(q_old))

                print("p_new_half")
                print(p_new_half)
                
                print("q_new")
                print(q_new)
                
                print("du(q_new)")
                print(du(q_new))
                
                print("p_neq")
                print(p_new)

            p_old = p_new
            q_old = q_new
            
        if verbose >= 1:
            print("end of leap frog")
            print("q_new")
            
            print(q_new)
            print("p_new")
        
               
        # reverse momentum
        p_new = -p_new
        
        # correct for simulation error
        
        if verbose >= 1:
            print("H(current)")
            print(H(u, q_current, p_current))
            print("H(new)")
            print(H(u, q_new, p_new))


        # we do not use the correction for simulation error here
 
       #  alphalog = min(0, H(u, q_new, p_new) - H(u, q_current, p_current))
       #  ulog = jnp.log(np.random.rand())
        
       # # if H(u, q_current, p_current) > H(u, q_new, p_new):
       # #     raise Exception


       #  print("acceptance prob: {}".format(jnp.exp(H(u, q_new, p_new) - H(u, q_current, p_current))))
        
        #if (ulog <= alphalog):
        if True:
            num_accepts += 1
            q_current = q_new
            p_current = p_new

        qs.append(q_current)
        
    #clear_output(wait=True)
    if verbose > 0:
        print("100% finished", flush=True)
        t2 = time.time()
        print("Finished in {:2f} seconds".format(t2 - t1))
        print("Accepted {}% of samples".format(100*num_accepts/total_samples))
        
        
    start = round(burn_in * total_samples)

    ret = np.array(qs)
    
    thinned_samples = ret[start::thinning_factor, :]

    
    return thinned_samples
            
def normpi(q):
    try:
        q = q[0]
    except:
        pass

    return 1/jnp.sqrt(2*jnp.pi) * jnp.exp(-q**2 / 2)

def u(q):
    return -jnp.log(normpi(q))


def do_experiments(num_sampless, weights, mus, sigmas,
                   h='heuristic1'):
    """
    more samples -> better estimate

    For a fixed # of samples, tuning the # of eigenvectors based on
    the procedure they outline using a threshold value of 0.99

    how does the log likelihood scale as the number of samples increases

    """

    log_lik = log_lik_gmm(weights, mus, sigmas)

    # get log lik from HMC using ground truth
    q_init = np.zeros(len(mus[0])).reshape(-1, len(mus[0]))

    #u: negative log density
    #also need du

    # du = grad(u)

    # true_samples = sample_gmm()

    # log_lik(hmc_samples)

    # print("Log Likelihood of ground truth -GLD")
    # print(log_lik(hmc_samples.reshape(-1, 1)))

    ground_truth_samples = sample_gmm(num_samples=100,
                             weights=weights,
                             mus=mus,
                             sigmas=sigmas)

    print(pd.DataFrame(ground_truth_samples).cov())

    # get sufficient coverage of 
    #
    x = np.linspace(-5, 5, 100)

    for num_samples in num_sampless:
        print("Running HMC using {} samples for GLD".format(num_samples))
        # get samples from ground truth sampler
        samples = sample_gmm(num_samples=num_samples,
                             weights=weights,
                             mus=mus,
                             sigmas=sigmas)

        # get GLD estimate
        ssge = SSGE(samples,
#                g=x.reshape(-1, 1),
                J=num_samples,
                width_rule='heuristic1',
                    r=0.99)

        # determine optimal num eigvecs
        J = ssge.J
        print("Using {} eigenvectors".format(J))

        # get - GLD
        def du(x):
            try:
                x.shape
            except:
                x = np.array([[x]])

            return -1 * ssge.gradient_estimate_vectorized(J, x)

        # get the HMC samples
        hmc_samples = do_hmc(q_init,
                     du,
                     total_samples=250,
                     step_size=0.01,
                     leapfrog_steps=1000,
                     burn_in=.2,
                     thinning_factor=2,
                         verbose=False)

        try:
            hmc_samples = hmc_samples[:, 0, :]
        except:
            pass

        print(pd.DataFrame(hmc_samples).corr())
        print("mean {} | num_samps {} | num hmc samples {} | log_lik {} | log_lik hmc {}".format(
            np.mean(hmc_samples, axis=0),
            num_samples,
            len(hmc_samples),
            log_lik(ground_truth_samples),
            log_lik(hmc_samples)
        ))
    
  





if __name__ == '__main__':
    num_samples = [10,
                   25,
                   50,
                   75,
                   100,
                   250,
                   500,
                   750,
                   1000,
                   2500,
                   5000,
                   # 7500,
                   # 10000
    ]

    dim = 2
    weights = [1]
    mus = [jnp.zeros(dim)]
    sigmas = [jnp.eye(dim)]

    do_experiments(num_samples, weights, mus, sigmas)
