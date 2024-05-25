from scipy.stats import qmc # Quasi-Monte Carlo submodule
import numpy as np
import time


def lhs(d, n, centered=True, method='lloyd', seed=None):
    # Latin Hypercube Sampling (LHS)
    #
    # d: int 
    # The number of dimensions. 
    # 
    # n: int 
    # The number of samples.
    #
    # centered: bool 
    # When centered=True, samples are placed in center within cells,
    # otherwise samples are placed randomly within cells.
    #
    # optimization: {None, “random-cd”, “lloyd”}
    # I am not sure if this matters!
    #
    # seed: int
    # Seed number for pseudo random number generator
    
    # if seed is not given, choose it as function of time which will be different
    # every time it is used
    if not seed:
        seed = int(time.time() * 1000000)
    rng = np.random.default_rng(seed)
    
    sampler = qmc.LatinHypercube(d, 
                                 scramble=not centered,
                                 optimization=method,
                                 seed=rng)
    
    sample = sampler.random(n)
    return sample
