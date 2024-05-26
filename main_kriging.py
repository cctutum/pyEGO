#
#
#

import numpy as np
from sampling import lhs
import matplotlib.pyplot as plt
from likelihood import likelihood, likelihood_wrapper
from testproblems import kriging
from scipy.optimize import minimize

k = 2
n = 4*k

#%% Create initial samples using LHS

X = lhs(k, n)

y = []
for x in X:
    y.append( kriging(x) )
    print(x, '---', y[-1])

# Set upper and lower bounds for search of log theta
theta_max = 2.0
theta_min = -3.0

# Train Kriging (search for minimum likelihood)
bnds = tuple((theta_min, theta_max) for i in range(k))

#%% Implement a multi-start strategy

n_runs = 50 # Choose n_runs=1 for a single-start strategy
theta_list, NegLnLike_list = [], []
for i in range(50):
    theta0 = np.random.uniform(theta_min, theta_max, k) # random initial guess
    print(theta0)
    theta = minimize(likelihood_wrapper, theta0, args=(X, y), 
                      bounds=bnds, method='SLSQP', tol=1e-6,
                      options={'disp': True})
    NegLnLike, _, _ = likelihood(theta.x, X, y)
    theta_list.append(theta.x)
    NegLnLike_list.append(NegLnLike)

# Choose the best theta
zipped = list(zip(theta_list, NegLnLike_list))
min_index, _ = min(enumerate(zipped), key=lambda x: x[1][1])
theta_opt = zipped[min_index][0]

print(f"optimal theta-vector for Kriging model={theta_opt}")
print(f"NegLnLike= {NegLnLike}")

#%%






