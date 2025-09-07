import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt
import sympy as sp
from ekf_vindy.plotting import plotter
from ekf_vindy.utils import add_noise_with_snr
from ekf_vindy.filters.state import State
from ekf_vindy.filters.ekf import EKF
from scipy.integrate import odeint

# remember to run as python -m script (if there are slashes use dots as in dir1.dir2.script)
"""
Playing with Selkov model, and debugging
"""

def selkov(y, t, params, t_offset = 0, t_bifurcation = 0, train = True):
    # Selkov model
    u1, u2 = y
    
    if train:
        rho = params['rho']
    if not train:
        if t < t_bifurcation:
            rho = params['rho']
        elif t > t_bifurcation + t_offset:
            rho = params['rho_bifurcation']
        else:
            rho = params['rho'] + (params['rho_bifurcation'] - params['rho']) * (t - t_bifurcation) / t_offset
   
    du1 = rho - params['sigma'] * u1 - params['b122_1'] * u1 * u2**2
    du2 = params['sigma'] * u1 - params['k2'] * u2 + params['b122_2'] * u1 * u2**2

    return [du1, du2]

# set parameters
params = {
    "rho": 0.92,
    "rho_bifurcation": 0.7,  # for example, after bifurcation
    "sigma": 0.1,
    "k2": 1.0,
    "b122_1": 1.0,
    "b122_2": 1.0,
}
np.random.seed(3)

u1_0 = np.random.normal(0.2, 1.7, 16)
u2_0 = np.random.normal(0.2, 1.7, 16)

time_instances = np.arange(0, 100, 0.1)
x_train = []

for i in range(16):
    y0 = [u1_0[i], u2_0[i]]
    sol = odeint(selkov, y0, time_instances, args=(params,))
    x_train.append(sol)

model = ps.SINDy(feature_names=['u1', 'u2'],
                 feature_library=ps.PolynomialLibrary(degree=3),
                 optimizer=ps.STLSQ(threshold=5e-2))

model.fit(x_train, t=0.1, multiple_trajectories=True)
model.print()

# generate trajectories with this model
time_instances = np.arange(0, 300, 0.1)
u1_0 = np.random.normal(0.2, 1, 1)
u2_0 = np.random.normal(0.2, 1, 1)

y0 = np.array([u1_0[0], u2_0[0]])
X = odeint(selkov, y0, time_instances, args=(params, 50, 50, False))
noisy_X = add_noise_with_snr(X, 12)
fig, ax = plotter.plot_trajectory(noisy_X, time_instances, x_tick_skip=30, title='Selkov oscillator')
plt.show()

# Now use filter !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1

# setup covariance matrices... sometimes values in code disagree with value in paper...

# p0 = np.diag([1e-8, 1e-8, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-4])
p0 = np.diag([1e-8, 1e-8, 5e-4, 1e-3, 5e-4, 1e-3, 1e-4, 5e-4, 1e-3])
# q = np.diag([1e-6, 1e-6, 1e-14, 1e-14, 1e-14, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12]) 
q = np.diag([8e-7, 8e-7, 5e-6, 1e-9, 1e-10, 1e-10, 1e-11, 1e-9, 1e-12])

r = np.diag([5e-4, 5e-4])

coeffs = model.coefficients()
library_terms = model.get_feature_names()
variables = model.feature_names
x0 = noisy_X[0,:]

print(f'library_terms: {library_terms}')
tracked_terms = [[0, 1, 4, 8],
                 [1, 2, 8]]

filter = EKF(x0, p0, q, r, variables, library_terms, tracked_terms, coeffs, integration_rule="RK4")

dts = np.diff(time_instances)
observations = noisy_X[1:,:]

filter.run_filter(dts, observations)

filter_estimates = filter.states.x_states
fig, x = plotter.plot_trajectory(filter_estimates, time_instances, x_tick_skip=30, title='Selkov model')
plt.show()
# TODO: It's useless to pass the state, just pass the x state 
#       and the entire coefficient matrix (you need it anyway)
#       from which you will slice and build your state.

# fix this little thing and we should be able to test...