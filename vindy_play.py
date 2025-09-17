import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt
import sympy as sp
from ekf_vindy.plotting import plotter
from ekf_vindy.utils import add_noise_with_snr
from ekf_vindy.filters.config import DynamicsConfig
from ekf_vindy.filters.ekf import EKF
from scipy.integrate import odeint

# vector field definition
def selkov(y, t, params, t_offset = 0, t_bifurcation = 0, train = True):
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

# parameters of vector field
params = {
    "rho": 0.92,
    "rho_bifurcation": 0.7,  # for example, after bifurcation
    "sigma": 0.1,
    "k2": 1.0,
    "b122_1": 1.0,
    "b122_2": 1.0,
}

# generate training data
np.random.seed(3)
u1_0 = np.random.normal(0.2, 1.7, 30)
u2_0 = np.random.normal(0.2, 1.7, 30)

time_instances = np.arange(0, 100, 0.1)

x_train = np.zeros((30, len(time_instances), 2))
for i in range(30):
    y0 = [u1_0[i], u2_0[i]]
    sol = odeint(selkov, y0, time_instances, args=(params,))
    x_train[i] = sol


# model fitting, to be changed with VINDy!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# compute finite differences


# print(len(dxdt))
# print(len(dxdt[0]))
dxdt = np.array([np.gradient(x, 0.1, axis=0) for x in x_train])

fig, ax = plotter.plot_trajectory(dxdt[0, :, :], time_instances, title='Derivatives')
plt.show()


# # see how it evolves... 
# y0 = np.array([u1_0[0], u2_0[0]])
# time_instances = np.arange(0, 300, 0.1)
# X = odeint(selkov, y0, time_instances, args=(params, 50, 50, False))
# noisy_X = add_noise_with_snr(X, 25)
# fig, ax = plotter.plot_trajectory(noisy_X, time_instances, x_tick_skip=30, title='Selkov oscillator')
# plt.show()

# # p0 should be given by VINDy, the rest is just like before
# p0 = np.diag([1e-8, 1e-8, 5e-4, 1e-3, 5e-4, 1e-3, 1e-4, 5e-4, 1e-3])
# q = np.diag([8e-7, 8e-7, 5e-6, 1e-9, 1e-10, 1e-10, 1e-11, 1e-9, 1e-12])
# r = np.diag([5e-4, 5e-4])

# coeffs = model.coefficients()
# library_terms = model.get_feature_names()
# variables = model.feature_names
# x0 = noisy_X[0,:]

# tracked_terms = [[0, 1, 4, 8],
#                  [1, 2, 8]]

# config = DynamicsConfig(variables, library_terms, tracked_terms, coeffs, q, r)
# filter = EKF(x0, p0, config=config, integration_rule="RK4")

# dts = np.diff(time_instances)
# observations = noisy_X[1:,:]

# filter.run_filter(dts, observations)

# filter_estimates = filter.states.xcal_states[:, 0:2]
# sdevs = filter.states.sdev_states[:, 0:2]

# fig, x = plotter.plot_trajectory(filter_estimates, time_instances, sdevs, x_tick_skip=30, title='Selkov oscillator')
# plt.show()
