import torch
import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt
from ekf.plotting import plotter
from ekf.utils import add_noise_with_snr
from aesindy import torch_config
from scipy.integrate import odeint

seed = 29

# Setup torch device and dtype
torch_config.setup_device_and_type()
np.random.seed(seed)

# Define Selkov system to be found with SINDy

def selkov(x, t, params):
    rho = params['rho']
    sigma1 = params['sigma1']
    sigma2 = params['sigma2']
    chi1 = params['chi1']
    k2 = params['k2']
    chi2 = params['chi2']
    
    x1 = x[0]
    x2 = x[1]
    
    dx1 = rho - sigma1 * x1 - chi1 * x1 * x2 ** 2
    dx2 = sigma2 * x1 - k2 * x2 + chi2 * x1 * x2 ** 2

    return [dx1, dx2]

import numpy as np

# x_train_np: shape (n_train, T, 2)
# dt: scalar time step
dt = 0.01  # or whatever you used

params = {
    'rho': 0.92,
    'sigma1': 0.10,  
    'sigma2': 0.10,
    'chi1': 1.0,  
    'k2': 1.0,    
    'chi2': 1.0,  
}

# Generate observations and cast them to appropriate torch tensors of shape (#datapoints, #time_instances, #dim)
dt = 0.01
n_train = 20
std = 2
mean_ic = np.array([0.05, 0.05])
x_0 = np.random.randn(n_train, 2) * std + mean_ic
time_instances = np.arange(0, 100, dt)
x_train = []

for i in range(n_train):
    sol = odeint(selkov, x_0[i, :], time_instances, args=(params,))
    x_train.append(sol)

model = ps.SINDy(feature_names=['u1', 'u2'],
                 feature_library=ps.PolynomialLibrary(degree=3),
                 optimizer=ps.STLSQ(threshold=5e-2))
model.fit(x_train, t=0.01, multiple_trajectories=True)
model.print()

x_train_np = np.stack(x_train, axis=0)     # shape: (n_train, T, 2)

# Central difference for interior points
dxdt_mid = (x_train_np[:, 2:, :] - x_train_np[:, :-2, :]) / (2 * dt)

# Forward difference for first point
dxdt_0 = (x_train_np[:, 1:2, :] - x_train_np[:, 0:1, :]) / dt

# Backward difference for last point
dxdt_T = (x_train_np[:, -1:, :] - x_train_np[:, -2:-1, :]) / dt

# Concatenate to match shape (n_train, T, 2)
dxdt_np = np.concatenate([dxdt_0, dxdt_mid, dxdt_T], axis=1)

print(dxdt_np.shape)  # should be (n_train, T, 2)


model = ps.SINDy(
    feature_names=['u1', 'u2'],
    feature_library=ps.PolynomialLibrary(degree=3),
    optimizer=ps.STLSQ(threshold=5e-2)
)

x_list = [x_train_np[i] for i in range(n_train)]
dxdt_list = [dxdt_np[i] for i in range(n_train)]

model.fit(x=x_list, t=dt, x_dot=dxdt_list, multiple_trajectories=True)
model.print()
