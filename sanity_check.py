import tensorflow as tf
import os
import scipy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from vindy import SindyNetwork
from vindy.layers import SindyLayer, VindyLayer
from vindy.distributions import Gaussian, Laplace
from vindy.libraries import PolynomialLibrary
from vindy.callbacks import (
    SaveCoefficientsCallback,
)
from vindy.utils import add_lognormal_noise
from ekf_vindy.plotting.plotter import plot_trajectory
from scipy.integrate import odeint

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


sindy_type = "vindy"  # "sindy" or "vindy", if you either want a deterministic or probabilistic model for the dynamics, respectively.
model_name = "selkov"
seed = 29 # random seed
measurement_noise_factor = 0.01 # measurement noise factor
n_train = 30 # number of training trajectories
n_test = 4 # number of test trajectories


def generate_directories(model_name, sindy_type, scenario_info, outdir):
    # Base output directory
    outdir = os.path.join(outdir, model_name, sindy_type)
    
    # Subdirectories
    figdir = os.path.join(outdir, "figures", scenario_info)
    log_dir = os.path.join(outdir, model_name, "log", scenario_info)
    weights_dir = os.path.join(outdir, "weights", scenario_info)
    
    # Create directories if they don't exist
    for dir_path in [outdir, figdir, log_dir, weights_dir]:
        os.makedirs(dir_path, exist_ok=True)

    return outdir, figdir, log_dir, weights_dir

scenario_info = f"{sindy_type}_nbd_me_seed_{seed}_noise_{measurement_noise_factor}"
_, _, _, weights_dir = generate_directories(model_name, sindy_type, scenario_info, "results")

params = {
    "rho": 0.92,
    "rho_bifurcation": 0.7,  # for example, after bifurcation
    "sigma": 0.1,
    "k2": 1.0,
    "b122_1": 1.0,
    "b122_2": 1.0,
}

# generate data from Selkov model and compute derivatives (from data)
initial_conditions = np.random.normal(0.2, 1, (n_train + n_test, 2))
dt = 0.1
time_instances = np.arange(0, 100, dt)


x_train = np.array([
        odeint(selkov, y0, time_instances, args=(params,))
        for y0 in initial_conditions[:n_train]
    ])

x_train = np.array([add_lognormal_noise(x, measurement_noise_factor)[0] for x in x_train])

dxdt_train = np.array([
                    np.gradient(x, dt, axis=0)
                    for x in x_train
                ])

# reshape data to fit the model (why does it get rid of trajectories and puts them in one stack?)

x_train_feed = np.concatenate(x_train, axis=0)
dxdt_train_feed = np.concatenate(dxdt_train, axis=0)

# model parameters
libraries = [PolynomialLibrary(2, include_bias=True),]


# Parameters for SINDy/VINDy layer (not really explained in the documentation...)
layer_params = dict(
    state_dim=x_train.shape[1],
    param_dim=0,
    feature_libraries=libraries,
    second_order=False,
    mask=None,
    kernel_regularizer=tf.keras.regularizers.L1L2(l1=0, l2=0),
)

print("before sindy layer")

if sindy_type == "vindy":
    sindy_layer = VindyLayer(
        beta=1e-3,
        priors=Laplace(0.0, 1.0),
        **layer_params,
    )
else: 
    print('Sorry, only VINDy is implemented for the Selkov model.')
    
model = SindyNetwork(
    sindy_layer=sindy_layer,
    x=x_train,
    l_dz=1e0,
    dt=dt,
    second_order=False,
)
print("HEREAJDHJASDHASKJD!!")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="huber")