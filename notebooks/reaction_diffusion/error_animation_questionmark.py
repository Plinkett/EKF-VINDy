import gc
import time
import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
from ekf.utils import add_noise_with_snr
from scipy.integrate import odeint
from ekf.filters.ekf import EKF
from ekf.filters.config import DynamicsConfig
from ekf.plotting import plotter
from ekf.filters import error_computation
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter

seed = 29
np.random.seed(seed)


# Collect data and compute modes

# Extract data, and stack it

start_t = time.time()

filename = "simulation_data/rd_spiral/rd_spiral_mu_0.900_to_1.100_d1_0.01_d2_0.01_m_1_beta_1.1.npz"
data = np.load(filename)

print(f"Data loaded in {time.time() - start_t:.4f} seconds.")

# Reshape data, the third dimension are the number of parameter instances

u = data['u']
v = data['v']
t = u.shape[2]

flat_dim = u.shape[0] * u.shape[1]

mu_instances = u.shape[3]

u_matrix = u.reshape((flat_dim, t * mu_instances))
v_matrix = v.reshape((flat_dim, t * mu_instances))
full_uv = np.vstack((u_matrix, v_matrix))

# # Scaling and zero-centering
# mean_uv = np.mean(full_uv, axis=1, keepdims=True)
# full_uv = full_uv - mean_uv

top_k = 5
start_t = time.time()
U, S, VT = svds(full_uv, k=top_k, which='LM')
U = U[:, ::-1]
S = S[::-1]
print(f"SVD computed in {time.time() - start_t:.4f} seconds.")

# Since the matrix is big, just get rid of them after computing the SVD. We also keep full_uv for later use
full_uv = full_uv.reshape((-1, t, mu_instances))
print(f'full_uv shape: {full_uv.shape}')

del u, v, u_matrix, v_matrix, data
gc.collect()

# Project onto top-K modes

top_k = 2
t_span = np.arange(0, 40.005, 0.05)

mu_index = 0
pod_projection = full_uv[:, :, mu_index].T @ U[:, :top_k]  # Shape (time, modes)


# Fit SINDy model on the first parameter instance trajectory

mu_index = 0
pod_projection = full_uv[:, :, mu_index].T @ U[:, :top_k]

# Now we fit the first parameter instance (corresponding to the zeroth index and mu=0.8)
poly_lib = ps.PolynomialLibrary(degree=3)
fourier_lib = ps.FourierLibrary(n_frequencies=1)
combined_lib = poly_lib + fourier_lib

model = ps.SINDy(feature_library=combined_lib,
                 optimizer=ps.STLSQ(threshold=5e-2),
                 feature_names=[f"z{i}" for i in range(top_k)])

model.fit(pod_projection, t=0.05)
model.print()


############
# Run EKF  #
############

# ekf_callback = PODProjection(pod_modes=U[:, :top_k])
filename = "simulation_data/rd_spiral/transient/rd_spiral_transient_mu_0.7_to_1.5_d1_0.01_d2_0.01_m_1.npz"
data = np.load(filename)

# Boring reshaping
u_test, v_test = data['u'], data['v']
flat_dim = u_test.shape[0] * u_test.shape[1]
t_test_length = u_test.shape[2]
full_uv_test = np.vstack((u_test.reshape((flat_dim, t_test_length)),
                           v_test.reshape((flat_dim, t_test_length))))
print(f'full_uv_test shape: {full_uv_test.shape}')

# Data to be fed to the filter
z_test = (full_uv_test.T @ U[:, :top_k])  # Shape (time, modes)
z0_test = z_test[0, :]
z_test = z_test[1:, :]  # Remove initial condition
time_instances = np.arange(1, t_test_length) * 0.05
dts = np.diff(time_instances)

# EKF configuration (2 POD modes + 2 parameters)
p0 = np.diag([1e-8, 1e-8, 1e-4, 1e-4])  # Initial covariance
q = np.diag([1e-8, 1e-8, 1e-8, 1e-8])  # Process noise covariance
r = np.diag([4e-4, 4e-4])  # Measurement noise covariance

library_terms = model.get_feature_names()
coeffs = model.coefficients()
variables = model.feature_names
print(f'library_terms: {library_terms}')
tracked_terms = [[2],
                 [1]]

# Now we run the 
config = DynamicsConfig(variables, library_terms, tracked_terms, coeffs, q, r)
filter = EKF(z0_test, p0, config=config, integration_rule='RK4')
print(f'z_test shape: {z_test.shape}, dts shape: {dts.shape}')
filter.run_filter(dts, z_test)

# Plot results

filter_estimates = filter.states.xcal_states[:, :2]
sdevs = filter.states.sdev_states[:, :2]

# fig, x = plotter.plot_trajectory(filter_estimates, time_instances, sdevs, x_tick_skip=4, title='Top 2 POD modes', xlabel='Time', ylabel='States',
#                                  state_names=["$z_0(t)$", "$z_1(t)$"])
# plt.show()

coupling_terms = filter.states.xcal_states[:, 2:]
coupling_sdevs = filter.states.sdev_states[:, 2:]

coupling_terms = np.abs(coupling_terms)

# fig, x = plotter.plot_trajectory(coupling_terms, time_instances, coupling_sdevs, x_tick_skip=4, title="Damping coefficients $\delta_1(t)$ and $\delta_2(t)$ (no constraints)", xlabel='Time', ylabel='Damping',
#                                  state_names=["$|\\mu_0(t)|$", "$|\\mu_1(t)|$"], palette="viridis")
# plt.show()

filter_sol = U[:, :top_k] @ filter.states.xcal_states[:, :top_k].T
filter_u = filter_sol[:2500, :].reshape(50, 50, -1)

full_uv_test = full_uv_test[:, 1:]
u_test = full_uv_test[:2500, :].reshape(50, 50, -1)

# Compute absolute error
error_u = np.abs(filter_u - u_test)

# --- Setup figure ---
fig, ax = plt.subplots(figsize=(6, 5))
axes = [ax]

titles = [r'Absolute error $u(x,t)$']
fields = [error_u]
cmaps = ['coolwarm']

vmin_err, vmax_err = error_u.min(), error_u.max()

caxes = []
for i, ax in enumerate(axes):
    field = fields[i]
    cmap = plt.get_cmap(cmaps[i])
    cax = ax.imshow(field[:, :, 0], cmap=cmap, origin='lower',
                    vmin=vmin_err, vmax=vmax_err)
    fig.colorbar(cax, ax=ax, shrink=0.8)
    ax.set_title(titles[i], fontsize=12)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    caxes.append(cax)

time_text = fig.text(0.5, 0.02, f't = {time_instances[0]:.2f}', ha='center', fontsize=12)

def update(frame):
    for i, cax in enumerate(caxes):
        cax.set_data(fields[i][:, :, frame])
    time_text.set_text(f't = {time_instances[frame]:.2f}')
    return caxes + [time_text]

anim = FuncAnimation(fig, update, frames=len(time_instances), interval=50, blit=False)
writer = FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)
print("Saving absolute error animation...")
anim.save("abs_error.mp4", writer=writer)
print("✅ Saved as abs_error.mp4")