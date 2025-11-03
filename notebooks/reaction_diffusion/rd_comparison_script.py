import gc
import time
import matplotlib
import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
from ekf.plotting import plotter
from ekf.utils import add_noise_with_snr
from ekf.filters.config import DynamicsConfig
from ekf.filters.ekf import EKF
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from IPython.display import Video
from ekf.filters import error_computation


seed = 29
np.random.seed(seed)

def compare_animation(field_sindy: np.ndarray, field_pod: np.ndarray, t: np.ndarray):
    """
    Animate two 3D fields (nx, ny, nt) side by side to compare visually.
    Shows animation live when run from a script.
    """
    assert field_sindy.shape == field_pod.shape, "Both fields must have the same shape"
    nx, ny, nt = field_sindy.shape

    vmin = min(field_sindy.min(), field_pod.min())
    vmax = max(field_sindy.max(), field_pod.max())

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    ax1, ax2 = axes
    im1 = ax1.imshow(field_sindy[:, :, 0], origin='lower', vmin=vmin, vmax=vmax)
    im2 = ax2.imshow(field_pod[:, :, 0], origin='lower', vmin=vmin, vmax=vmax)
    ax1.set_title('SINDy Solution')
    ax2.set_title('True POD Projection')

    for ax in axes:
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    fig.colorbar(im1, ax=axes, orientation='vertical', fraction=0.046, pad=0.04)
    time_text = fig.text(0.5, 0.02, f"t = {t[0]:.2f}", ha='center', va='bottom', fontsize=12)

    def update(frame):
        im1.set_data(field_sindy[:, :, frame])
        im2.set_data(field_pod[:, :, frame])
        time_text.set_text(f"t = {t[frame]:.2f}")
        return [im1, im2, time_text]

    anim = FuncAnimation(fig, update, frames=nt, interval=50, blit=False)
    plt.show()  # <-- show live animation


# Extract data, and stack it

start_t = time.time()

filename = "simulation_data/rd_spiral/rd_spiral_mu_0.800_to_1.600_d1_0.01_d2_0.01_m_1_beta_1.1.npz"
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

# # Scaling and zero-centering, is it necessary in this case?
# mean_uv = np.mean(full_uv, axis=1, keepdims=True)
# full_uv = full_uv - mean_uv

top_k = 10
start_t = time.time()
U, S, VT = svds(full_uv, k=top_k, which='LM')
print(f"SVD computed in {time.time() - start_t:.4f} seconds.")

# Since the matrix is big, just get rid of them after computing the SVD. We also keep full_uv for later use. Also we reverse the columns of U
U = U[:, ::-1]  
S = S[::-1]     # Reverse singular values
full_uv = full_uv.reshape((-1, t, mu_instances))
print(f'full_uv shape: {full_uv.shape}')

del u, v, u_matrix, v_matrix, data
gc.collect()

top_k = 4
mu_index = 0
pod_projection = full_uv[:, :, mu_index].T @ U[:, :top_k]

# Now we fit the first parameter instance (corresponding to the zeroth index and mu=0.8)
poly_lib = ps.PolynomialLibrary(degree=3)
fourier_lib = ps.FourierLibrary(n_frequencies=1)
combined_lib = poly_lib + fourier_lib

model = ps.SINDy(feature_library=combined_lib,
                 optimizer=ps.STLSQ(threshold=1e-2))

model.fit(pod_projection, t=0.05)
model.print()

# Solve SINDy model with RK4 
initial_condition = pod_projection[0, :]
t_span = np.arange(0, pod_projection.shape[0] * 0.05, 0.05)
rhs = lambda x, t: model.predict(x.reshape(1, -1)).flatten()
sindy_solution = odeint(rhs, initial_condition, t_span)


sindy_sol = U[:, :top_k] @ sindy_solution.T
pod_sol = U[:, :top_k] @ pod_projection.T


u_sindy = sindy_sol.reshape((2, 50, 50, 801))[0, :, :, :]
u_true = pod_sol.reshape((2, 50, 50, 801))[0, :, :, :]
anim = compare_animation(u_sindy, u_true, t=np.linspace(0, 40, 801))