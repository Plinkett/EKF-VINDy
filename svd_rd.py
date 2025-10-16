import numpy as np
import time
import numpy as np
from scipy.sparse.linalg import svds
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
# import pysindy as ps
from scipy.integrate import odeint
# from ekf_vindy.plotting import plotter
# Load data
data = np.load("spiral_data.npz")
u = data["u"]
v = data["v"]


print(f'u shape: {u.shape}, v shape: {v.shape}')
time_steps = u.shape[2]
flat_dim = u.shape[0] * u.shape[1]
u_flat = u.reshape((flat_dim, time_steps))
v_flat = v.reshape((flat_dim, time_steps))

full_uv = np.vstack((u_flat, v_flat))
print(f'full_uv shape: {full_uv.shape}')
# Full SVD
start = time.time()
U, S, VT = np.linalg.svd(full_uv, full_matrices=False)

pod_modes = U[:, :2]
#init = pod_modes.T @ full_uv[:, 0]
init = np.array([9, 5])

# OK, so there's probably sth wrong with how I compute the SVD with multiple trajectories
# I would expect the dynamics to be relatively simple in POD space, but instead I get too many polynomial terms
# There's something wrong...

time_instances = np.arange(0, 40, 0.05)

def ode(x, t):
    dx0 = 1.376 * x[1]
    dx1 = -1.377 * x[0]
    return [dx0, dx1]

sol = odeint(ode, init, time_instances)
# fig, ax = plotter.plot_trajectory(sol, time_instances)
# plt.show()

# plot solutions in POD space
reconstructed = pod_modes @ sol.T

print(f'reconstructed shape: {reconstructed.shape}')
# animate 
u_reconstructed = reconstructed[:2500, :]
u_reconstructed = u_reconstructed.reshape((50, 50, -1))

print(f'u_reconstructed shape: {u_reconstructed.shape}')

field = u_reconstructed
fig, ax = plt.subplots(figsize=(5,5))
cax = ax.imshow(field[:, :, 0], origin='lower',
                extent=[-10, 10, -10, 10])

vmin, vmax = field.min(), field.max()
cax.set_clim(vmin, vmax)

ax.set_xlabel('x')
ax.set_ylabel('y')

time_instances = np.linspace(0, 40, 400)

# Use ax.text instead of ax.set_title
time_text = ax.text(0.5, 1.02, f't = {time_instances[0]:.2f}', transform=ax.transAxes,
                    ha='center', va='bottom', fontsize=12)

def update(frame):
    cax.set_data(field[:, :, frame])
    time_text.set_text(f't = {time_instances[frame]:.2f}')  # update the text
    return [cax, time_text]

# Use blit=False for safety
anim = FuncAnimation(fig, update, frames=len(time_instances), interval=50, blit=False)
plt.show()

# modes = 2
# best_modes = U[:, :modes]

# print(f'Best {modes} modes shape: {best_modes.shape}')

# print(f'full_uv.shape: {full_uv.shape})')
# proj = best_modes.T @ full_uv
# t = data['t'].squeeze()  # shape: (401,)
# print(f'proj shape: {proj.shape}')

# proj = proj.T

# model = ps.SINDy(feature_library=ps.PolynomialLibrary(degree=3),
#                  optimizer=ps.STLSQ(threshold=5e-2))
# model.fit(proj, t=0.05, multiple_trajectories=False)
# model.print()




# plt.figure()
# for i in range(proj.shape[0]):  # iterate through 10 modes
#     plt.plot(t, proj[i, :], label=f"Mode {i+1}")

# plt.show()


# # from the singular values
# energy = S**2
# cumulative_energy = np.cumsum(energy) / np.sum(energy)

# modes = np.arange(1, min(100, len(S)) + 1)

# plt.figure()
# plt.plot(modes, cumulative_energy[:len(modes)])
# plt.xlabel('Mode Number')
# plt.ylabel('Cumulative Energy')
# plt.title('Cumulative Energy of First 100 Spatial Modes')
# plt.grid(True)
# plt.show()

# # Truncated SVD (e.g., top 50 modes)
# start = time.time()
# U_k, S_k, VT_k = svds(full_uv, k=30)
# print(f"Top-30 svds: {time.time() - start:.4f} sec")