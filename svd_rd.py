import numpy as np
import time
import numpy as np
from scipy.sparse.linalg import svds
from matplotlib import pyplot as plt
import pysindy as ps

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

modes = 5
best_modes = U[:, :modes]

print(f'Best {modes} modes shape: {best_modes.shape}')

proj = best_modes.T @ full_uv
t = data['t'].squeeze()  # shape: (401,)

proj = proj.T

model = ps.SINDy(feature_library=ps.PolynomialLibrary(degree=3),
                 optimizer=ps.STLSQ(threshold=5e-2))
model.fit(proj, t=0.05, multiple_trajectories=False)
model.print()


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