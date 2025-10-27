import numpy as np
import time
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

# Load data
filename = "simulation_data/rd_spiral/rd_spiral_mu_0.800_to_1.600_d1_0.01_d2_0.01_m_1_beta_1.1.npz"
start = time.time()
data = np.load(filename)
end = time.time()
print(f"Data loading time: {end - start:.4f} seconds")

u = data["u"][:, :, :-1, :]  # shape: (Nx, Ny, 800, N_mu)
print(f'u shape: {u.shape}')

# μ values
mu_list = [0.8, 1.1, 1.3, 1.6]
mu_values = np.linspace(0.9, 1.6, 31)
mu_dict = {round(val, 3): idx for idx, val in enumerate(mu_values)}
mu_list = [round(mu, 3) for mu in mu_list if round(mu, 3) in mu_dict]
print(f"Showing μ values: {mu_list}")

# Time vector
t = np.arange(0, 40, 0.05)

# Set up multiple subplots
n = len(mu_list)
fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
if n == 1:
    axes = [axes]  # ensure iterable

images = []
time_texts = []

for i, mu in enumerate(mu_list):
    mu_idx = mu_dict[mu]
    field = u[:, :, :, mu_idx]
    ax = axes[i]
    im = ax.imshow(field[:, :, 0], aspect='auto', origin='lower')
    im.set_clim(field.min(), field.max())
    ax.set_title(f"μ = {mu}")
    ax.set_xlabel("x")
    if i == 0:
        ax.set_ylabel("y")
    txt = ax.text(0.5, 1.02, f't = {t[0]:.2f}', transform=ax.transAxes,
                  ha='center', va='bottom', fontsize=10)
    images.append((im, field))
    time_texts.append(txt)

# Update function for all plots
def update(frame):
    for (im, field), txt in zip(images, time_texts):
        im.set_data(field[:, :, frame])
        txt.set_text(f't = {t[frame]:.2f}')
    return [im for im, _ in images] + time_texts

# Create animation
anim = FuncAnimation(fig, update, frames=len(t), interval=50, blit=False)
plt.tight_layout()
plt.show()
