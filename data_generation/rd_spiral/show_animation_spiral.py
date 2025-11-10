# Show spiral animation from generated data

import numpy as np
import time
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

def show_animation(mu_index: int):
    # Load data 
    filename = "simulation_data/rd_spiral/rd_spiral_mu_0.900_to_1.100_d1_0.01_d2_0.01_m_1_beta_1.1.npz"

    start = time.time()
    data = np.load(filename)
    end = time.time()
    print(f"Data loading time: {end - start:.4f} seconds")

    u = data["u"][:, :, :-1, :]
    v = data["v"][:, :, :-1, :]
    
    print(f'u shape: {u.shape}, v shape: {v.shape}')

    # We show only u
    t = np.arange(0, 40, 0.05)

    fig, ax = plt.subplots(figsize=(5,5))
    field = u[:, :, :, mu_index]
    cax = ax.imshow(field[:, :, 0], aspect='auto', origin='lower')
    vmin, vmax = field.min(), field.max()
    cax.set_clim(vmin, vmax)

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Use ax.text instead of ax.set_title
    time_text = ax.text(0.5, 1.02, f't = {t[0]:.2f}', transform=ax.transAxes,
                        ha='center', va='bottom', fontsize=12)

    def update(frame):
        cax.set_data(field[:, :, frame])
        time_text.set_text(f't = {t[frame]:.2f}')  # update the text
        return [cax, time_text]

    # Use blit=False for safety
    anim = FuncAnimation(fig, update, frames=len(t), interval=50, blit=False)

    plt.show()
    
if __name__ == "__main__":
    mu_index = 5 # Change this according to the index of mu you want to visualize
    show_animation(mu_index)