import numpy as np
import time
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

def show_animation(mu_index: int):
    # Load data
    filename = "simulation_data/rd_spiral/mu_1.000_to_1.100_d_0.01_m_2_beta_1.1_T_10.0_dt_0.05.npz"
    start = time.time()
    data = np.load(filename)
    end = time.time()
    print(f"Data loading time: {end - start:.4f} seconds")

    u = data["u"][:, :, :-1, :]
    print(f'u shape: {u.shape}')

    # Use only u
    field = u[:, :, :, mu_index]
    frames = field.shape[2]
    t = np.linspace(0, 10, frames)

    fig, ax = plt.subplots(figsize=(5,5))
    cax = ax.imshow(field[:, :, 0], aspect='auto', origin='lower')
    cax.set_clim(field.min(), field.max())

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    time_text = ax.text(0.5, 1.02, f't = {t[0]:.2f}', transform=ax.transAxes,
                        ha='center', va='bottom', fontsize=12)

    def update(frame):
        cax.set_data(field[:, :, frame])
        time_text.set_text(f't = {t[frame]:.2f}')
        return [cax, time_text]

    # Create animation
    anim = FuncAnimation(fig, update, frames=frames, interval=50, blit=False)

    # Save as MP4
    anim.save("spiral.mp4", writer="ffmpeg", fps=20, dpi=200)
    print("Animation saved as spiral.mp4")
    
if __name__ == "__main__":
    mu_index = 1 # Change this according to the mu_i you want to visualize
    show_animation(mu_index)