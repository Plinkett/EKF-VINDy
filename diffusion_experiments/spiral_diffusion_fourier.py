import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# ----------------------
# PARAMETERS
# ----------------------
Nx, Ny = 100, 100      # grid size
dx = dy = 0.05            # spatial step
dt = 0.05              # time step
total_time = 40.0      # total simulation time
d1, d2 = 0.01, 0.01    # diffusion coefficients
mu = 0.5               # phase rotation
substeps = 5           # number of small RK4+diffusion steps per frame

# compute total frames
n_steps = int(total_time / dt)
n_frames = n_steps // substeps

# ----------------------
# INITIAL CONDITIONS
# ----------------------
u = np.zeros((Nx, Ny))
v = np.zeros((Nx, Ny))
cx, cy = Nx//2, Ny//2   
R = 5
for i in range(Nx):
    for j in range(Ny):
        if (i - cx)**2 + (j - cy)**2 < R**2:
            u[i,j] = 0.1
            v[i,j] = 0.1

# ----------------------
# PRECOMPUTE WAVENUMBERS FOR FFT
# ----------------------
kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
KX, KY = np.meshgrid(kx, ky, indexing='ij')
K2 = KX**2 + KY**2

# ----------------------
# REACTION FUNCTION
# ----------------------
def reaction(u, v):
    r2 = u**2 + v**2
    du = (1 - r2) * u + mu * r2 * v
    dv = -mu * r2 * u + (1 - r2) * v
    return du, dv

# ----------------------
# ANIMATION SETUP
# ----------------------
fig, ax = plt.subplots()
im = ax.imshow(u, cmap='viridis', vmin=-1, vmax=1)
time_text = ax.text(0.02, 0.95, '', color='white', transform=ax.transAxes, fontsize=12)
ax.set_title("u concentration (FFT, implicit diffusion)")

# ----------------------
# Parameters for changes
# ----------------------
t_bifur = 20.0           # time at which diffusion and mu change
#d1_new, d2_new = 0.01, 0.01  # new diffusion coefficients
mu_new = 3.0             # new phase rotation

# ----------------------
# UPDATE FUNCTION
# ----------------------
def update(frame):
    global u, v, d1, d2, mu
    t = frame * substeps * dt  # current simulation time

    # change diffusion and mu at t_bifur
    if t >= t_bifur:
        # d1, d2 = d1_new, d2_new
        mu = mu_new
    for _ in range(substeps):
        # --- RK4 for reaction ---
        k1u, k1v = reaction(u, v)
        k2u, k2v = reaction(u + 0.5*dt*k1u, v + 0.5*dt*k1v)
        k3u, k3v = reaction(u + 0.5*dt*k2u, v + 0.5*dt*k2v)
        k4u, k4v = reaction(u + dt*k3u, v + dt*k3v)

        u_react = u + dt/6 * (k1u + 2*k2u + 2*k3u + k4u)
        v_react = v + dt/6 * (k1v + 2*k2v + 2*k3v + k4v)

        # --- implicit diffusion in Fourier space ---
        u_hat = np.fft.fftn(u_react)
        v_hat = np.fft.fftn(v_react)

        u = np.fft.ifftn(u_hat / (1 + dt*d1*K2)).real
        v = np.fft.ifftn(v_hat / (1 + dt*d2*K2)).real

    # update image and time text
    im.set_array(u)
    time_text.set_text(f"t = {t:.2f}")

    return [im, time_text]

# ----------------------
# RUN ANIMATION
# ----------------------
ani = animation.FuncAnimation(
    fig, update, frames=n_frames, blit=False, repeat=False
)
plt.show()