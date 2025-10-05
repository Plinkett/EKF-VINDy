import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# ----------------------
# PARAMETERS
# ----------------------
L = 10                  # domain [-L, L]^2
Nx, Ny = 100, 100       # grid points
dx = dy = 2*L / Nx
dt = 0.01
total_time = 40.0
d1 = d2 = 0.04
mu = 1.0
substeps = 5

n_steps = int(total_time / dt)
n_frames = n_steps // substeps

# ----------------------
# GRID AND INITIAL CONDITION (real-valued)
# ----------------------
x = np.linspace(-L, L, Nx)
y = np.linspace(-L, L, Ny)
X, Y = np.meshgrid(x, y)

beta = 0.8

# Real-valued initial condition derived from tanh/cosh complex formula
Z = X + 1j*Y
u = np.tanh(beta*(X**2 + Y**2)) * np.real(np.cos(Z)) / np.cosh(beta*(X**2 + Y**2))
v = np.tanh(beta*(X**2 + Y**2)) * np.imag(np.cos(Z)) / np.cosh(beta*(X**2 + Y**2))

# ----------------------
# WAVENUMBERS FOR FFT
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
ax.set_title("u concentration (tanh/cosh initial condition)")

# ----------------------
# UPDATE FUNCTION
# ----------------------
def update(frame):
    global u, v
    for _ in range(20):
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

    # update image
    im.set_array(u)

    # update timer
    t = frame * substeps * dt
    time_text.set_text(f"t = {t:.2f}")

    return [im, time_text]

# ----------------------
# RUN ANIMATION
# ----------------------
ani = animation.FuncAnimation(fig, update, frames=n_frames, blit=False, repeat=False)
plt.show()
