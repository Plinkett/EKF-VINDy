import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# ----------------------
# PARAMETERS
# ----------------------
Nx, Ny = 100, 100       # grid size
dx = dy = 1.0           # spatial step
dt = 0.01               # time step
T = 20                  # total time
n_steps = int(T/dt)

# diffusion coefficients
d1, d2 = 5.0, 5.0

# phase rotation
mu = 2.0

# ----------------------
# INITIAL CONDITIONS
# ----------------------
u = np.zeros((Nx, Ny))
v = np.zeros((Nx, Ny))

# small circular perturbation in center
R = 5
cx, cy = Nx//2, Ny//2
for i in range(Nx):
    for j in range(Ny):
        if (i - cx)**2 + (j - cy)**2 < R**2:
            u[i,j] = 0.4
            v[i,j] = 0.1

# ----------------------
# HELPER FUNCTIONS
# ----------------------
def laplacian(Z):
    """Compute 2D Laplacian with periodic BCs"""
    return (
        np.roll(Z, 1, axis=0) + np.roll(Z, -1, axis=0) +
        np.roll(Z, 1, axis=1) + np.roll(Z, -1, axis=1) -
        4 * Z
    ) / dx**2

def reaction(u, v):
    """Reaction terms"""
    r2 = u**2 + v**2
    du = (1 - r2)*u + mu*r2*v
    dv = -mu*r2*u + (1 - r2)*v
    return du, dv

# ----------------------
# SETUP PLOTTING
# ----------------------
fig, ax = plt.subplots()
im = ax.imshow(u, cmap='inferno', vmin=-1, vmax=1, animated=True)
ax.set_title("u concentration")

def update(frame):
    global u, v
    # Run a few RK4 steps per frame
    for _ in range(1):
        # RK4 for reaction only
        k1u, k1v = reaction(u, v)
        k2u, k2v = reaction(u + 0.5*dt*k1u, v + 0.5*dt*k1v)
        k3u, k3v = reaction(u + 0.5*dt*k2u, v + 0.5*dt*k2v)
        k4u, k4v = reaction(u + dt*k3u, v + dt*k3v)
        u_react = u + dt/6*(k1u + 2*k2u + 2*k3u + k4u)
        v_react = v + dt/6*(k1v + 2*k2v + 2*k3v + k4v)
        
        # Add diffusion (explicit Euler)
        u = u_react + dt*d1*laplacian(u_react)
        v = v_react + dt*d2*laplacian(v_react)
    im.set_array(u)
    return [im]

ani = animation.FuncAnimation(fig, update, frames=200, blit=True)
plt.show()
