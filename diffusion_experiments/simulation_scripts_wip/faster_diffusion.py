import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from numba import njit

# ----------------------
# PARAMETERS
# ----------------------
Nx, Ny = 100, 100       # grid size
dx = dy = 1.0           # spatial step
dt = 0.01               # time step
n_steps = 2000          # total steps
save_every = 5          # save snapshots

d1, d2 = 2.0, 1.0       # diffusion
mu = 2.0                 # phase rotation

# ----------------------
# INITIAL CONDITIONS
# ----------------------
u = np.zeros((Nx, Ny))
v = np.zeros((Nx, Ny))

# half-circle bump + tiny noise for spiral
R = 10
cx, cy = Nx//2, Ny//2
for i in range(Nx):
    for j in range(Ny):
        if (i - cx)**2 + (j - cy)**2 < R**2 and i > cx:
            u[i,j] = 0.05
            v[i,j] = 0.0

u += 0.001*np.random.rand(Nx, Ny)
v += 0.001*np.random.rand(Nx, Ny)

# ----------------------
# NUMBA-ACCELERATED FUNCTIONS
# ----------------------
@njit
def laplacian(Z, dx):
    Nx, Ny = Z.shape
    L = np.empty_like(Z)
    for i in range(Nx):
        for j in range(Ny):
            ip = (i+1) % Nx
            im = (i-1) % Nx
            jp = (j+1) % Ny
            jm = (j-1) % Ny
            L[i,j] = (Z[ip,j] + Z[im,j] + Z[i,jp] + Z[i,jm] - 4*Z[i,j]) / dx**2
    return L

@njit
def step(u, v, dt, d1, d2, mu, dx):
    r2 = u**2 + v**2
    du = (1 - r2)*u + mu*r2*v
    dv = -mu*r2*u + (1 - r2)*v
    u_new = u + dt*du + dt*d1*laplacian(u, dx)
    v_new = v + dt*dv + dt*d2*laplacian(v, dx)
    return u_new, v_new

# ----------------------
# SIMULATION
# ----------------------
snapshots_u = []

for n in range(n_steps):
    u, v = step(u, v, dt, d1, d2, mu, dx)
    
    if n % save_every == 0:
        snapshots_u.append(u.copy())

snapshots_u = np.array(snapshots_u)

# ----------------------
# ANIMATION
# ----------------------
fig, ax = plt.subplots()
im = ax.imshow(snapshots_u[0], cmap='inferno', vmin=-0.1, vmax=0.5)
ax.set_title("u concentration (spiral)")

def update(frame):
    im.set_array(snapshots_u[frame])
    return [im]

ani = animation.FuncAnimation(fig, update, frames=len(snapshots_u), interval=30, blit=True)
plt.show()