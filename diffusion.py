import numpy as np
import matplotlib.pyplot as plt

N = 100          # grid size
dx = 1.0         # spatial step
dt = 0.01        # time step
d1, d2 = 0.2, 0.1  # diffusion coefficients
mu = 1.0         # reaction coupling
steps = 500      # number of time steps

# Create a grid
x = np.linspace(-5,5,N)
y = np.linspace(-5,5,N)
X, Y = np.meshgrid(x, y)

# Create a grid
x = np.linspace(-5,5,N)
y = np.linspace(-5,5,N)
X, Y = np.meshgrid(x, y)

# Initial condition (your tanh-like pattern)
beta = 1.0
U = np.tanh(beta * np.sqrt(X**2 + Y**2) * np.cos(X + Y - beta*np.sqrt(X**2+Y**2)))
V = np.tanh(beta * np.sqrt(X**2 + Y**2) * np.sin(X + Y - beta*np.sqrt(X**2+Y**2)))

def laplacian(Z):
    return (np.roll(Z,1,axis=0) + np.roll(Z,-1,axis=0) +
            np.roll(Z,1,axis=1) + np.roll(Z,-1,axis=1) - 4*Z) / dx**2

def f(U,V):
    return (1 - (U**2 + V**2))*U + mu*(U**2 + V**2)*V

def g(U,V):
    return -mu*(U**2 + V**2)*U + (1 - (U**2 + V**2))*V

# Time evolution
for n in range(steps):
    U_new = U + dt * (d1 * laplacian(U) + f(U,V))
    V_new = V + dt * (d2 * laplacian(V) + g(U,V))
    U, V = U_new, V_new

# Plot the final pattern
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(U, cmap='RdBu', origin='lower')
plt.title('U')
plt.colorbar()
plt.subplot(1,2,2)
plt.imshow(V, cmap='RdBu', origin='lower')
plt.title('V')
plt.colorbar()
plt.show()