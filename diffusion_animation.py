import numpy as np
import matplotlib.pyplot as plt

# Spatial and temporal resolution
dx = 0.4           # spatial step
dt = 0.05          # time step
Tfinal = 20        # final time
steps = int(Tfinal / dt)

# Domain
x = np.arange(-10, 10+dx, dx)
y = np.arange(-10, 10+dx, dx)
X, Y = np.meshgrid(x, y)
N = len(x)  # number of grid points per axis

# PDE parameters
d1, d2 = 0.01, 0.01
mu = 1.0
beta = 1.0  # for initial condition

theta = np.arctan2(Y, X)
U = 0.5 * np.tanh(beta*(np.sqrt(X**2 + Y**2) - 3)) * np.cos(theta)
V = 0.5 * np.tanh(beta*(np.sqrt(X**2 + Y**2) - 3)) * np.sin(theta)

kx = 2*np.pi*np.fft.fftfreq(N, d=dx)
ky = 2*np.pi*np.fft.fftfreq(N, d=dx)
KX, KY = np.meshgrid(kx, ky)
laplace_factor = -(KX**2 + KY**2)

def f(U, V):
    return (1 - (U**2 + V**2))*U + mu*(U**2 + V**2)*V

def g(U, V):
    return -mu*(U**2 + V**2)*U + (1 - (U**2 + V**2))*V

for _ in range(steps):
    # Fourier transforms
    U_hat = np.fft.fft2(U)
    V_hat = np.fft.fft2(V)
    
    # Diffusion in spectral space
    U_hat += dt * d1 * laplace_factor * U_hat
    V_hat += dt * d2 * laplace_factor * V_hat
    
    # Back to real space
    U = np.real(np.fft.ifft2(U_hat))
    V = np.real(np.fft.ifft2(V_hat))
    
    U += 0.01 * (np.random.rand(*U.shape) - 0.5)
    V += 0.01 * (np.random.rand(*V.shape) - 0.5)

    # Add reaction
    U += dt * f(U, V)
    V += dt * g(U, V)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(U, extent=[-5,5,-5,5], origin='lower', cmap='RdBu')
plt.title('U field')
plt.colorbar()

plt.subplot(1,2,2)
plt.imshow(V, extent=[-5,5,-5,5], origin='lower', cmap='RdBu')
plt.title('V field')
plt.colorbar()
plt.show()