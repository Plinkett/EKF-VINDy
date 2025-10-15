import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from numpy.fft import fft2, ifft2
from matplotlib.animation import FuncAnimation

np.random.seed(100)

integrator_keywords = {}
integrator_keywords['rtol'] = 1e-12
integrator_keywords['atol'] = 1e-12
integrator_keywords['method'] = 'RK45'  # switch to RK45 integrator


# Define the reaction-diffusion PDE in the Fourier (kx, ky) space
def reaction_diffusion(t, uvt, K22, d1, d2, mu, n, N):
    # Flattened Fourier coefficients of u and v
    ut = np.reshape(uvt[:N], (n, n))
    vt = np.reshape(uvt[N : 2 * N], (n, n))

    # Inverse FFT to reconstruct real-space fields
    u = np.real(ifft2(ut))
    v = np.real(ifft2(vt))

    # Auxiliary terms
    u3 = u ** 3
    v3 = v ** 3
    u2v = (u ** 2) * v
    uv2 = u * (v ** 2)

    # The actual RHS in Fourier space
    utrhs = np.reshape((fft2(u - u3 - uv2 + mu * u2v + mu * v3)), (N, 1))
    vtrhs = np.reshape((fft2(v - u2v - v3 - mu * u3 - mu * uv2)), (N, 1))
    uvt_reshaped = np.reshape(uvt, (len(uvt), 1))

    # Laplacian becomes just a multiplication in Fourier space
    uvt_updated = np.squeeze(
        np.vstack(
            (-d1 * K22 * uvt_reshaped[:N] + utrhs, 
             -d2 * K22 * uvt_reshaped[N:] + vtrhs)
        )
    )
    return uvt_updated


# Generate the data
T = 20.0
timestep = 0.05
t = np.linspace(0, T, int(T / timestep) + 1)

d1 = 0.01
d2 = 0.01

# Coupling parameter (this is what can be time-varying!!!)
mu = 1.4

L = 20  # Domain size in X and Y directions
spatial_step = 0.4
n = int(L / spatial_step)  # number of spatial points in X and Y directions
N = n * n  # total number of spatial points

x_uniform = np.linspace(-L / 2, L / 2, (int(L / spatial_step) + 1))

# The last point is the same as the first point, so we drop it
x = x_uniform[:n]
y = x_uniform[:n]

n2 = int(n / 2)


# Define Fourier wavevectors (kx, ky), i.e., the frequency mesh
kx = (2 * np.pi / L) * np.hstack((np.linspace(0, n2 - 1, n2), 
                                  np.linspace(-n2, -1, n2)))
ky = kx
X, Y = np.meshgrid(x, y)
KX, KY = np.meshgrid(kx, ky)
K2 = KX ** 2 + KY ** 2
K22 = np.reshape(K2, (N, 1))


# solution buffers
u = np.zeros((len(x), len(y), len(t)))
v = np.zeros((len(x), len(y), len(t)))

m = 1  # number of spirals
beta = 1.1
# Initial conditions
u[:, :, 0] = np.tanh(beta * np.sqrt(X ** 2 + Y ** 2)) * np.cos(
    m * np.angle(X + 1j * Y) - beta * (np.sqrt(X ** 2 + Y ** 2))
)
v[:, :, 0] = np.tanh(beta * np.sqrt(X ** 2 + Y ** 2)) * np.sin(
    m * np.angle(X + 1j * Y) - beta * (np.sqrt(X ** 2 + Y ** 2))
)

# uvt is the solution vector in Fourier space, so below
# we are initializing the 2D FFT of the initial condition, uvt0
uvt0 = np.squeeze(
    np.hstack(
        (np.reshape(fft2(u[:, :, 0]), (1, N)), 
         np.reshape(fft2(v[:, :, 0]), (1, N)))
    )
)

# Solve the PDE in the Fourier space, where it reduces to system of ODEs
print('Starting the time integration...')
uvsol = solve_ivp(
    reaction_diffusion, (t[0], t[-1]), y0=uvt0, t_eval=t, 
    args=(K22, d1, d2, mu, n, N), **integrator_keywords
)
print('Done with time integration.')
uvsol = uvsol.y

# Reshape things and ifft back into (x, y, t) space from (kx, ky, t) space
for j in range(len(t)):
    ut = np.reshape(uvsol[:N, j], (n, n))
    vt = np.reshape(uvsol[N:, j], (n, n))
    u[:, :, j] = np.real(ifft2(ut))
    v[:, :, j] = np.real(ifft2(vt))

print(f'u shape: {u.shape}, v shape: {v.shape}, t shape: {t.shape}')
np.savez("spiral_data.npz", u=u, v=v, t=t, x=x, y=y)
print('Done with reshaping.')

# # Choose which field to animate: u or v
# field = u  # shape: (nx, ny, nt)
# fig, ax = plt.subplots(figsize=(5,5))
# cax = ax.imshow(field[:, :, 0], origin='lower',
#                 extent=[x.min(), x.max(), y.min(), y.max()])

# vmin, vmax = field.min(), field.max()
# cax.set_clim(vmin, vmax)

# ax.set_xlabel('x')
# ax.set_ylabel('y')

# # Use ax.text instead of ax.set_title
# time_text = ax.text(0.5, 1.02, f't = {t[0]:.2f}', transform=ax.transAxes,
#                     ha='center', va='bottom', fontsize=12)

# def update(frame):
#     cax.set_data(field[:, :, frame])
#     time_text.set_text(f't = {t[frame]:.2f}')  # update the text
#     return [cax, time_text]

# # Use blit=False for safety
# anim = FuncAnimation(fig, update, frames=len(t), interval=50, blit=False)

# plt.show()