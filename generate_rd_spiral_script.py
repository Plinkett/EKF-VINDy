import numpy as np
import os
import argparse
from scipy.integrate import solve_ivp
from numpy.fft import fft2, ifft2
from tqdm import tqdm

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

def generate_simulations(T: float, dt: float, mu_values: np.ndarray,
                         d1: float, d2: float, m: int, beta: float):
    integrator_keywords = {}
    integrator_keywords['rtol'] = 1e-12
    integrator_keywords['atol'] = 1e-12
    integrator_keywords['method'] = 'RK45'  # switch to RK45 integrator

    # Time array
    t = np.linspace(0, T, int(T / dt) + 1)

    # Fixed spatial domain and grid
    L = 20
    spatial_step = 0.4
    n = int(L / spatial_step)
    N = n * n

    # Generate grid (skip last point due to periodicity)
    x_uniform = np.linspace(-L / 2, L / 2, (int(L / spatial_step) + 1))
    x = x_uniform[:n]
    y = x_uniform[:n]

    # Handle frequency domain 
    n2 = int(n / 2)
    kx = (2 * np.pi / L) * np.hstack((np.linspace(0, n2 - 1, n2), 
                                  np.linspace(-n2, -1, n2)))
    ky = kx
    X, Y = np.meshgrid(x, y)
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX ** 2 + KY ** 2
    K22 = np.reshape(K2, (N, 1))

    u_tot = np.zeros((len(x), len(y), len(t), len(mu_values)))
    v_tot = np.zeros((len(x), len(y), len(t), len(mu_values)))
    
    print("Starting simulations...")
    # Loop over mu values
    for i, mu in enumerate(tqdm(mu_values)):
        print(f"Simulating for mu = {mu}")
        u_tot[:, :, 0, i] = np.tanh(beta * np.sqrt(X ** 2 + Y ** 2)) * np.cos(
            m * np.angle(X + 1j * Y) - beta * (np.sqrt(X ** 2 + Y ** 2))
        )
        v_tot[:, :, 0, i] = np.tanh(beta * np.sqrt(X ** 2 + Y ** 2)) * np.sin(
            m * np.angle(X + 1j * Y) - beta * (np.sqrt(X ** 2 + Y ** 2))
        )

        uvt0 = np.squeeze(
            np.hstack(
                (np.reshape(fft2(u_tot[:, :, 0, i] ), (1, N)), 
                np.reshape(fft2(v_tot[:, :, 0, i]), (1, N)))
            )
        )

        # Time integration of discretized PDE
        uvsol = solve_ivp(
            reaction_diffusion, (t[0], t[-1]), y0=uvt0, t_eval=t, 
            args=(K22, d1, d2, mu, n, N), **integrator_keywords
        )

        uvsol = uvsol.y

        # Reshape things and ifft back into (x, y, t) space from (kx, ky, t) space
        for j in range(len(t)):
            ut = np.reshape(uvsol[:N, j], (n, n))
            vt = np.reshape(uvsol[N:, j], (n, n))
            u_tot[:, :, j, i] = np.real(ifft2(ut))
            v_tot[:, :, j, i] = np.real(ifft2(vt))

    print(f'u_tot.shape: {u_tot.shape}, v_tot.shape: {v_tot.shape}')
    print("Saving data...")
    filename = f"rd_spiral_mu_{mu_values[0]:.3f}_to_{mu_values[-1]:.3f}_d1_{d1}_d2_{d2}_m_{m}_beta_{beta}.npz"
    filepath = os.path.join("simulation_data/rd_spiral", filename)
    np.savez(filepath, u=u_tot, v=v_tot)

def main():
    parser = argparse.ArgumentParser(description="Simulation parameters")

    parser.add_argument("--T", type=float, default=40.0,
                        help="Total time (float), default 20")
    parser.add_argument("--dt", type=float, default=0.05,
                        help="Time step (float), default 0.05")
    parser.add_argument("--beta", type=float, default=1.1,
                        help="Beta parameter (float), default 1.1")
    # Mu range
    parser.add_argument("--mu_start", type=float, default=0.8, help="Start of mu range")
    parser.add_argument("--mu_end", type=float, default=1.5, help="End of mu range")
    parser.add_argument("--mu_points", type=int, default=2, help="Number of points in mu range")
    
    parser.add_argument("--m", type=int, default=1, help="Number of spirals (int), default 1")
    parser.add_argument("--d1", type=float, default=0.01,
                        help="Diffusion parameter 1 (float), default 0.01")
    parser.add_argument("--d2", type=float, default=0.01,
                        help="Diffusion parameter 2 (float), default 0.01")
    args = parser.parse_args()
    
    # Generate arrays of values
    mu_values = np.linspace(args.mu_start, args.mu_end, args.mu_points)

    # Create directory if it does not exist
    if not os.path.exists("simulation_data/rd_spiral"):
        os.makedirs("simulation_data/rd_spiral")

    generate_simulations(args.T, args.dt, mu_values, args.d1, args.d2,
                         args.m, args.beta)

if __name__ == "__main__":
    main()

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