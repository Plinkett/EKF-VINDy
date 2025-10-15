import numpy as np
from scipy.integrate import solve_ivp
from numpy.fft import fft2, ifft2


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


if __name__ == "__main__":
    print('Hello')