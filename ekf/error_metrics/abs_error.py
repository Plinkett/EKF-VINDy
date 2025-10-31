"""
Utility class for managing ROM and FOM comparison. We will consider the absolute value, we can even show an animation as it evolves over time.
Or we can also show snapshots (which are easier to focus on).
"""
import numpy as np

def abs_error(rom_solution: np.ndarray, fom_solution: np.ndarray):
    """
    We compute the absolute difference between the FOM and ROM solutions.
    Tensors are expected to be of size (flattened_dimension, time_instances).
    The subsequent reshaping must be done by the user, since it is highly dependent on the PDE considered.
    """
    return np.abs(fom_solution - rom_solution)

def rel_error(rom_solution: np.ndarray, fom_solution: np.ndarray):
    """
    Same as abs_error, but we compute the relative error. Outputs a tensor of size time_instances. So we can see
    how this percentage (sorta) evolves over time.
    """
    return np.abs(fom_solution - rom_solution) / np.abs(fom_solution)


def rd_error(rom_solution: np.ndarray, fom_solution: np.ndarray, spatial_dim: int = 50):
    """
    Ad hoc error computation (with reshaping) for reaction-diffusion PDE.
    We expect a square domain.
    """
    abs_error = np.abs(fom_solution - rom_solution)
    num_time_instances = abs_error.shape[1]
    domain_area = spatial_dim * spatial_dim

    abs_error = abs_error.reshape((2, domain_area, num_time_instances))
    error_u = abs_error[0, : ,:].reshape((spatial_dim, spatial_dim, num_time_instances))
    error_v = abs_error[1, : ,:].reshape((spatial_dim, spatial_dim, num_time_instances))
    rel_error = rel_error(rom_solution, fom_solution)
    
    return error_u, error_v, rel_error

