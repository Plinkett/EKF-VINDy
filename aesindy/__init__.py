"""
Most of the VINDy code is directly adapted from the VENI-VINDy-VICI paper by Conti, Kneifl et al. 
See: https://arxiv.org/abs/2405.20905

This just a PyTorch version of their TensorFlow code.

We setup a singleton class to manage the PyTorch device and dtype configuration.
This is useful to avoid having to pass the device and dtype around in the code.
The user must set the device and dtype before using the library.

Obviously, the backend for the differential equation zoo must be compatible
with the PyTorch configuration of the solvers. Device and type must match.

"""
from .torch_config import TorchConfig

torch_config = TorchConfig()