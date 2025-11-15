import torch

DEFAULT_DEVICE = torch.device("cpu")
DEFAULT_DTYPE = torch.float32 

"""
Singleton class to manage the PyTorch device and dtype configuration. See also __init__.py for usage.
"""

class TorchConfig:
    _instance = None
    _config_chosen = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TorchConfig, cls).__new__(cls)
        return cls._instance
    
    def setup_device_and_type(self, device=DEFAULT_DEVICE, dtype=DEFAULT_DTYPE):
        """
        Set the device and dtype for PyTorch tensors.

        Parameters
        ----------
        device : torch.device
            The device to use for PyTorch tensors.
        dtype : torch.dtype
            The data type to use for PyTorch tensors.
        """
        if self._config_chosen:
            raise RuntimeError("PyTorch device and dtype have already been set. If you want to change them, restart the backend.")
        if not isinstance(device, torch.device):
            raise TypeError(f"Device must be of type torch.device. {type(device)} is not.")
        if not isinstance(dtype, torch.dtype):
            raise TypeError(f"Dtype must be of type torch.dtype. {type(dtype)} is not.")

        self._device = device
        self._dtype = dtype
        self._config_chosen = True
        print(f"PyTorch device set to {self._device} and dtype set to {self._dtype}.")
        
    @property
    def device(self):
        if not self._config_chosen:
            raise RuntimeError("PyTorch device and dtype have not been set. Please set them first.")
        return self._device
    
    @property 
    def dtype(self):
        if not self._config_chosen:
            raise RuntimeError("PyTorch device and dtype have not been set. Please set them first.")
        return self._dtype