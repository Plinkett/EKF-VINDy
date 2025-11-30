"""
Simple callback blueprint for managing projecting on to POD basis for use with autoencoder.
"""
from abc import ABC, abstractmethod
from ekf_vindy.ekf.filters.state import State
import numpy as np

class EKFCallback(ABC):
    """ 
    Base class for EKF callbacks, pre-processing and post-processing routines.
    We don't submit to any particular signature for our callback...
    """
    
    @abstractmethod
    def preprocess(self, observation: np.ndarray):
        raise NotImplementedError
    
    @abstractmethod
    def postprocess(self, state_upd: State):
        raise NotImplementedError
    
class PODProjection(EKFCallback):
    """
    Concrete implementation of callbacks for POD modes projection.
    We expect the pod_modes to have shape (num_states, k), where k is the number of POD modes,
    and we the observations to be of shape (num_states, ).
    
    TODO: correct the shape assumptions if needed.
    """
    
    def __init__(self, pod_modes: np.ndarray):
        super().__init__()
        self.pod_modes = pod_modes
        self.k = pod_modes.shape[1]
        
    def preprocess(self, observation: np.ndarray):
        """
        Project onto the top-K POD modes
        """
        return self.pod_modes @ observation.T
    
    def postprocess(self, state_upd: State):
        """
        We do not implement it here. In the future, if you want to save the results
        of the online inference you can do it here, for example.  
        """
        pass
        
    
    
    