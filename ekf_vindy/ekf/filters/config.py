import numpy as np
from typing import List
from dataclasses import dataclass

"""
Contains all variables who dictate the dynamical system definition and its evolution (e.g., process and measurement noise)
"""
@dataclass
class DynamicsConfig:
    variables: List[str]
    library_terms: List[str]
    tracked_terms: List[List[str]]
    initial_coeffs: np.ndarray
    Q: np.ndarray
    R: np.ndarray