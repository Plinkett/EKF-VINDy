import numpy as np
import pysindy as ps
from typing import List
from pysindy.feature_library import PolynomialLibrary, FourierLibrary

n_input = 2
poly_lib = PolynomialLibrary(degree=3).fit(np.zeros((1, n_input)))
print(np.zeros((1, n_input)))
print(poly_lib.get_feature_names())

