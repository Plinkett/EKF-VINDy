import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # show only errors
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # disable GPU
warnings.filterwarnings('ignore')

import tensorflow as tf
import scipy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from vindy import SindyNetwork
from vindy.libraries import PolynomialLibrary
from vindy.layers import SindyLayer, VindyLayer
from vindy.distributions import Gaussian, Laplace
from vindy.callbacks import (
    SaveCoefficientsCallback,
)
from vindy.utils import add_lognormal_noise
