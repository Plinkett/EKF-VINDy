import pysindy as ps
import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
from pysindy.utils import linear_damped_SHO
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from ekf_vindy.jacobian_utils import sympify_str, differentiate_library, lambdify_library
np.random.seed(1000)  # Seed for reproducibility
import ekf_vindy.utils as utils
# Integrator keywords for solve_ivp
integrator_keywords = {}
integrator_keywords['rtol'] = 1e-12
integrator_keywords['method'] = 'LSODA'
integrator_keywords['atol'] = 1e-12

dt = 0.01
t_train = np.arange(0, 25, dt)
t_train_span = (t_train[0], t_train[-1])
x0_train = [2, 0]

# Generate trajectory on which to train, solved with RK45 
x_train = solve_ivp(linear_damped_SHO, t_train_span,
                    x0_train, t_eval=t_train, **integrator_keywords).y.T

# Fit the model, in our case we will use VINDy, but the library will come from
# PySINDy because it's useful. 

poly_order = 5
threshold = 0.05

model = ps.SINDy(
    optimizer=ps.STLSQ(threshold=threshold),
    feature_library=ps.PolynomialLibrary(degree=poly_order),
)

model.fit(x_train, t=dt)
model.print()

coeffs = model.coefficients()
print(f'coeffs: {coeffs}')
print(f'type(coeffs): {type(coeffs)}')
non_zero = utils.find_non_zero(coeffs)

print(f'non_zero: {non_zero}')
#################### automate the symbolic extraction of learned equations


# in general, define symbols
x0, x1 = sp.symbols('x0 x1')    

################à# get the feature names from the model
sindy_library_names = model.get_feature_names()
var_names = model.feature_names

var_symbols, library_symbols = sympify_str(var_names, sindy_library_names)
derivatives, sym_derivatives = differentiate_library(var_symbols, library_symbols)

print(f'var_symbols: {var_symbols}')
print(f'library_symbols: {library_symbols}')
print(f'derivatives: {sym_derivatives}')

"""  
How to track the coefficients? 

You got many equations for your dynamical system, so those are the x-axis.
You need to track some coefficients, you only need the column i.e. which term
out of those you need to track. I mean yeah, one big matrix in the end.

eq1 -> [3, 5, 6] (you track the third, fifth and sixth term)
eq2 -> [2, 3] same
...........................

so you end up with a matrix of size #eq x maxnumber_coefficients...

better a list of lists to be honest. 

[[3, 4, 5],
 [2, 3], 
 [],
 [3]] for example
"""