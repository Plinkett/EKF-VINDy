import pysindy as ps
import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
from pysindy.utils import linear_damped_SHO
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
np.random.seed(1000)  # Seed for reproducibility

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

# automate the symbolic extraction of learned equations

# in general, define symbols
x0, x1 = sp.symbols('x0 x1')    

# get the feature names from the model
sindy_library_names = model.get_feature_names()

# what is sindy_library_names? they are strings
print("Feature names:", type(sindy_library_names[0]))

terms = []
for term in sindy_library_names:
    # Replace '^' with '**' for exponentiation in SymPy and remove any spaces
    term = term.replace('^', '**').replace(' ', '*')
    try:
        terms.append(eval(term, {'u1': x0, 'u2': x1}))
    except SyntaxError as e:
        print(f"Error parsing term '{term}': {e}")

# I don't know what "term" type is
# but yeah this can be automated and simpler