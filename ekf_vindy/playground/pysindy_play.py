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

#################### automate the symbolic extraction of learned equations

# in general, define symbols
x0, x1 = sp.symbols('x0 x1')    

################à# get the feature names from the model
sindy_library_names = model.get_feature_names()
var_names = model.feature_names

print(f'var_names: {var_names}')
var_str = " ".join(var_names)
output = sp.symbols(var_str)
print(type(output[0]))




print(".--------")
# what is sindy_library_names? they are strings
#print("Feature names:", type(sindy_library_names[0]))


############## sympify everything
terms = []
for term in sindy_library_names:
    # Replace '^' with '**' for exponentiation in SymPy and remove any spaces
    # this may need to be modified for other weird symbols you may need
    term = term.replace('^', '**').replace(' ', '*') # term is still a string
    try:
        terms.append(sp.sympify(term, locals={'x0': x0, 'x1': x1}))
    except SyntaxError as e:
        print(f"Error parsing term '{term}': {e}")

print(f'LEN(VARIABLES): {len(var_names)}')
print(f'LEN(LIBRARY): {len(terms)}')
#mmm these are sympy types... is this the output of eval? a sympy thingy? except for scalars, which are ints and whatnot
#print(type(terms[0]))

############# Now we have sympy terms, get lambda functions out of them

# takes sympy symbols x0 and x1, which will be matched with the sympy symbol-format version of 
# our functions
sindy_library = [sp.lambdify((x0, x1), expr) for expr in terms]
# for term in terms:
#     print(term)

############ Now do the same with the derivatives for the Jacobian

# first see the output of sp.diff given an expression (sympy symbol) and the var (x0, also symbol)
diff=sp.diff(terms[7], x0) # this is a sympy symbol, which can be lambdified 
sindy_library_dx0 = [sp.lambdify((x0, x1), sp.diff(expr, x0)) for expr in terms]
for expr in terms:
    print(sp.diff(expr, x0))

# I don't know what "term" type is
# but yeah this can be automated and simpler