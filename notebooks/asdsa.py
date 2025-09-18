# import numpy as np
# from scipy.integrate import solve_ivp
# import matplotlib.pyplot as plt

# # Cubic-quintic Duffing oscillator
# def duffing_cq(t, y, alpha=1.0, beta=1.0, gamma=1.0):
#     x, v = y
#     dxdt = v
#     dvdt = -alpha*x - beta*x**3 - gamma*x**5
#     return [dxdt, dvdt]

# # Energy function
# def energy(x, v, alpha=1.0, beta=1.0, gamma=1.0):
#     return 0.5*v**2 + 0.5*alpha*x**2 + 0.25*beta*x**4 + (1/6)*gamma*x**6

# # Initial condition
# x0 = [1.0, 0.0]   # start at x=1, v=0
# t_span = (0, 50)

# # Integrate with adaptive solver
# sol = solve_ivp(duffing_cq, t_span, x0, method="RK45", rtol=1e-9, atol=1e-12)

# # Compute energy along trajectory
# E = energy(sol.y[0], sol.y[1])

# # Plot energy
# plt.plot(sol.t, E)
# plt.xlabel("Time")
# plt.ylabel("Energy")
# plt.title("Energy conservation in cubic–quintic Duffing oscillator")
# plt.show()

# import numpy as np
# from scipy.integrate import solve_ivp
# import matplotlib.pyplot as plt

# # Cubic–quintic Duffing oscillator
# def duffing_cq(t, y, alpha=1.0, beta=1.0, gamma=1.0):
#     x, v = y
#     dxdt = v
#     dvdt = -alpha*x - beta*x**3 - gamma*x**5
#     return [dxdt, dvdt]

# # Initial condition
# x0 = [1.0, 0.0]   # start at x=1, v=0
# t_span = (0, 50)

# # Integrate with adaptive solver
# sol = solve_ivp(duffing_cq, t_span, x0, method="RK45",
#                 rtol=1e-9, atol=1e-12, dense_output=True)

# # Uniform evaluation points for plotting
# t_eval = np.linspace(*t_span, 5000)
# y = sol.sol(t_eval)  # y[0] = x(t), y[1] = v(t)

# # Plot
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# # Trajectory x(t)
# ax1.plot(t_eval, y[0])
# ax1.set_xlabel("Time")
# ax1.set_ylabel("x(t)")
# ax1.set_title("Cubic–Quintic Duffing Oscillator: Trajectory")

# # Phase-space orbit (x, v)
# ax2.plot(y[0], y[1])
# ax2.set_xlabel("x")
# ax2.set_ylabel("v")
# ax2.set_title("Phase-Space Orbit")

# plt.tight_layout()
# plt.show()
