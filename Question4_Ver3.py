import numpy as np
import matplotlib.pyplot as plt

#-------------------Constants----------------#
sigma = 0.05
r = 0.03
q = 0.01
max_price = 300
price_step = 5
E = 170  # Strike price

# Simulation parameters
mu = 0.03
period = 0.1  # in years
L = 1/3650    # timestep
steps = int(period / L)
S0 = np.arange(0, max_price + price_step, price_step)
num_paths = 100  # number of paths per starting price

#----------------Generate Stock Paths----------------#
stock_paths_all = []

for s0 in S0:
    paths = []
    for _ in range(num_paths):
        prices = [s0]
        for _ in range(steps):
            Y = np.random.normal(0, 1)
            s_new = prices[-1] * (1 + mu*L + sigma*np.sqrt(L)*Y)
            prices.append(s_new)
        paths.append(prices)
    stock_paths_all.append(np.array(paths))  # shape: (num_paths, steps+1)

time = np.arange(steps + 1) * L  # in years

#----------------Calculate Option Values Pathwise----------------#
option_values_surface = np.zeros((len(S0), steps + 1))

for i, paths in enumerate(stock_paths_all):
    # For each time step, compute expected discounted payoff from that time to maturity
    for t in range(steps + 1):
        remaining_time = period - t*L
        payoffs = np.maximum(E - paths[:, t:], 0)  # payoff from time t onwards
        # discount from time t to today
        discounted = payoffs * np.exp(-r * remaining_time)
        # mean across paths
        option_values_surface[i, t] = np.mean(discounted[:, 0])  # value at time t

# Transpose so we have shape (time, starting price) for plotting
Options_plot = option_values_surface.T

#----------------3D Plot----------------#
S_grid, T_grid = np.meshgrid(S0, time)

fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(S_grid, T_grid, Options_plot, cmap='viridis', edgecolor='k', alpha=0.8)

ax.set_xlabel('Starting Stock Price S0')
ax.set_ylabel('Time (years)')
ax.set_zlabel('Option Value')
ax.set_title('Pathwise Monte Carlo Option Value Surface')
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.show()
