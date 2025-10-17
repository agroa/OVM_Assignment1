import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

#-----------------------------------------PDE---------------------------------------------#
# dV/dt + 0.5 * sigma^2 * S^2 * d2V/dS2 + r * S * dV/dS - (1-r) * q * S * dV/dS - r * V = 0

#----------------------BCs-------------------#
# P(S,T) = max(E-S(T),0)
# P(0,t) = E * exp(-r(T-t))
# P(s,t) = 0 for large S >> E


#-------------------Constants----------------#
sigma = 0.05
r = 0.03
q = 0.01
max_price = 300
price_step = 0.5

#----------------Generate Random Stock Price Movements-----------------#
# S0 = 170 #Starting Price
mu = 0.03
period = 0.1
L = 1/3650
period_step = L


# stock_price = []
# time = []
# stock_price.append(S0)
# time.append(0)
# for i in range(int(period/L)):
#     Y = np.random.normal(0,1)
#     s_new = stock_price[i] * (1 + mu * L + sigma * np.sqrt(L) * Y)
#     stock_price.append(s_new)
#     time.append(i + 1)



# plt.plot(time, stock_price)
# plt.show()


# #----------------Generate Random Stock Price Movements Starting from a Range of S0-----------------#
S0 = np.arange(0, max_price, price_step)
steps = int(period / L)

stock_paths = []
for starting_price in S0:
    prices = [starting_price]
    for i in range(steps):
        Y = np.random.normal(0, 1)
        s_new = prices[-1] * (1 + mu * L + sigma * np.sqrt(L) * Y)
        prices.append(s_new)
    stock_paths.append(prices)


time = np.arange(steps + 1)
S0_grid, time_grid = np.meshgrid(S0, time)

Z = np.array(stock_paths).T

df = pd.DataFrame(stock_paths).T

df.insert(0, 'Time (days)', time)

df.to_csv('stock_paths.csv', index=False)

#Plot the figure

# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(111, projection='3d')

# for j in range(len(S0)):
#     ax.plot(time, [S0[j]] * len(time), Z[:, j])

# ax.set_xlabel('Time (days)')
# ax.set_ylabel('Starting Price')
# ax.set_zlabel('Stock Price')
# ax.set_title('Simulated Stock Price Paths for Different Starting Prices')

# plt.show()



#----------------Setting Up Mesh for Numerical Method-----------------#
# a = 0.5 * (sigma**2) * (S**2)
# b = (r - q) * S
# c = r


# options = np.empty((int(max_price/price_step), int(period/L)))
# stock = Z

# for j in range(int(period/L)):
#     for i in range(int(max_price/price_step)):
#         options[i][j] = ((0.5 * (sigma**2) * stock[i][j]) * period_step * (2 * options[i][j-1] - options[i][j-2]) - (r-q) * stock[i][j] * period_step * price_step * options[i][j-1] - (price_step**2) * options[i-1][j]) / (0.5 * (sigma**2) * stock[i][j] * period_step - (r-q) * stock[i][j] * period_step * price_step - r * period_step * (price_step**2) - (price_step**2))


#----------------Setting Up Mesh for Numerical Method (Bottom-Right Start)-----------------#
S = np.arange(0, max_price + price_step, price_step) 
num_S = len(S)
num_T = int(period/L) + 1 


stock = Z

E = 170

options = np.zeros((num_S, num_T))
options[:, -1] = np.maximum(E - S, 0)
options[0, :] = E * np.exp(-r * (time[-1] - time)) 
options[-1, :] = 0 


for j in range(num_T - 2, -1, -1):
    for i in range(1, num_S - 1):
        S_val = S[i]
        options[i, j] = (
            (0.5 * sigma**2 * S_val**2 * period_step * (2*options[i, j+1] - options[i, j+1] - options[i, j+1])
             - (r-q) * S_val * period_step * price_step * options[i, j+1]
             - price_step**2 * options[i-1, j])
            / (0.5 * sigma**2 * S_val**2 * period_step 
               - (r-q) * S_val * period_step * price_step 
               - r * period_step * price_step**2
               - price_step**2)
        )



fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111, projection='3d')


S_grid, T_grid = np.meshgrid(S, time) 
Options_plot = options.T 



surf = ax.plot_surface(S_grid, T_grid, Options_plot, cmap='viridis', edgecolor='k', alpha=0.8)


ax.set_xlabel('Stock Price S')
ax.set_ylabel('Time (days)')
ax.set_zlabel('Option Value')
ax.set_title('Option Value Surface for Different Stock Prices Over Time')


fig.colorbar(surf, shrink=0.5, aspect=10)

plt.show()
