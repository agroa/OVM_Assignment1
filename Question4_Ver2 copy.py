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
E = 170
sigma = 0.05
r = 0.03
q = 0.01
max_price = 300
price_step = 30

#----------------Generate Random Stock Price Movements-----------------#
# S0 = 170 #Starting Price
mu = 0.03
period = 1
L = 1/20
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
S0 = np.arange(max_price, -1, -price_step)
time_steps = int(period / L)
numcols = time_steps + 1
numrows = S0.size

stock_paths = []
for starting_price in S0:
    prices = [starting_price]
    for i in range(time_steps):
        Y = np.random.normal(0, 1)
        s_new = prices[-1] * (1 + mu * L + sigma * np.sqrt(L) * Y)
        prices.append(s_new)
    stock_paths.append(prices)


time = np.arange(time_steps + 1)
# S0_grid, time_grid = np.meshgrid(S0, time)

Z = np.array(stock_paths)

df = pd.DataFrame(stock_paths)

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

#----------------Boundary Conditions-----------------#
# Proper independent initialization
max_time_price = [[0 for _ in range(numcols)] for _ in range(numrows)]
zero_price_array = [[0 for _ in range(numcols)] for _ in range(numrows)]

# Fill only the final column and final row
for row in range(numrows):
    # Final (rightmost) column: payoff at maturity
    max_time_price[row][numcols - 1] = max(E - stock_paths[row][numcols - 1], 0)

for column in range(numcols):
    # Bottom (lowest S) row: price when S=0
    zero_price_array[numrows - 1][column] = E * np.exp(-r * (numcols - column))



# print(max_time_price)
print(zero_price_array)





def genVsArray(price_array):
    



    return None
