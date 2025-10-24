import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as norm
from scipy.stats import lognorm

S0 = 170
sigma = 0.344
mu = 0.1
T = 1
L = 5
deltaT = T / L

n_simulations = 30

stock_prices = []

for j in range(n_simulations):
    stock_price = [S0]
    for i in range(L):
        Z = np.random.normal(0, 1)
        S_next = stock_price[i] * (1 + mu * deltaT + sigma * np.sqrt(deltaT) * Z)
        stock_price.append(S_next)
    stock_prices.append(stock_price)


stock_prices = np.array(stock_prices)


#------------------------------ 3(a) --------------------------------#
#Show historgram of all final prices and overlay theoretical distribution

# final_prices = stock_prices[:, -1]
# mu_ln = np.log(S0) + (mu - 0.5 * sigma**2) * T
# sigma_ln = sigma * np.sqrt(T)

# s = sigma_ln
# scale = np.exp(mu_ln)

# x_min = max(0.0, final_prices.min()*0.9)
# x_max = final_prices.max()*1.1
# x = np.linspace(x_min, x_max, 300)

# pdf_theoretical = lognorm.pdf(x, s=s, scale=scale)

# plt.figure(figsize=(8,5))
# plt.hist(final_prices, bins=50, density=True, alpha=0.6, edgecolor='black',
#          label='Simulated final prices (Euler-like)')
# plt.plot(x, pdf_theoretical, 'r-', lw=2, label='Theoretical GBM (lognormal) PDF')

# plt.xlabel("Final Stock Price $S_T$")
# plt.ylabel("Density")
# plt.title("Histogram of Final Prices vs. Theoretical GBM Distribution")
# plt.legend()
# plt.grid(alpha=0.2)
# plt.show()

#------------------------------ 3(b) --------------------------------#
t = np.linspace(0, T, L + 1)
# plt.plot(t, stock_prices.T, alpha=0.3) 
# plt.xlabel("Time")
# plt.ylabel("Stock Price")
# plt.title("Simulated Stock Price Paths")
# plt.show()


#--------------------------------Running Sum of Squared Increments --------------------------------#

# Compute running sum of squared increments for each simulation
running_sums = []

for path in stock_prices:
    increments = np.diff(path)
    sq_increments = increments**2
    running_sum = np.cumsum(sq_increments)
    running_sums.append(np.insert(running_sum, 0, 0))  # insert 0 at t=0 for alignment

running_sums = np.array(running_sums)

# Plot running sum of squared increments
plt.plot(t, running_sums.T, alpha=0.5)
plt.xlabel("Time")
plt.ylabel("Running Sum of Squared Increments")
plt.title("Running Sum of Squared Increments Over Time")
plt.grid(alpha=0.3)
plt.show()