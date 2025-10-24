import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def bs_call(S, K, r, sigma, T):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def implied_vol(C_market, S, K, r, T, tol=1e-6, max_iter=100):
    low, high = 1e-6, 1.0
    for _ in range(max_iter):
        mid = (low + high) / 2
        price = bs_call(S, K, r, mid, T)
        if abs(price - C_market) < tol:
            return mid
        if price > C_market:
            high = mid
        else:
            low = mid
    return mid

strike_prices = np.array([24.5, 25, 25.5, 26, 27, 27.5, 28, 28.5, 29, 29.5], float)
market_prices = np.array([1.5, 1.71, 1.71, 2.34, 2.58, 2.7, 2.85, 3.5, 4.25, 4.25], float)

S0 = 27.24
r = 0.05
T = 1/12

implied_vols = [implied_vol(C, S0, K, r, T) for C, K in zip(market_prices, strike_prices)]

plt.plot(strike_prices, implied_vols, 'kd-')
plt.axvline(S0, color='k', linestyle='--')
#plt.text(S0 + 30, min(implied_vols) + 0.001, 'Current asset price', fontsize=9)
plt.xlabel("Strike price")
plt.ylabel("Implied volatility")
plt.show()
