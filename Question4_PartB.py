import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- Parameters ---
mu = 0.1
sigma = 0.344
q = 0.01
E = 170
S0 = 170
r = 0.05
dt = 1/(365*2)

# --- Black-Scholes put price function ---
def bs_put_price(S, K, T, r, q, sigma):
    if T <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r*T) * norm.cdf(-d2) - S * np.exp(-q*T) * norm.cdf(-d1)

# --- Simulate stock path (single path) ---
t = 0.0
S = S0
price_path = [S0]
time_path = [0.0]
strike = [E]

while t < 1.0:
    Y = np.random.normal(0, 1)
    dS = (r - q) * S * dt + sigma * S * np.sqrt(dt) * Y
    S += dS
    t += dt
    price_path.append(S)
    time_path.append(t)
    strike.append(E)

# --- Compute put Delta along the path ---
delta_path = []
for i in range(len(time_path)):
    tau = 1.0 - time_path[i]
    S_i = price_path[i]
    if tau <= 0:
        delta_path.append(0.0 if S_i > E else -1.0)
    else:
        d1 = (np.log(S_i / E) + (r - q + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
        # put delta (Black-Scholes)
        delta_put = np.exp(-q * tau) * (norm.cdf(d1) - 1.0)
        delta_path.append(delta_put)


put_price_0 = bs_put_price(S0, E, 1.0, r, q, sigma)
A_path = delta_path.copy()
D_path = [put_price_0 - A_path[0] * S0]   # D0 = V0 - A0 * S0
Pi_path = [A_path[0] * price_path[0] + D_path[0]]


for i in range(1, len(time_path)):
    S_prev = price_path[i-1]
    S_curr = price_path[i]
    A_prev = A_path[i-1]
    A_curr = A_path[i]
    D_prev = D_path[i-1]

    D_new = (1.0 + r * dt) * D_prev - (A_curr - A_prev) * S_curr
    D_path.append(D_new)

    Pi_new = A_curr * S_curr + D_new
    Pi_path.append(Pi_new)

S_T = price_path[-1]
put_payoff = max(E - S_T, 0.0)
hedge_error = Pi_path[-1] - put_payoff

print(f"S(T) = {S_T:.4f}")
print(f"Put payoff        = {put_payoff:.4f}")
print(f"Replicating Pi(T) = {Pi_path[-1]:.4f}")
print(f"Hedging error Pi(T) - payoff = {hedge_error:.6f}")

# --- Plot ---
fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

axs[0].plot(time_path, price_path, color='blue', lw=1)
axs[0].plot(time_path, strike, color='red', linestyle='--', lw=1)
axs[0].set_ylabel('Stock Price (S)')
axs[0].grid(True)

axs[1].plot(time_path, delta_path, color='orange', lw=1)
axs[1].set_ylabel('Put Delta (A)')
axs[1].grid(True)

axs[2].plot(time_path, D_path, color='green', lw=1)
axs[2].set_ylabel('Cash (D)')
axs[2].grid(True)

axs[3].plot(time_path, Pi_path, color='red', lw=1)
axs[3].axhline(put_payoff, color='black', linestyle='--', lw=1, label='Final Put Payoff')
axs[3].set_ylabel('Portfolio Value (Pi)')
axs[3].set_xlabel('Time (years)')
axs[3].grid(True)
axs[3].legend()

plt.tight_layout()
plt.show()
