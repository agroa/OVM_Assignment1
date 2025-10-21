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

# --- Simulate stock path ---
t = 0
S = S0
price_path = [S0]
time_path = [0]
strike = [E]

while t < 1:
    Y = np.random.normal(0,1)
    dS = (mu - q) * S * dt + sigma * S * np.sqrt(dt) * Y
    S += dS
    t += dt
    price_path.append(S)
    time_path.append(t)
    strike.append(E)

# --- Compute put Delta along the path ---
delta_path = []
for i in range(len(time_path)):
    tau = 1 - time_path[i]
    if tau <= 0:
        if price_path[i] > E:
            delta_path.append(0)
        else:
            delta_path.append(-1)
    else:
        d1 = (np.log(price_path[i]/E) + (r - q + 0.5*sigma**2)*tau) / (sigma * np.sqrt(tau))
        delta_put = np.exp(-q * tau) * (norm.cdf(d1) - 1)
        delta_path.append(delta_put)

# --- Delta hedging ---
A_path = delta_path.copy() 
D_path = [1]           
Pi_path = [A_path[0]*price_path[0] + D_path[0]]

for i in range(1, len(time_path)):
    S_prev = price_path[i-1]
    S_curr = price_path[i]
    A_prev = A_path[i-1]
    A_curr = A_path[i]
    D_prev = D_path[i-1]
    
    D_new = (A_prev - A_curr)*S_curr + (1 + r*dt)*D_prev + A_prev*q*S_prev*dt
    D_path.append(D_new)
    
    Pi_new = A_curr*S_curr + D_new
    Pi_path.append(Pi_new)

# --- Plot ---
fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

axs[0].plot(time_path, price_path, color='blue')
axs[0].plot(time_path, strike, color='red', linestyle='--')
axs[0].set_ylabel('Stock Price (S)')
axs[0].grid(True)

axs[1].plot(time_path, delta_path, color='orange')
axs[1].set_ylabel('Put Delta (A)')
axs[1].grid(True)

axs[2].plot(time_path, D_path, color='green')
axs[2].set_ylabel('Cash (D)')
axs[2].grid(True)

axs[3].plot(time_path, Pi_path, color='red')
axs[3].set_ylabel('Portfolio Value (Pi)')
axs[3].set_xlabel('Time (years)')
axs[3].grid(True)

plt.tight_layout()
plt.show()
