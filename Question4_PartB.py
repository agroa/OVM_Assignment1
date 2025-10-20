import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

mu = 0.1
sigma = 0.344
q = 0.01
E = 170
S0 = 170
r = 0.05
dt = 1/(365*2)
t = 0
S = S0
price_path = [S0]
time_path = [0]
strike = [E]

# ---- simulate price path ----
while t < 1:
    Y = np.random.normal(0,1)
    dS = (mu - q) * S * dt + sigma * S * np.sqrt(dt) * Y
    S += dS
    t += dt
    price_path.append(S)
    time_path.append(t)
    strike.append(E)

# ---- compute Delta_P along the path ----
delta_path = []

for i in range(len(time_path)):
    tau = 1 - time_path[i]
    if tau <= 0:
        if price_path[i] > E:
            delta_path.append(0)   # out of the money
        else:
            delta_path.append(-1)  # in the money
    else:
        d1 = (np.log(price_path[i]/E) + (r - q + 0.5*sigma**2)*tau) / (sigma * np.sqrt(tau))
        delta_put = np.exp(-q * tau) * (norm.cdf(d1) - 1)
        delta_path.append(delta_put)


# ---- plot ----
plt.plot(time_path, delta_path, label='Put Delta')
plt.xlabel("Time (years)")
plt.ylabel("Delta")
plt.legend()
plt.show()
