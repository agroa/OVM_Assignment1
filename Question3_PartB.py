import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

S0 = 170
mu = 0.1
sigma = 0.344
T = 1
M = 5000
L_values = [5, 10, 20, 50, 100]
deltaT = T / L_values[0]

def simulate_stock_price(S0, mu, sigma, L):
    dt = T / L
    S = S0
    for i in range(L):
        Z = np.random.randn()
        S = S * (1 + mu * dt + sigma * np.sqrt(dt) * Z)
    return S

def sim(L):
    final_prices = np.zeros(M)
    for j in range(M):
        final_prices[j] = simulate_stock_price(S0, mu, sigma, L)
    return final_prices

def lognorm_pdf(x, S0, mu, sigma, T):
    m = np.log(S0) + (mu - 0.5 * sigma ** 2) * T
    v = (sigma ** 2) * T
    return (1.0 / (x * np.sqrt(2 * np.pi * v))) * np.exp(-(np.log(x) - m) ** 2 / (2 * v))

for L in L_values:
    ST = sim(L).copy()
    plt.figure()
    counts, bins, _ = plt.hist(ST, bins=50, density=True, alpha=0.7, label=f"Simulated (L={L})")
    x = np.linspace(max(1e-12, bins[0]), bins[-1], 600)
    plt.plot(x, lognorm_pdf(x, S0, mu, sigma, T), linewidth=2, label="Lognormal PDF")
    plt.xlabel(r"$S_T$")
    plt.ylabel("Density")
    plt.title(f"Histogram of $S_T$ with Lognormal PDF (M={M}, L={L})")
    plt.legend()
    plt.grid(True)
    plt.show()

def simulate_paths_euler(S0, mu, sigma, T_end, dt, n_paths=8, seed=42):
    rng = np.random.default_rng(seed)
    n_steps = int(np.round(T_end / dt))
    t = np.linspace(0.0, T_end, n_steps + 1)
    S = np.empty((n_paths, n_steps + 1))
    S[:, 0] = S0
    for i in range(n_steps):
        Z = rng.normal(0.0, 1.0, size=n_paths)
        S[:, i + 1] = S[:, i] * (1 + mu * dt + sigma * np.sqrt(dt) * Z)
    return t, S

def running_sum_sq_returns(S, dt):
    R = (S[:, 1:] - S[:, :-1]) / S[:, :-1]
    RSS = np.cumsum(R ** 2, axis=1)
    return RSS

T_end_A = 0.5
dt_A = 5e-3
tA, SA = simulate_paths_euler(S0, mu, sigma, T_end_A, dt_A, n_paths=8, seed=123)
RSSA = running_sum_sq_returns(SA, dt_A)
tA_mid = tA[1:]

T_end_B = 0.1
dt_B = 1e-4
tB, SB = simulate_paths_euler(S0, mu, sigma, T_end_B, dt_B, n_paths=8, seed=456)
RSSB = running_sum_sq_returns(SB, dt_B)
tB_mid = tB[1:]

fig, axes = plt.subplots(2, 2, figsize=(10, 7))

ax = axes[0, 0]
for i in range(SA.shape[0]):
    ax.plot(tA, SA[i], alpha=0.5)
ax.set_title(r"$T=0.5,\ \Delta t=5\times10^{-3}$")
ax.set_xlabel("Time")
ax.set_ylabel("Stock Price")
ax.grid(alpha=0.3)

ax = axes[0, 1]
for i in range(SB.shape[0]):
    ax.plot(tB, SB[i], alpha=0.5)
ax.set_title(r"$T=0.1,\ \Delta t=10^{-4}$")
ax.set_xlabel("Time")
ax.set_ylabel("Stock Price")
ax.grid(alpha=0.3)

ax = axes[1, 0]
for i in range(RSSA.shape[0]):
    ax.plot(tA_mid, RSSA[i], alpha=0.5)
ax.plot(tA, sigma ** 2 * tA, linestyle=":", linewidth=2)
ax.set_xlabel("Time")
ax.set_ylabel("Running Sum of Squared Increments")
ax.set_title("Running Sum of Squared Increments ($T=0.5$)")
ax.grid(alpha=0.3)

ax = axes[1, 1]
for i in range(RSSB.shape[0]):
    ax.plot(tB_mid, RSSB[i], alpha=0.5)
ax.plot(tB, sigma ** 2 * tB, linestyle=":", linewidth=2)
ax.set_xlabel("Time")
ax.set_ylabel("Running Sum of Squared Increments")
ax.set_title("Running Sum of Squared Increments ($T=0.1$)")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
