import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Parameters
# -------------------------
K = 170.0       # Strike
T = 1.0         # maturity
a = 0.3         # sigma
b = 0.02        # r
c = 0.01        # q
S_max = 5 * K   # large S boundary
M = 200         # spatial steps
N = 400         # time steps
# -------------------------

dS = S_max / M
dt = T / N
S = np.linspace(0.0, S_max, M + 1)
t_grid = np.linspace(0.0, T, N + 1)

U = np.zeros((M + 1, N + 1))
U[:, -1] = np.maximum(K - S, 0.0)

def boundary_low(t):
    # U(0,t) = const * exp(-b*(T - t))
    return K * np.exp(-b * (T - t))

def boundary_high(t):
    return 0.0

A = 0.5 * a**2 * S**2

B1 = b * S                
B2 = - (1 - b) * c * S     

B_total = B1 + B2

# FD stencils for interior nodes
i_vals = np.arange(1, M)
alpha = A[i_vals] / dS**2 - B_total[i_vals] / (2.0 * dS)
beta  = -2.0 * A[i_vals] / dS**2
gamma = A[i_vals] / dS**2 + B_total[i_vals] / (2.0 * dS)


lower = 0.5 * alpha
main  = 1.0 / dt + 0.5 * beta - 0.5 * b
upper = 0.5 * gamma

def thomas_solve(a_lower, a_main, a_upper, d):
    n = len(a_main)
    c_prime = np.zeros(n)
    d_prime = np.zeros(n)
    x = np.zeros(n)

    c_prime[0] = a_upper[0] / a_main[0]
    d_prime[0] = d[0] / a_main[0]

    for i in range(1, n):
        denom = a_main[i] - a_lower[i] * c_prime[i-1]
        c_prime[i] = a_upper[i] / denom if i < n-1 else 0.0
        d_prime[i] = (d[i] - a_lower[i] * d_prime[i-1]) / denom

    x[-1] = d_prime[-1]
    for i in range(n-2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]
    return x

# Time march
for k in range(N, 0, -1):
    t_now = t_grid[k]       # known time level (later)
    t_new = t_grid[k - 1]   
    U_now = U[:, k]

    rhs = np.zeros(M - 1)
    for j, i in enumerate(i_vals):
        L_now = alpha[j] * U_now[i - 1] + beta[j] * U_now[i] + gamma[j] * U_now[i + 1]
        #CN Splitting
        rhs[j] = (1.0 / dt) * U_now[i] - 0.5 * L_now + 0.5 * b * U_now[i]

    
    rhs[0] -= lower[0] * boundary_low(t_new)
    rhs[-1] -= upper[-1] * boundary_high(t_new)

    
    a_lower = np.zeros(M - 1); a_lower[1:] = lower[1:]
    a_main  = main.copy()
    a_upper = np.zeros(M - 1); a_upper[:-1] = upper[:-1]

    u_inner = thomas_solve(a_lower, a_main, a_upper, rhs)

    U[0, k - 1] = boundary_low(t_new)
    U[M, k - 1] = boundary_high(t_new)
    U[1:M, k - 1] = u_inner


S_mesh, T_mesh = np.meshgrid(S, t_grid)
Z = U.T

fig = plt.figure(figsize=(11, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(S_mesh, T_mesh, Z, linewidth=0, antialiased=True)
ax.set_xlabel('S')
ax.set_ylabel('t')
ax.set_zlabel('U(S,t)')
ax.set_title('Numerical solution U(S,t) — Crank–Nicolson (exact PDE form)')
plt.show()
