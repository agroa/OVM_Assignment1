import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# population_disc = np.random.uniform(low=0, high=10, size=100000)
# population_cont = np.random.uniform(low=0, high=10, size=100000)

population_cont = population_disc = np.random.uniform(low=0, high=10, size=100000)

n = 30
num_samples = 10000
sample_means_disc = []

for _ in range(num_samples):
    sample = np.random.choice(population_disc, size=n, replace=True)
    sample_means_disc.append(np.mean(sample))

sample_means_disc = np.array(sample_means_disc)

sample_means_cont = []
for _ in range(num_samples):
    sample = np.random.choice(population_cont, size=n, replace=True)
    sample_means_cont.append(np.mean(sample))

sample_means_cont = np.array(sample_means_cont)



# Mean and std of sample means
mean_disc = np.mean(sample_means_disc)
std_disc = np.std(sample_means_disc)

# Plot histogram
plt.hist(sample_means_disc, bins=100, density=True, alpha=0.6, color='g', edgecolor='black')

# Overlay normal distribution
x = np.linspace(min(sample_means_disc), max(sample_means_disc), 100)
plt.plot(x, norm.pdf(x, mean_disc, std_disc), 'r', lw=2)
plt.title('CLT Simulation with Normal Fit Disc.')
plt.show()


mean_cont = np.mean(sample_means_cont)
std_cont = np.std(sample_means_cont)

x = np.linspace(min(sample_means_cont), max(sample_means_cont), 100)
plt.hist(sample_means_cont, bins=30, density=True, alpha=0.6, color='skyblue', edgecolor='black')
plt.plot(x, norm.pdf(x, mean_cont, std_cont), 'r', lw=2)
plt.title('CLT Simulation with Normal Fit Cont.')
plt.show()
