import numpy as np
from matplotlib import pyplot as plt
from Monte_Carlo import Monte_Carlo
from tabulate import tabulate

# Boltzmann's constant set to 1
k_B = 1

def E(p):
    return np.dot(p, p) / 2

epsilon_vector = [0.01, 0.1, 1]
dimensions = 3
N = 100
N_sweeps = 10000
table = np.zeros((len(epsilon_vector),5))
i = 0
for epsilon in epsilon_vector:
    p, E_d, single_particle_energy = Monte_Carlo(dimensions, E, N, N_sweeps, epsilon)

    # plot log(f(E_d))
    counts, bins = np.histogram(E_d)
    log_counts = np.log(counts)
    plt.figure(f"log(f(E_d) with N_sweeps= {N_sweeps} and epsilon = {epsilon}")
    plt.bar(bins[0:-1], log_counts, width=bins[1:] - bins[0:-1], color = 'tab:blue', align="edge")
    plt.xlabel(r"$E_d$")
    plt.ylabel(r"log$(\, f(E_d))$")
    plt.ylim([np.floor(np.amin(log_counts)), np.ceil(np.amax(log_counts))])
    plt.show()

    # calculate the midpoints of the bins
    bins_midpoints = (bins[0:-1] + bins[1:])/2
    # use the midpoints to calculate the line of best fit of the log(f) graphs
    slope, intercept = np.polyfit(bins_midpoints, log_counts, 1)
    T_approx_slope = -1/(slope*k_B)
    # average E_d
    mean_E_d = np.sum(E_d) / (N * N_sweeps)
    table[i,:] = [epsilon, N_sweeps, slope, T_approx_slope, mean_E_d]
    i += 1


head = ["epsilon", "N_sweeps", "gradient", "-1/gradient", "mean of E_d"]
print(tabulate(table, headers = head))


# Calculating the sample mean and variance of both approximations of T
# Sample size
samples = 50
T_approx_slope = np.zeros(samples)
mean_E_d = np.zeros(samples)
epsilon = 1
for i in range(samples):
    p, E_d, single_particle_energy = Monte_Carlo(dimensions, E, N, N_sweeps, epsilon)

    # calculate log(f(E_d))
    counts, bins = np.histogram(E_d)
    log_counts = np.log(counts)
    # calculate the midpoints of the bins
    bins_midpoints = (bins[0:-1] + bins[1:])/2
    # use the midpoints to calculate the line of best fit of the log(f) graphs
    slope, intercept = np.polyfit(bins_midpoints, log_counts, 1)
    T_approx_slope[i] = -1/(slope*k_B)
    # average E_d
    mean_E_d[i] = np.sum(E_d) / (N * N_sweeps)

# find sample mean for the estimate of T from the gradient
T_approx_slope_mean = np.sum(T_approx_slope)/samples
print(f"sample mean for the estimate of T from the gradient = {T_approx_slope_mean}")
# find sample variance for the estimate of T from the gradient
T_approx_slope_variance = np.sum((T_approx_slope - T_approx_slope_mean)**2)/(samples - 1)
print(f"sample variance for the estimate of T from the gradient = {T_approx_slope_variance}")
# find sample mean for the estimate of T from the mean of E_d
mean_E_d_mean = np.sum(mean_E_d)/samples
print(f"sample mean for the estimate of T from the mean of E_d = {mean_E_d_mean}")
# find sample variance for the estimate of T from the mean of E_d
mean_E_d_variance = np.sum((mean_E_d - mean_E_d_mean)**2)/(samples - 1)
print(f"sample variance for the estimate of T from the mean of E_d = {mean_E_d_variance}")