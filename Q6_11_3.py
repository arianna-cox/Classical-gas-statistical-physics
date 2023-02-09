import numpy as np
from matplotlib import pyplot as plt
from Monte_Carlo import Monte_Carlo
from tabulate import tabulate
plt.rcParams.update({'font.size': 15})
k_B = 1

def E(p):
    return np.linalg.norm(p)

dimensions = 2
N = 100
N_sweeps = 1000

# First, plot histograms of E_d and of the single-particle energy for a = 1
epsilon = 1
for a in [1]:
    p, E_d, single_particle_energy = Monte_Carlo(dimensions, E, N, N_sweeps, epsilon)

    # plot log(f(E_d))
    counts, bins = np.histogram(E_d)
    log_counts = np.log(counts)
    plt.figure(f"log(f(E_d)) with N_sweeps= {N_sweeps} and epsilon = {epsilon}")
    plt.bar(bins[0:-1], log_counts, width=bins[1:] - bins[0:-1], color = 'coral', align="edge")
    plt.xlabel(r"$E_d$")
    plt.ylabel(r"log$(\, f(E_d))$")
    try:
        plt.ylim([np.floor(np.amin(log_counts)), np.ceil(np.amax(log_counts))])
    except:
        plt.ylim([0, np.ceil(np.amax(log_counts))])
    plt.savefig(f"Q6_log(f(E_d))_{a}_{N_sweeps}_{epsilon}.eps", bbox_inches ="tight")
    plt.show()

    # calculate the midpoints of the bins
    bins_midpoints = (bins[0:-1] + bins[1:])/2
    # use the midpoints to calculate the line of best fit of the log(f) graphs
    slope, intercept = np.polyfit(bins_midpoints, log_counts, 1)
    # estimate the temperature
    estimate_of_T = -1/slope
    print(f"For the histogram with a = 1, the estimate of T using the gradient is {estimate_of_T}")
    mean_E_d = np.sum(E_d) / (N * N_sweeps)
    print(f"For the histogram with a = 1, the estimate of T using the mean is {mean_E_d}")

    # plot histogram of single particle energy
    plt.figure(f"f(single_particle_energy) with N_sweeps= {N_sweeps} and epsilon = {epsilon}")
    plt.hist(single_particle_energy, color = 'darkseagreen')
    plt.xlabel("single-particle energy")
    plt.ylabel("frequency")
    plt.savefig(f"Q6_f(single_particle_energy)_{a}_{N_sweeps}_{epsilon}.eps", bbox_inches ="tight")
    plt.show()

    # plot log(f(single-particle-energy))
    counts, bins = np.histogram(single_particle_energy)
    log_counts = np.log(counts)
    plt.figure(f"log(f(single_particle_energy)) with N_sweeps= {N_sweeps} and epsilon = {epsilon}")
    plt.bar(bins[0:-1], log_counts, width=bins[1:] - bins[0:-1], color='darkseagreen', align="edge")
    plt.xlabel("single-particle energy")
    plt.ylabel("log(f(single-particle energy))")
    try:
        plt.ylim([np.floor(np.amin(log_counts)), np.ceil(np.amax(log_counts))])
    except:
        plt.ylim([0, np.ceil(np.amax(log_counts))])
    plt.savefig(f"Q6_log(f(single_particle_energy))_{a}_{N_sweeps}_{epsilon}.eps", bbox_inches ="tight")
    plt.show()

    # calculate the midpoints of the bins
    bins_midpoints = (bins[0:-1] + bins[1:])/2
    # use the midpoints to calculate the line of best fit of the log(f) graphs
    slope, intercept = np.polyfit(bins_midpoints, log_counts, 1)
    print(f"the gradient of the log(f(single-particle-energy)) graph = {slope}")


# for varying a
a_vector = np.linspace(0.01,10,101)
N_sweeps = 1000
total_energy = np.zeros(len(a_vector))
T_approx_slope = np.zeros(len(a_vector))
T_approx_mean = np.zeros(len(a_vector))
i = 0
for a in a_vector:
    epsilon = a

    p, E_d, single_particle_energy, total_energy[i] = Monte_Carlo(dimensions, E, N, N_sweeps, epsilon, a, print_total_energy = True)

    # plot log(f(E_d))
    counts, bins = np.histogram(E_d)
    log_counts = np.log(counts)
    # calculate the midpoints of the bins
    bins_midpoints = (bins[0:-1] + bins[1:]) / 2
    # use the midpoints to calculate the line of best fit of the log(f) graphs
    slope, intercept = np.polyfit(bins_midpoints, log_counts, 1)
    T_approx_slope[i] = -1 / (slope * k_B)
    # average E_d
    T_approx_mean[i] = np.sum(E_d) / (N * N_sweeps)
    i += 1

# plot temperature as a function of the total energy (using the gradient estimate for temperature)
plt.figure("Temperature estimated using gradient")
plt.plot(total_energy, T_approx_slope,'x')
plt.grid()
plt.xlabel('E')
plt.ylabel('T')
plt.show()
# find the gradient of the graph
slope, intercept = np.polyfit(total_energy, T_approx_slope, 1)
print(f"T (estimated using gradient) against E has gradient = {slope}")

# plot temperature as a function of the total energy (using the mean estimate for temperature)
plt.figure("Temperature estimated using mean")
plt.plot(total_energy, T_approx_mean,'x')
plt.grid()
plt.xlabel('E')
plt.ylabel('T')
plt.show()
# find the gradient of the graph
slope, intercept = np.polyfit(total_energy, T_approx_mean, 1)
print(f"T (estimated using mean) against E has gradient = {slope}")
