import numpy as np
from matplotlib import pyplot as plt
from Monte_Carlo import Monte_Carlo
from tabulate import tabulate
plt.rcParams.update({'font.size': 14})

k_B = 1

def E(p):
    return np.dot(p, p) / 2

dimensions = 3
N = 100

a_vector = [0.1,1,10]
table = np.zeros((3*len(a_vector),5))
i = 0
for a in a_vector:
    for N_sweeps in [10,100,1000]:
        epsilon = a

        p, E_d, single_particle_energy = Monte_Carlo(dimensions, E, N, N_sweeps, epsilon, a)

        # plot log(f(E_d))
        counts, bins = np.histogram(E_d)
        log_counts = np.log(counts)
        plt.figure(f"log(f(E_d) with N_sweeps= {N_sweeps} and epsilon = {epsilon}")
        plt.bar(bins[0:-1], log_counts, width=bins[1:] - bins[0:-1], color = 'coral', align="edge")
        plt.xlabel(r"$E_d$")
        plt.ylabel(r"log$(\, f(E_d))$")
        try:
            plt.ylim([np.floor(np.amin(log_counts)), np.ceil(np.amax(log_counts))])
        except:
            plt.ylim([0, np.ceil(np.amax(log_counts))])
        plt.savefig(f"Q5_log(f(E_d))_{a}_{N_sweeps}_{epsilon}.eps", bbox_inches ="tight")
        plt.show()

        # calculate the midpoints of the bins
        bins_midpoints = (bins[0:-1] + bins[1:]) / 2
        # use the midpoints to calculate the line of best fit of the log(f) graphs
        slope, intercept = np.polyfit(bins_midpoints, log_counts, 1)
        T_approx_slope = -1 / (slope * k_B)
        # average E_d
        mean_E_d = np.sum(E_d) / (N * N_sweeps)
        table[i, :] = [a, epsilon, N_sweeps, T_approx_slope, mean_E_d]
        i += 1

        # plot histogram of single particle energy
        plt.figure(f"f(single_particle_energy) with N_sweeps= {N_sweeps} and epsilon = {epsilon}")
        plt.hist(single_particle_energy, color = 'darkseagreen')
        plt.xlabel("single-particle energy")
        plt.ylabel("frequency")
        plt.savefig(f"Q5_f(single_particle_energy)_{a}_{N_sweeps}_{epsilon}.eps", bbox_inches ="tight")
        plt.show()

        # plot log(f(single-particle-energy)
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
        plt.savefig(f"Q5_log(f(single_particle_energy))_{a}_{N_sweeps}_{epsilon}.eps", bbox_inches="tight")
        plt.show()

head = ["a","epsilon", "N_sweeps", "-1/gradient", "mean of E_d"]
print(tabulate(table, headers = head))

# Find temperature as a function of a
a_vector = np.linspace(0.01,10,51)
N_sweeps = 1000
T_approx_slope = np.zeros(len(a_vector))
T_approx_mean = np.zeros(len(a_vector))
i = 0
for a in a_vector:
    epsilon = a

    p, E_d, single_particle_energy = Monte_Carlo(dimensions, E, N, N_sweeps, epsilon, a)

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

np.save('Q5_a_vector', a_vector)
np.save('Q5_T_approx_slope', T_approx_slope)
np.save('Q5_T_approx_mean', T_approx_mean)

a_vector = np.load('Q5_a_vector.npy')
T_approx_slope = np.load('Q5_T_approx_slope.npy')
T_approx_mean = np.load('Q5_T_approx_mean.npy')

# plot temperature as a function a^2 (using the gradient estimate for temperature)
plt.figure("Temperature estimated using gradient")
plt.plot(a_vector**2, T_approx_slope,'x')
plt.grid()
plt.xlabel(r'$a^2$')
plt.ylabel('T')
plt.show()
# find the gradient of the graph
slope, intercept = np.polyfit(a_vector**2, T_approx_slope, 1)
print(f"T (estimated using gradient) against a^2 has gradient = {slope}")

# plot temperature as a function of a^2 (using the mean estimate for temperature)
plt.figure("Temperature estimated using mean")
plt.plot(a_vector**2, T_approx_mean,'x')
plt.grid()
plt.xlabel(r'$a^2$')
plt.ylabel('T')
plt.show()
# find the gradient of the graph
slope, intercept = np.polyfit(a_vector**2, T_approx_mean, 1)
print(f"T (estimated using mean) against a^2 has gradient = {slope}")
