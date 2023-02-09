import numpy as np
from matplotlib import pyplot as plt
from Monte_Carlo import Monte_Carlo
from tabulate import tabulate
plt.rcParams.update({'font.size': 13})
k_B = 1

def E(p):
    return np.sqrt(1+np.dot(p,p)) -1

dimensions = 3
N = 100

# Consider different values of the total energy by varying a in the range 0.1 to 2.0
a_vector = np.linspace(0.1,2,51)
N_sweeps = 1000
total_energy = np.zeros(len(a_vector))
T_approx_slope = np.zeros(len(a_vector))
T_approx_mean = np.zeros(len(a_vector))
i = 0
for a in a_vector:
    epsilon = a

    p, E_d, single_particle_energy, total_energy[i] = Monte_Carlo(dimensions, E, N, N_sweeps, epsilon, a, print_total_energy = True)

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
    print(i)

np.save('Q7_a_vector', a_vector)
np.save('Q7_total_energy', total_energy)
np.save('Q7_T_approx_slope', T_approx_slope)
np.save('Q7_T_approx_mean', T_approx_mean)

a_vector = np.load('Q7_a_vector.npy')
total_energy = np.load('Q7_total_energy.npy')
T_approx_slope = np.load('Q7_T_approx_slope.npy')
T_approx_mean = np.load('Q7_T_approx_mean.npy')

plt.plot(total_energy, T_approx_slope, 'x')
plt.grid()
plt.xlabel('E')
plt.ylabel('T')
plt.show()

plt.plot(total_energy, T_approx_mean, 'x')
plt.grid()
plt.xlabel('E')
plt.ylabel('T')
plt.show()

gradient, intercept = np.polyfit(total_energy[0:10], T_approx_mean[0:10], 1)
print(gradient)
gradient, intercept = np.polyfit(total_energy[-5:-1], T_approx_mean[-5:-1], 1)
print(gradient)

# print histograms of single particle energies
N_sweeps = 10000
for a in [0.001,100]:
    epsilon = a
    p, E_d, single_particle_energy, totalE = Monte_Carlo(dimensions, E, N, N_sweeps, epsilon, a, print_total_energy = True)

    # plot histogram of single particle energy
    plt.figure(f"f(single_particle_energy) with N_sweeps= {N_sweeps} and epsilon = {epsilon}")
    plt.hist(single_particle_energy)
    plt.xlabel("single-particle energy")
    plt.ylabel("frequency")
    plt.show()

    # plot log(f(single-particle-energy)
    counts, bins = np.histogram(single_particle_energy)
    log_counts = np.log(counts)
    plt.figure(f"log(f(single_particle_energy)) with N_sweeps= {N_sweeps} and epsilon = {epsilon}")
    plt.bar(bins[0:-1], log_counts, width=bins[1:] - bins[0:-1], color='tab:blue', align="edge")
    plt.xlabel("single-particle energy")
    plt.ylabel("log(f(single-particle energy))")
    try:
        plt.ylim([np.floor(np.amin(log_counts)), np.ceil(np.amax(log_counts))])
    except:
        plt.ylim([0, np.ceil(np.amax(log_counts))])
    plt.show()

    # calculate the midpoints of the bins
    bins_midpoints = (bins[0:-1] + bins[1:]) / 2
    # use the midpoints to calculate the line of best fit of the log(f) graphs
    slope, intercept = np.polyfit(bins_midpoints, log_counts, 1)
    print(f"E = {totalE}")
    print(f"gradient = {slope}")
    print(f"-1/gradient = {-1/slope}")
    print(f"E/2N = {totalE/(2*N)}")
    print(f"2E/3N = {2*totalE/(3*N)}")