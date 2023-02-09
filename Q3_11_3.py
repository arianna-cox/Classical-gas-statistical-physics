import numpy as np
from matplotlib import pyplot as plt
from Monte_Carlo import Monte_Carlo
from tabulate import tabulate
plt.rcParams.update({'font.size': 15})

def E(p):
    return np.dot(p, p) / 2

dimensions = 3
epsilon_vector = [0.01,0.1,1,10]
N = 100
table_E_d = np.zeros((3*len(epsilon_vector),3))
table_single_particle_energy = np.zeros((3*len(epsilon_vector),3))
i = 0
for epsilon in epsilon_vector:
    for N_sweeps in [10,100,1000]:
        p, E_d, single_particle_energy = Monte_Carlo(dimensions, E, N, N_sweeps, epsilon)

        # plot log(f(E_d))
        counts, bins = np.histogram(E_d)
        log_counts = np.log(counts)
        plt.figure(f"log(f(E_d)) with N_sweeps= {N_sweeps} and epsilon = {epsilon}")
        plt.bar(bins[0:-1], log_counts, width=bins[1:] - bins[0:-1], color = 'tab:blue', align="edge")
        plt.xlabel(r"$E_d$")
        plt.ylabel(r"log$(\, f(E_d))$")
        try:
            plt.ylim([np.floor(np.amin(log_counts)), np.ceil(np.amax(log_counts))])
        except:
            plt.ylim([0, np.ceil(np.amax(log_counts))])
        plt.savefig(f"Q3_log(f(E_d))_{N_sweeps}_{epsilon}.eps", bbox_inches ="tight")
        plt.show()

        # calculate the midpoints of the bins
        bins_midpoints = (bins[0:-1] + bins[1:])/2
        # use the midpoints to calculate the line of best fit of the log(f) graphs
        slope, intercept = np.polyfit(bins_midpoints, log_counts, 1)
        table_E_d[i,:] = [epsilon, N_sweeps, slope]

        # plot histogram of single particle energy
        plt.figure(f"f(single_particle_energy) with N_sweeps= {N_sweeps} and epsilon = {epsilon}")
        plt.hist(single_particle_energy)
        plt.xlabel("single-particle energy")
        plt.ylabel("frequency")
        plt.savefig(f"Q3_f(single_particle_energy)_{N_sweeps}_{epsilon}.eps", bbox_inches ="tight")
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
        plt.savefig(f"Q3_log(f(single_particle_energy))_{N_sweeps}_{epsilon}.eps", bbox_inches ="tight")
        plt.show()

        # calculate the midpoints of the bins
        bins_midpoints = (bins[0:-1] + bins[1:])/2
        # use the midpoints to calculate the line of best fit of the log(f) graphs
        slope, intercept = np.polyfit(bins_midpoints, log_counts, 1)
        table_single_particle_energy[i,:] = [epsilon, N_sweeps, slope]
        i += 1

# table of the gradient of the log(f) graphs for E_d
head = ["epsilon", "N_sweeps", "slope"]
print("table of the gradient of the log(f) graphs for E_d")
print(tabulate(table_E_d, headers = head))

# table of the gradient of the log(f) graphs for single_particle_energy
print("table of the gradient of the log(f) graphs for single-particle energy")
print(tabulate(table_single_particle_energy, headers = head))
