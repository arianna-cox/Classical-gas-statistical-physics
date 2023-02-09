import numpy as np
import matplotlib as plt
import tabulate as tabulate

# if a is not provided, each particle is initialise with momentum e_1
# if a is provided, each particle is initialised with a randomly assigned momentum with each component in (-a,a)
def Monte_Carlo(dimensions, E, N, N_sweeps, epsilon, a = None, print_total_energy = None):
    # initial configuration
    if a is None:
        p = np.zeros((dimensions, N))
        p[0, :] = np.ones(N)
    else:
        p = (np.random.rand(dimensions,N)-0.5)*2*a
    if print_total_energy == True:
        initial_energy = 0
        for i in range(N):
            initial_energy += E(p[:,i])
    E_d = 0
    N_updates = N * N_sweeps
    E_d_vector = np.zeros(N_updates)
    single_particle_energy = np.zeros(N_updates)
    # run through steps 2 to 4 N_updates times
    for i in range(N_updates):
        # choose one of the N particles at random
        particle_number = np.random.randint(0, N)
        E_curr = E(p[:, particle_number])
        # choose a random change delta_p and calculate the new momentum and energy
        delta_p = (np.random.rand(dimensions) - 0.5) * 2 * epsilon
        E_prop = E(p[:, particle_number] + delta_p)
        delta_E = E_prop - E_curr
        if delta_E <= E_d:
            # accept the change
            p[:, particle_number] += delta_p
            E_d -= delta_E
            single_particle_energy[i] = E_prop
        else:
            # reject the changes
            single_particle_energy[i] = E_curr
        E_d_vector[i] = E_d
    if print_total_energy == True:
        return p, E_d_vector, single_particle_energy, initial_energy
    else:
        return p, E_d_vector, single_particle_energy