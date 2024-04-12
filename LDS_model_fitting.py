import autograd.numpy as np
import autograd.numpy.random as npr
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec
import seaborn as sns
import ssm
from pynwb import NWBHDF5IO

from data_loaders import *

file_name = "sub-monk-g_ses-session0.nwb"

with NWBHDF5IO(file_name, "r") as io:
    read_nwbfile = io.read()
    original_input, original_spike,_ ,_ = getData_for_LDS(file_name, 
                                                        num_permutation = 0)



obs=original_spike
inputs = original_input
print(obs.shape)

# Known paramters
time_bins = obs.shape[0]
obs_dim = obs.shape[1]
input_dim = 4

# Assumption
state_dim = 4

# fit the model
# Poisson emission
lds_inp = ssm.LDS(obs_dim, state_dim, M=input_dim, emissions="poisson")
elbos, q = lds_inp.fit((obs).astype(int), inputs=inputs, num_iters=20)

# Plot the ELBOs
plt.plot(elbos/time_bins, label="Laplace-EM")
plt.xlabel("Iteration")
plt.ylabel("ELBO")
plt.legend()
plt.show()

# Guassion emission
lds_inp = ssm.LDS(obs_dim, state_dim, M=input_dim, emissions="guassian")
elbos, q = lds_inp.fit((obs).astype(int), inputs=inputs, num_iters=20)

# Plot the ELBOs
plt.plot(elbos/time_bins, label="Laplace-EM")
plt.xlabel("Iteration")
plt.ylabel("ELBO")
plt.legend()
plt.show()