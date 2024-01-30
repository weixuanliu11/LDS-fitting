import autograd.numpy as np
import autograd.numpy.random as npr
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec
import seaborn as sns
import ssm
from ssm.util import random_rotation
from pynwb import NWBHDF5IO

from data_loaders import InputMatrix
from data_loaders import getData_for_LDS

file_name = "sub-monk-g_ses-session0.nwb"

with NWBHDF5IO(file_name, "r") as io:
    read_nwbfile = io.read()
    inputs = InputMatrix(read_nwbfile)

obs=getData_for_LDS(file_name)
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

# Trajectores and Eigenvalues + Eigenvecters Visualization using PCA 

# What I did in details:
# 
# Trajectories
# Get 3 random initial states (x_0)
# Use x_{t+1} = x_{t} + dt * Ax_{t} to get the state of the next time point
# Do 1000 steps so get 1000 state points (starting from x_1)
# The state matrix is 1000x4 (time points * state_dim)
# Use PCA to compose the matrix to be 2 dim --> (1000x2)
# Visual the 1000 points in a 2d plot 
# Also visual the steps using color: as step increases, color become lighter
# 
# Eigenvalues + Eigenvectors
# Get the true eigenval & eigenvec
# scale eigenvec: scaled_eigenval = eigenvec * eigenval to contain evolution speed information
# Apply PCA on scaled_eigenvec and visualize them


from sklearn.decomposition import PCA
import pandas as pd

dim = 4
steps = 50
def dyn(A):
    x=np.random.randint(0, 21, size=(4, 1))
    dx = A.dot(x)
    points = np.zeros((steps, dim))

    for step_i in range(steps):
        points[step_i,:] = x.T
        x = A@x
    print(points)
    pca = PCA(n_components=2)
    result = pca.fit_transform(points)
    
    time_points = np.arange(steps)
    
    plt.scatter(result[:, 0], result[:, 1], c=time_points, cmap='viridis', marker='.')

A = lds_inp.dynamics.A
for i in range(3):
    dyn(A)
cbar = plt.colorbar()

eigenvalues, eigenvectors = np.linalg.eig(A)
v=eigenvectors[:,0]

scaled_vectors = eigenvectors * eigenvalues

matrix = scaled_vectors.T
pca = PCA(n_components=2)
result_eigvec = pca.fit_transform(matrix)
print("eigvectors:\n",result_eigvec)

# Plotting
for i in range(result_eigvec.shape[0]):
    plt.arrow(0, 0, 10*result_eigvec[i,0],10*result_eigvec[i,1], head_width=0.05, head_length=0.05, ec='black')


# Visualization trajectories by visualizing each entry of x seperately

# Choose 5 intial values using np.linspace(0,np.max(obs),5)
# Assume each entry of x have the same intial value, so we have 5 different intial states
# As the previous method, get x throughout certain timespan
# Plot each entry of x in a plot seperately.
import numpy as np
import matplotlib.pyplot as plt

def traj(A, x0):
    plt.figure()
    
    num_steps = 100

    trajectories = []
    current_state = x0.copy()
    for _ in range(num_steps):
#         print(current_state)
        current_state = np.dot(A, current_state)
        trajectories.append(current_state.flatten())
    
    trajectories = np.array(trajectories)

    
    plt.plot(trajectories)
    plt.xlabel('Time Steps')
    plt.ylabel('State Values')
    plt.title(f'inital state: {x0}')
    plt.show()
    
xs = np.linspace(0,np.max(obs),5)
A = lds_inp.dynamics.A
for x in xs:
    if x == 0.5:
        x0 = np.array([x,x,x,x]).reshape(4,1)
        print(x0)
        traj(A, x0)