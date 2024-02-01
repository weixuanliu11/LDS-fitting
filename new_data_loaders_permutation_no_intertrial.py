import pynwb

import numpy as np

from pynwb import NWBHDF5IO, NWBFile, TimeSeries

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import bisect


from data_loaders import Bin_spike_cursor
from data_loaders import Cluster
from data_loaders import LastCursorPosition

# Get the trial id of the last direct-reaching trial
def getLastTrialId(read_nwbfile, acqRate=60, newRate=60):
    # Get obs_details data 
    obs_details = read_nwbfile.processing["behavior"]["obs_details"].data[:]
    
    # The index of the first non-na entry
    non_na_index = next((i for i, val in enumerate(obs_details) if val != 'na'), None)
    
    # Get the greatest stop index that is less thna than non_na_index
    stop_times = read_nwbfile.trials["stop_time"].data[:]
    total_spike_counts,_ = Bin_spike_cursor(read_nwbfile, acqRate = acqRate, newRate=newRate)
    stop_Indices = [round(stop_time * newRate - 1) for stop_time in stop_times]
    
    # Get the last trial id
    lastTrialId = bisect.bisect_right(stop_Indices, non_na_index) - 1
    return lastTrialId

# Get the cursor position matrix
def CursorPositionMatrix_list(read_nwbfile, last_trial_id, acqRate=60, newRate = 60):
    _, total_cursor_positions = Bin_spike_cursor(read_nwbfile, acqRate = acqRate, newRate=newRate)
    cursor_position_list=[]
    for trial_id in range(last_trial_id + 1):

        #Get the cooresponding spike counts matrix
        start_time = read_nwbfile.trials["start_time"].data[trial_id]
        stop_time = read_nwbfile.trials["stop_time"].data[trial_id]

        start_Index = round(start_time*newRate - 1)
        stop_Index = round(stop_time*newRate - 1)

        # add in the list
        cursor_position_list.append(total_cursor_positions[start_Index: stop_Index + 1, :])
    
    return cursor_position_list

# Get the target matrix
def targetMatrix_list(read_nwbfile, lastTrialId, cursor_positions=None, acqRate=60, newRate=60, num_clusters = 8  ):
    ''' Target input matrix (feedforward inputs )'''

    trial_num = len(read_nwbfile.trials["id"].data)
    timespan = read_nwbfile.processing["behavior"]["cursor"].data.shape[0]
    tfactor = round( acqRate/newRate )
    timespan = timespan//tfactor

    # read endpoints from file if not given
    if cursor_positions is None:
        cursor_positions=[]
        for trial_id in range(trial_num):
            cursor_positions.append(LastCursorPosition(trial_id, read_nwbfile, acqRate=acqRate))

    cluster_assignments, center_xs, center_ys = Cluster(cursor_positions, num_clusters=num_clusters, vis = False)
    
    target_matrix_list = []
    for trial_id in range(lastTrialId + 1):

        #Get the start time and stop time
        start_time = read_nwbfile.trials["start_time"].data[trial_id]
        stop_time = read_nwbfile.trials["stop_time"].data[trial_id]
        
        #Caculate the index of start time and stop time
        start_Index = round(start_time*newRate - 1)
        stop_Index = round(stop_time*newRate - 1)

        # In which cluster
        cluster_id = cluster_assignments[trial_id]
        x, y = center_xs[cluster_id], center_ys[cluster_id] 
        
        # get the matrix for the trial and add it to the list
        result_matrix = np.array([[x, y]] * (stop_Index + 1 - start_Index))
        target_matrix_list.append(result_matrix)
    
    return target_matrix_list

# Get the input matrix
def InputMatrices(read_nwbfile, permutation = 0, acqRate=60, newRate=60 ):
    ''' 4D inputs with 2D target input and 2D feedback error input '''
    
    # get the trial id of the last direct-reaching trial
    lastTrialId = getLastTrialId(read_nwbfile, acqRate=60, newRate=60)

    # Get the number of matrices in the list
    num_matrices = lastTrialId + 1

    # Get the lists of target and cursor positions
    target_matrix_list = targetMatrix_list(read_nwbfile, lastTrialId, None, acqRate=acqRate, newRate=newRate )
    cursor_positions_list = CursorPositionMatrix_list(read_nwbfile, lastTrialId, acqRate=acqRate, newRate=newRate)

    # The list for all original and permuted input matrices
    input_matrices = []
    # The list to store the permutation for all permuted input matrices
    matrix_permutations = []

    # Get the original input matrix
    target_matrix = np.concatenate(target_matrix_list, axis=0)
    cursor_positions = np.concatenate(cursor_positions_list, axis = 0)
    feedback_matrix = target_matrix-cursor_positions
    input_matrix = np.concatenate((target_matrix, feedback_matrix), axis=1)

    # Add the input matrix to the list and updata the permutation list
    input_matrices.append(input_matrix)
    matrix_permutations.append(None)
    # Do the permutations
    for i in range(permutation):
        # Create a random permutation for the matrices
        matrix_permutation = np.random.permutation(num_matrices)

        # Shuffle the matrices inside the list
        shuffled_target_list = [target_matrix_list[i] for i in matrix_permutation]
        shuffled_cursor_list = [cursor_positions_list[i] for i in matrix_permutation]

        # Get the cooresponding input matrix
        target_matrix = np.concatenate(shuffled_target_list, axis=0)
        cursor_positions = np.concatenate(shuffled_cursor_list, axis = 0)
        feedback_matrix = target_matrix-cursor_positions
        input_matrix = np.concatenate((target_matrix, feedback_matrix), axis=1)

        # add the input matrix to the list and update the permutation list
        input_matrices.append(input_matrix)
        matrix_permutations.append(matrix_permutation)
    return input_matrices, matrix_permutations