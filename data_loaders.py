import pynwb

from datetime import datetime
from uuid import uuid4

import numpy as np
from dateutil import tz

from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.behavior import Position, SpatialSeries
from pynwb.epoch import TimeIntervals
from pynwb.file import Subject
import autograd.numpy as np
import autograd.numpy.random as npr

import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import bisect


def LastCursorPosition(trial_id, read_nwbfile, acqRate=60 ):
    ''' 
    Return a list of cursor positions at the end of each trial from an open NWB file. 
    :param trial_id: trial id
    :param read_nwbfile: the contents of NWB file.
    :param acqRate: the acquired rate of data
    :return: cursor_position
    '''
    #Get the start time and stop time
    stop_time = read_nwbfile.trials["stop_time"].data[trial_id]
    stop_Index = round(stop_time*acqRate - 1)
    
    #The array of cursor position at the end of the given trial
    cursor_position = read_nwbfile.processing["behavior"]["cursor"].data[stop_Index,:]-np.mean(read_nwbfile.processing["behavior"]["cursor"].data[:], axis=0)
    return cursor_position


#----------------------------------------------#



def Cluster( cursor_positions, num_clusters = 8 , acqRate = 60, vis=True, read_nwbfile=None ):
    ''' 
    Cluster a list of x-y cursor positions using kMeans 
    :param cursor_postions: an array of cursor positions at the end of all trials
    :param num_clusters: the number of clusters of the given cursor_positions
    :param acqRate: the acquired rate of data
    :param vis: whether we visualize the clusters and print the angular positions of 
                every cursor position or not
    :return: cluster_assignments, center_xs, center_ys
    '''

    ## read file if no cursor positions given
    
    if cursor_positions is None:
        trial_num = len(read_nwbfile.trials["id"].data)
        cursor_positions=[]
        for trial_id in range(trial_num):
            cursor_positions.append(LastCursorPosition(trial_id, read_nwbfile, 
                                                       acqRate=acqRate
                                                       ))

    cursor_positions = np.array(cursor_positions)
    # Use KMeans to cluster
    kmeans = KMeans(n_clusters=num_clusters)
    cluster_assignments = kmeans.fit_predict(cursor_positions)
    
    if vis:
        # A figure visualizing the clusters
        plt.figure()
    for cluster_id in range(num_clusters):
        cluster_points = cursor_positions[cluster_assignments == cluster_id]
        if vis:
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], marker='.', label=f'Cluster {cluster_id}')

        # Plot cluster centers
        center_xs = kmeans.cluster_centers_[:, 0]
        center_ys = kmeans.cluster_centers_[:, 1]
        
        if vis:
            plt.scatter(center_xs, center_ys, marker='x', s=200, color='red', label='Cluster Centers')
            plt.title('KMeans Clustering')
            plt.legend()

    if vis:
        # Calculate the degree
        angle_degs=[]
        for i in range(len(center_xs)):
            center_x = center_xs[i]
            center_y = center_ys[i]
            angle_rad = math.atan2(center_y, center_x)
            angle_deg = (math.degrees(angle_rad)+ 360) % 360
            angle_degs.append(angle_deg)
    
        print("Angle Positions of each center:\n",angle_degs)


    return cluster_assignments, center_xs, center_ys

    
#----------------------------------------------#


def Bin_spike_cursor(read_nwbfile, acqRate=60, newRate=60):
    ''' 
    (Approximately) Downsample data from acqRate to newRate. 
    Spike counts are summed up and cursor positions are averaged. 
    :param read_nwbfile: the contents of NWB file.
    :param acqRate: the acquired rate of data
    :param newRate: the new rate of data
    :return: new_spike_counts, new_cursor_positions
    '''

    num_timepoints = read_nwbfile.processing["behavior"]["spike_counts"].data.shape[0]
    num_neurons = read_nwbfile.processing["behavior"]["spike_counts"].data.shape[1]
    tfactor = round(acqRate/newRate)
    

    #Get spikes counts & cursor positions
    total_spike_counts = read_nwbfile.processing["behavior"]["spike_counts"].data[:]
    total_cursor_positions = read_nwbfile.processing["behavior"]["cursor"].data[:]
    # center cursor targets around zero (to get angles appropriately)
    total_cursor_positions = total_cursor_positions - np.mean(total_cursor_positions, axis=0)

    # downsample if needed
    if tfactor>1:
        new_num_timepoints = num_timepoints//tfactor
        #fill up the 100ms spike counts matrix and cursor position matrix
        new_spike_counts = np.zeros((new_num_timepoints, num_neurons))
        new_cursor_positions = np.zeros((new_num_timepoints, 2))
        
        for time_id in range(new_num_timepoints):
            subset = total_spike_counts[time_id*tfactor:(time_id+1)*tfactor,:]      # sum spike counts
            new_spike_counts[time_id] = np.sum(subset,axis=0)
            
            subset = total_cursor_positions[time_id*tfactor:(time_id+1)*tfactor,:]  # mean cursor position
            new_cursor_positions[time_id] = np.mean(subset,axis=0)
    else:
        new_spike_counts=total_spike_counts
        new_cursor_positions=total_cursor_positions
    return new_spike_counts, new_cursor_positions

    
#----------------------------------------------#


def getLastTrialId(read_nwbfile, acqRate=60, newRate=60):
    '''
    Get the id of the last trial performing direct-reaching
    :param read_nwbfile: the contents of NWB file.
    :param acqRate: the acquired rate of data
    :param newRate: the new rate of data
    :return: lastTrialId
    '''

    # Get obs_details data 
    obs_position = read_nwbfile.processing["behavior"]["obstacle_position"].data[:]
    
    # The index of the first non-na entry
    nan_indices = np.where(np.isnan(obs_position).any(axis=1))
    non_na_index = nan_indices[0][-1] + 1
    
    
    # Get the greatest stop index that is less thna than non_na_index
    stop_times = read_nwbfile.trials["stop_time"].data[:]
    total_spike_counts,_ = Bin_spike_cursor(read_nwbfile, acqRate = acqRate, newRate=newRate)
    stop_Indices = [round(stop_time * newRate - 1) for stop_time in stop_times]
    
    # Get the last trial id
    lastTrialId = bisect.bisect_right(stop_Indices, non_na_index) - 1
    return lastTrialId



#----------------------------------------------#




def CursorPositionMatrix_list(read_nwbfile, last_trial_id, acqRate=60, newRate = 60):
    '''
    Cursor Position Matrix list
    list[i] is a matrix of cursor positions in trial i
    :param read_nwbfile: the contents of NWB file.
    :param last_trial_id: the trial id of the last direct-reaching trial
    :param acqRate: the acquired rate of data
    :param newRate: the new rate of data
    :return: cursor_position_list
    '''
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



#----------------------------------------------#



def targetMatrix_list(read_nwbfile, lastTrialId, cursor_positions=None, acqRate=60, newRate=60, num_clusters = 8  ):
    ''' 
    Target input matrix list (feedforward inputs )
    :param read_nwbfile: the contents of NWB file.
    :param lastTrialId: the trial id of the last direct-reaching trial
    :param cursor_positions: cursor positions
    :param acqRate: the acquired rate of data
    :param newRate: the new rate of data
    :param num_clusters: the number of clusters of cursor_positions 
        at the end of all direct-reaching trials
    :return: target_matrix_list
    '''

    trial_num = len(read_nwbfile.trials["id"].data)
    timespan = read_nwbfile.processing["behavior"]["cursor"].data.shape[0]
    tfactor = round( acqRate/newRate )
    timespan = timespan//tfactor

    # read endpoints from file if not given
    if cursor_positions is None:
        cursor_positions=[]
        for trial_id in range(trial_num):
            cursor_positions.append(LastCursorPosition(trial_id, read_nwbfile, acqRate=acqRate))

    cluster_assignments, center_xs, center_ys = Cluster(cursor_positions, num_clusters=num_clusters, acqRate = acqRate, vis = False)
    
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



#----------------------------------------------#



def InputMatrix_list(read_nwbfile, acqRate=60, newRate=60, num_clusters = 8):
    ''' 
    4D inputs with 2D target input and 2D feedback error input 
    :param read_nwbfile: the contents of NWB file
    :param acqRate: the acquired rate of data
    :param newRate: the new rate of data
    :param num_clusters: the number of clusters of cursor_positions 
        at the end of all direct-reaching trials
    :return: input_matrix_list
    '''
    
    # get the trial id of the last direct-reaching trial
    lastTrialId = getLastTrialId(read_nwbfile, acqRate=acqRate, newRate=newRate)
        
    # Get the number of matrices in the list
    num_matrices = lastTrialId + 1

    # Get the lists of target and cursor positions
    target_matrix_list = targetMatrix_list(read_nwbfile, lastTrialId, None, acqRate=acqRate, newRate=newRate, num_clusters = num_clusters)
    cursor_positions_list = CursorPositionMatrix_list(read_nwbfile, lastTrialId, acqRate=acqRate, newRate=newRate)

    #Get the input matrix list
    input_matrix_list=[]
    for idx in range(len(target_matrix_list)):
        target_matrix = target_matrix_list[idx]
        cursor_matrix = cursor_positions_list[idx]

        # feedback input u at time t+1 = target - cursor[t]
        cursor_matrix = np.insert(cursor_matrix, 0, cursor_matrix[0], axis=0)
        cursor_matrix = np.delete(cursor_matrix, -1, axis=0)

        feedback_matrix = target_matrix-cursor_matrix
        input_matrix = np.concatenate((target_matrix, feedback_matrix), axis=1)
        input_matrix_list.append(input_matrix)
    return input_matrix_list



#----------------------------------------------#



def NeuralActivity_list(read_nwbfile, last_trial_id, acqRate=60, newRate = 60):
    '''
    Get the spike matrix list
    :param read_nwbfile: the contents of NWB file.
    :param lastTrialId: the trial id of the last direct-reaching trial
    :param acqRate: the acquired rate of data
    :param newRate: the new rate of data
    :return: spike_counts_list
    '''
    total_spike_counts, _ = Bin_spike_cursor(read_nwbfile, acqRate = acqRate, newRate=newRate)
    spike_counts_list=[]
    for trial_id in range(last_trial_id + 1):

        #Get the cooresponding spike counts matrix
        start_time = read_nwbfile.trials["start_time"].data[trial_id]
        stop_time = read_nwbfile.trials["stop_time"].data[trial_id]

        start_Index = round(start_time*newRate - 1)
        stop_Index = round(stop_time*newRate - 1)

        # add in the list
        spike_counts_list.append(total_spike_counts[start_Index: stop_Index + 1, :])
    
    return spike_counts_list        



#----------------------------------------------#



# orignal input, spikes, permutated inputs and spikes
def getData_for_LDS(filename, num_permutation, acqRate=60, newRate=60, num_clusters = 8):
    ''' 
    Read file and get data for fitting LDS including spike counts as observations and inputs, as well as cursor positions.
    Data is acquired at acqRate (Hz) and can be downsampled to newRate (Hz) 
    Do permutation on original inputs and spikes counts with respect to trial order.
    :param filename: the name of the NWBfile
    :param num_permutation: the number of permutations we want to perform 
    :param acqRate: the acquired rate of data
    :param newRate: the new rate of data
    :param num_clusters: the number of clusters of cursor_positions 
        at the end of all direct-reaching trials
    :return: original_input, original_spike, p_input_matrices, p_spike_matrices
    '''
    with NWBHDF5IO( filename , "r") as io:
        read_nwbfile = io.read()    

        trial_num = len(read_nwbfile.trials["id"].data)
        
        # get the trial id of the last direct-reaching trial
        lastTrialId = getLastTrialId(read_nwbfile, acqRate=acqRate, newRate=newRate)
        
        # Get the orginal input_list and spike list
        input_matrix_list = InputMatrix_list(read_nwbfile, acqRate=acqRate, newRate=newRate, num_clusters = num_clusters)
        spike_list = NeuralActivity_list(read_nwbfile, lastTrialId, acqRate=acqRate, newRate=newRate)
        print(input_matrix_list)
        # Get the original input matrix and spike matrix
        original_input = np.concatenate(input_matrix_list, axis=0)
        original_spike = np.concatenate(spike_list, axis=0)
        # Do the permutations
        p_input_matrices=[]
        p_spike_matrices=[]
        
        npr.seed(0)
        for i in range(num_permutation):
            # Create a random permutation for the matrices
            num_matrices = len(input_matrix_list)
            matrix_permutation = np.random.permutation(num_matrices)

            # Shuffle the matrices inside the list
            new_input_matrix_list = [input_matrix_list[i] for i in matrix_permutation]
            new_spike_list = [spike_list[i] for i in matrix_permutation]

            # Get the cooresponding input matrix
            input_matrix = np.concatenate(new_input_matrix_list, axis=0)
            spike_matrix = np.concatenate(new_spike_list, axis = 0)
            p_input_matrices.append(input_matrix)
            p_spike_matrices.append(spike_matrix)
    return original_input, original_spike, p_input_matrices, p_spike_matrices
