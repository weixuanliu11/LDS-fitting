import pynwb

from datetime import datetime
from uuid import uuid4

import numpy as np
from dateutil import tz

from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.behavior import Position, SpatialSeries
from pynwb.epoch import TimeIntervals
from pynwb.file import Subject


import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans



def LastCursorPosition(trial_id, read_nwbfile, vis = True,  acqRate=60 ):
    ''' Return a list of cursor positions at the end of each trial from an open NWB file. '''
    #Get the start time and stop time
    stop_time = read_nwbfile.trials["stop_time"].data[trial_id]
    stop_Index = round(stop_time*acqRate - 1)
    
    #The corresponded spike-count matrix
    cursor_position = read_nwbfile.processing["behavior"]["cursor"].data[stop_Index,:]-np.mean(read_nwbfile.processing["behavior"]["cursor"].data[:], axis=0)
    return cursor_position


#----------------------------------------------#


#Cluster the points using kmean and calculate the angle position of the centers of each cluster
def Cluster( cursor_positions, num_clusters = 8 , vis=True, read_nwbfile=None ):
    ''' Cluster a list of x-y cursor positions using kMeans '''

    ## read file if no cursor positions given
    
    if cursor_positions is None:
        trial_num = len(read_nwbfile.trials["id"].data)
        cursor_positions=[]
        for trial_id in range(trial_num):
            cursor_positions.append(LastCursorPosition(trial_id, read_nwbfile, 
                                                       #acqRate=acqRate
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
    ''' (Approximately) Downsample data from acqRate to newRate. 
    Spike counts are summed up and cursor positions are averaged. '''

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


def targetMatrix(read_nwbfile, cursor_positions=None, acqRate=60, newRate=60, num_clusters = 8  ):
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

    # The matrix of target
    target_matrix = np.zeros((timespan, 2))
    for trial_id in range(trial_num):

        #Get the start time and stop time
        start_time = read_nwbfile.trials["start_time"].data[trial_id]
        ##### Choose the appropriate settind
        # 1: fill intertrial time wth zero
        stop_time = read_nwbfile.trials["stop_time"].data[trial_id]
        # OR 2: fill intertrial time with last target?
        if trial_id==trial_num-1:
            stop_time = read_nwbfile.trials["stop_time"].data[trial_id]
        else:
            stop_time = read_nwbfile.trials["start_time"].data[trial_id+1]-1/acqRate
            
        #Caculate the index of start time and stop time
        start_Index = round(start_time*newRate - 1)
        stop_Index = round(stop_time*newRate - 1)

        # In which cluster
        cluster_id = cluster_assignments[trial_id]
        x, y = center_xs[cluster_id], center_ys[cluster_id] 

        # Fill in the matrix
        target_matrix[start_Index:stop_Index + 1, 0] = x
        target_matrix[start_Index:stop_Index + 1, 1] = y
    return target_matrix
    

#----------------------------------------------#


def InputMatrix(read_nwbfile, acqRate=60, newRate=60 ):
    ''' 4D inputs with 2D target input and 2D feedback error input '''
    # Get input matrix from target matrix and cursor position matrix
    target_matrix = targetMatrix(read_nwbfile, None, acqRate=acqRate, newRate=newRate )
    _, new_cursor_positions = Bin_spike_cursor(read_nwbfile,  acqRate=acqRate, newRate=newRate )
    feedback_matrix = target_matrix-new_cursor_positions
    input_matrix = np.concatenate((target_matrix, feedback_matrix), axis=1)
    
    return input_matrix

    
#----------------------------------------------#


def getData_for_LDS( filename, acqRate=60, newRate=60, num_clusters = 8 , vis=False ):
    ''' Read file and get data for fitting LDS including spike counts as observations and inputs, as well as cursor positions.
    Data is acquired at acqRate (Hz) and can be downsampled to newRate (Hz) '''
    with NWBHDF5IO( filename , "r") as io:
        read_nwbfile = io.read()    

        trial_num = len(read_nwbfile.trials["id"].data)

        # read spikes, and cursor position
        new_spike_counts, new_cursor_positions = Bin_spike_cursor(read_nwbfile,  acqRate=acqRate, newRate=newRate )

        # read enpoints
        cursor_positions=[]
        for trial_id in range(trial_num):
            cursor_positions.append(LastCursorPosition(trial_id, read_nwbfile, acqRate=acqRate))

        # get input matrix
        target_matrix = targetMatrix(read_nwbfile,cursor_positions, acqRate=acqRate, newRate=newRate )
        feedback_matrix = target_matrix-new_cursor_positions
        input_matrix = np.concatenate((target_matrix, feedback_matrix), axis=1)

    return new_spike_counts, input_matrix, new_cursor_positions 

    
#----------------------------------------------#

