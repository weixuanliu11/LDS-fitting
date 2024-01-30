from data_loaders import LastCursorPosition
from data_loaders import Bin_spike_cursor
from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.behavior import Position, SpatialSeries
from pynwb.epoch import TimeIntervals
from pynwb.file import Subject
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def population_activity_heatmap(trial_id, read_nwbfile, acqRate=60, newRate=60):
    

    #Get the start time and stop time
    start_time = read_nwbfile.trials["start_time"].data[trial_id]
    stop_time = read_nwbfile.trials["stop_time"].data[trial_id]
    
    total_spike_counts,_ = Bin_spike_cursor(read_nwbfile, acqRate = acqRate, newRate=newRate)
    start_Index = round(start_time*newRate - 1)
    stop_Index = round(stop_time*newRate - 1)
    
    spike_counts = total_spike_counts[start_Index : stop_Index+1, :]
    # plot heatmap
    print(spike_counts)
    plt.imshow(spike_counts,extent=[0,spike_counts.shape[1],stop_time-start_time,0],aspect='auto')
    plt.xlabel("Neurons")
    plt.ylabel("Time")
    plt.title(f'Population Activity of trial {trial_id}')
    plt.colorbar()


#----------------------------------------------#
    

def avg_activity_vs_time(trial_id, read_nwbfile, acqRate=60, newRate=60):
    
    #Get the cooresponding spike counts matrix
    start_time = read_nwbfile.trials["start_time"].data[trial_id]
    stop_time = read_nwbfile.trials["stop_time"].data[trial_id]
    
    total_spike_counts,_ = Bin_spike_cursor(read_nwbfile, acqRate = acqRate, newRate=newRate)
    start_Index = round(start_time*newRate - 1)
    stop_Index = round(stop_time*newRate - 1)
    
    spike_counts = total_spike_counts[start_Index : stop_Index+1, :]
    
    # get population average activity
    avg = np.mean(spike_counts,axis=1)
    
    #plot
    x = np.linspace(0,stop_time-start_time, spike_counts.shape[0])
    plt.plot(x,avg,'-o')
    plt.xlabel("time")
    plt.ylabel("Neural Population Activity")
    plt.title("trial " + str(trial_id))


#----------------------------------------------#


def Activity_vs_CursorPostion(neuron_id, read_nwbfile, acqRate=60, newRate=60, bin100ms = False):
     
    #Get the cooresponding spike counts matrix and cursor position matrix  
    total_spike_counts, total_cursor_positions = Bin_spike_cursor(read_nwbfile, acqRate = acqRate, newRate=newRate)
    
    spike_counts = total_spike_counts[:, neuron_id]
    cursor_positions = total_cursor_positions[:]
    
    
    x_data = cursor_positions[:, 0]
    y_data = cursor_positions[:, 1]
    # Create a scatter plot
    plt.figure()
    plt.plot(x_data, y_data)


    num_bins = 20

    # Create 2D histogram bins
    hist, x_edges, y_edges = np.histogram2d(x_data, y_data, bins=num_bins)

    #average activity in every bin
    avg_activity_bin_matrix = np.zeros((num_bins,num_bins))
    for i in range(num_bins):
        for j in range(num_bins):

            # a bin
            x_left = x_edges[i]
            x_right = x_edges[i+1]
            y_bottom = y_edges[j]
            y_top = y_edges[j+1]

            #For every cursor position that is in this bin,
            #get its timepoint
            row_idx = []
            for idx, x in enumerate(x_data):
                y = y_data[idx]

                #This would cause three points for session2 at the very right of all the positions 
                #not used at all
                if x >= x_left and x<x_right and y>=y_bottom and y<=y_top:
                    row_idx.append(idx)

            #Get the neural spikes corresponded to row index
            summation = 0
            for idx in row_idx:
                summation += spike_counts[idx]
            if len(row_idx) > 0:
                avg = summation/len(row_idx)
            else: 
                avg = 0
                
            #store the activity in the avg_activity_bin_matrix
            avg_activity_bin_matrix[j][i] = avg
                   
    
    plt.imshow(avg_activity_bin_matrix, origin='lower', extent=[x_edges.min(), x_edges.max(), y_edges.min(), 
                                                                y_edges.max()], aspect='auto', cmap='viridis')
    plt.figure()
    plt.imshow(avg_activity_bin_matrix, origin='lower', extent=[x_edges.min(), x_edges.max(), y_edges.min(), 
                                                                y_edges.max()], aspect='auto', cmap='viridis')
    plt.title(f'Neuron {neuron_id} Tuning Curve')
    plt.xlabel('Cursor X Position')
    plt.ylabel('Cursor Y Position')
    plt.colorbar()