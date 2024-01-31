
import numpy as np
import math
import matplotlib.pyplot as plt

import data_loaders as dl


import seaborn as sns
sns.set_style("white")
sns.set_context("talk")

color_names = ["windows blue",
               "red",
               "amber",
               "faded green",
               "dusty purple",
               "orange",
               "clay",
               "pink",
               "greyish",
               "mint",
               "light cyan",
               "steel blue",
               "forest green",
               "pastel purple",
               "salmon",
               "dark brown"]

colors = sns.xkcd_palette(color_names)

import ssm



def fit_LDS( file_name, acqRate=60, newRate=10, state_dim=8, emission='poisson', save_suffix = '', use_inp=True, use_fb=True, use_trials=None, useITI=False ):


    if useITI:
        obs, inputs, pos = dl.getData_for_LDS(file_name,acqRate=acqRate, newRate=newRate, use_trials=use_trials )
    else:
        obs, inputs, pos = dl.getData_for_LDS_noITI(file_name,acqRate=acqRate, newRate=newRate, use_trials=use_trials )

    obs = obs[20:,:]
    inputs = inputs[20:,:]
    pos = pos[20:,:]
    if not use_fb:
        inputs=inputs[:,:2] # use only feedforward input

    time_bins = obs.shape[0]
    obs_dim = obs.shape[1]
    input_dim = inputs.shape[1]

    if use_inp:
        lds_inp = ssm.LDS(obs_dim, state_dim, M=input_dim, emissions=emission)
        elbos, q = lds_inp.fit((obs).astype(int), inputs=inputs, num_iters=20)
    else:
        lds_inp = ssm.LDS(obs_dim, state_dim, M=0, emissions=emission)
        elbos, q = lds_inp.fit((obs).astype(int), inputs=None, num_iters=20)
    

    # Plot the ELBOs
    plt.figure()
    plt.plot(elbos/time_bins, label="Laplace-EM")
    plt.xlabel("Iteration")
    plt.ylabel("ELBO")
    plt.legend()
    plt.savefig('ELBO_training_'+save_suffix+'.png')
    plt.close()


    state_means = q.mean_continuous_states[0]
    if use_inp:
        smoothed_obs = lds_inp.smooth(state_means, obs, input=inputs)
    else:
        smoothed_obs = lds_inp.smooth(state_means, obs )

    A = lds_inp.dynamics.A
    eigenvalues, eigenvectors = np.linalg.eig(A-np.eye(A.shape[0]))
    plt.figure()
    plt.scatter( np.real(eigenvalues), np.imag(eigenvalues))
    plt.plot([0,0],[-.1,.1],'k--')
    plt.savefig('eigenvalues_'+save_suffix+'.png')
    plt.close()

    dic = {'lds':lds_inp, 'states':state_means, 'x':smoothed_obs,
            'obs':obs, 'pos':pos, 'inp':inputs, 'file_name':file_name }
    
    
    return dic




def fit_LDS_shuffle( file_name, acqRate=60, newRate=10, state_dim=8, emission='poisson', save_suffix = '', use_inp=True, use_fb=True, use_trials=None, useITI=False ):


    
    obs, inputs, pos = dl.getData_for_LDS_shuffle(file_name,acqRate=acqRate, newRate=newRate, use_trials=use_trials )

    obs = obs[20:,:]
    inputs = inputs[20:,:]
    pos = pos[20:,:]
    if not use_fb:
        inputs=inputs[:,:2] # use only feedforward input

    time_bins = obs.shape[0]
    obs_dim = obs.shape[1]
    input_dim = inputs.shape[1]

    if use_inp:
        lds_inp = ssm.LDS(obs_dim, state_dim, M=input_dim, emissions=emission)
        elbos, q = lds_inp.fit((obs).astype(int), inputs=inputs, num_iters=20)
    else:
        lds_inp = ssm.LDS(obs_dim, state_dim, M=0, emissions=emission)
        elbos, q = lds_inp.fit((obs).astype(int), inputs=None, num_iters=20)
    

    # Plot the ELBOs
    plt.figure()
    plt.plot(elbos/time_bins, label="Laplace-EM")
    plt.xlabel("Iteration")
    plt.ylabel("ELBO")
    plt.legend()
    plt.savefig('ELBO_training_'+save_suffix+'.png')
    plt.close()


    state_means = q.mean_continuous_states[0]
    if use_inp:
        smoothed_obs = lds_inp.smooth(state_means, obs, input=inputs)
    else:
        smoothed_obs = lds_inp.smooth(state_means, obs )

    A = lds_inp.dynamics.A
    eigenvalues, eigenvectors = np.linalg.eig(A-np.eye(A.shape[0]))
    plt.figure()
    plt.scatter( np.real(eigenvalues), np.imag(eigenvalues))
    plt.plot([0,0],[-.1,.1],'k--')
    plt.close()

    dic = {'lds':lds_inp, 'states':state_means, 'x':smoothed_obs,
            'obs':obs, 'pos':pos, 'inp':inputs, 'file_name':file_name }
    
    
    return dic