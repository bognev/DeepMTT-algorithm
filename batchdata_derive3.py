#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 21:26:40 2018

@author: ljx
"""

#==============================================================================
#Randomly select one batch and derive sample data to train the network
#==============================================================================

#from Trajectory_data_generator import*

import scipy.io as scio
import numpy as np
import random as rd
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import JulierSigmaPoints as SP

#using the data with small range of distances (60->20)
from Trajectory_data_generator2 import*
data_len = 64
#define UKF
dim_state = 4
dim_observation = 2
#transition noise
#Sampling time
sT = 0.1
state_n = 10.0
s_var = np.square(state_n)
T2 = np.power(sT,2)
T3 = np.power(sT,3)
T4 = np.power(sT,4)
var_m = np.array([[T4/4,0,T3/2,0],[0,T4/4,0,T3/2],[T3/2,0,T2,0],[0,T3/2,0,T2]]) * s_var
#var_m = np.array([[T4/4,0,0,0],[0,T4/4,0,0],[0,0,T2,0],[0,0,0,T2]]) * s_var
#observation noise
dis_n = 10.0   #distance
dis_var = np.square(dis_n)
azi_n = 8.0     #azimuth
azi_var = np.square(azi_n/1000)

#Modefy observationvations, make them continuous
def data_refine(observation):
    bs,dl,_ = np.shape(observation)
    new_observation = np.copy(observation)
    for j in range(bs):
        for i in range(dl-1):
            a = new_observation[j,i,0]
            b = new_observation[j,i+1,0]
            c = a-b
            if c > 6:
                new_observation[j,i+1:,0] = new_observation[j,i+1:,0] + 2*np.pi
            if c < -6:
                new_observation[j,i+1:,0] = new_observation[j,i+1:,0] - 2*np.pi
            
    return new_observation

#create batch for training

#State transition function
def fx(x, sT):
    """ state transition function for sstate [downrange, vel, altitude]"""
    F = np.array([[1,0,sT,0],[0,1,0,sT],[0,0,1,0],[0,0,0,1]],'float64') #F_cv

    return np.dot(F, x)
#observationvation function
def hx(x):
    """ returns slant range = np.array([[0],[0]]) based on downrange distance and altitude"""
    r = np.array([0,0],'float64')
    r[0] = np.arctan2(x[1],x[0])
    r[1] = np.sqrt(np.square(x[0])+np.square(x[1]))
    return r

#Batch creating
def create_batch(pos_noise,vel_noise,batch_size,data_len):
    gt_trajectory, observation, _ = trajectory_batch_generator(batch_size,data_len)
    ukf_trajectory = np.zeros((batch_size, data_len, 4))
    for i in range(batch_size):
        my_SP = SP(dim_state, kappa=0.)
        my_UKF = UKF(dim_x=dim_state, dim_z=dim_observation, dt=sT, hx=hx, fx=fx, points=my_SP)
        my_UKF.Q *= var_m
        my_UKF.R *= np.array([[azi_var,0],[0,dis_var]])
        x0_noise = np.array([np.random.normal(0,pos_noise,2),np.random.normal(0,vel_noise,2)])
        my_UKF.x = gt_trajectory[i,0,:] + np.reshape(x0_noise,[4,])
        my_UKF.P *= 1.
        
        #tracking results of UKF
        xs = []
        xs.append(my_UKF.x)
        for j in range(data_len-1):
            my_UKF.predict()
            my_UKF.update(observation[i,j+1,:])
            xs.append(my_UKF.x.copy())    
        ukf_trajectory[i] = np.asarray(xs)
        
    return gt_trajectory, observation, ukf_trajectory, gt_trajectory-ukf_trajectory

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import iqr

def norm_features(gt_trajectory, observation, ukf_trajectory, gt_ukf_residual):
    median = 16 * iqr(np.abs(gt_ukf_residual), rng=(0, 85), axis=0)  # ,
    ne = np.abs(gt_ukf_residual) > median
    ne = ne.any(axis=2)
    ukf_trajectory[ne] = 0
    gt_ukf_residual[ne] = 0
    dist_norm = np.linalg.norm(ukf_trajectory[:, :, 0:2], axis=2)
    vel_norm = np.linalg.norm(ukf_trajectory[:, :, 2:4], axis=2)
    acc_norm = np.diff(vel_norm, prepend=0)/0.1
    dist_diff_norm = np.diff(dist_norm, prepend=0)
    # ukf_trajectory = np.dstack((ukf_trajectory, dist_norm))
    # ukf_trajectory = np.dstack((ukf_trajectory, vel_norm))
    # ukf_trajectory = np.dstack((ukf_trajectory, acc_norm))
    # ukf_trajectory = np.dstack((ukf_trajectory, dist_diff_norm))
    # ukf_trajectory = np.dstack((ukf_trajectory, ne.astype(np.int)))

    ukf_trajectory_norm = np.zeros_like(ukf_trajectory)
    # ukf_trajectory_norm[:, :, 0:2] = ukf_trajectory[:, :, 0:2] / np.linalg.norm(ukf_trajectory[:, :, 0:2])
    # ukf_trajectory_norm[:, :, 2:4] = ukf_trajectory[:, :, 2:4] / np.linalg.norm(ukf_trajectory[:, :, 2:4])
    # scaler = MinMaxScaler()  # (feature_range=[-1, 1])
    # for i in range(4):
    #     scaler.fit(ukf_trajectory[:,:,i])
    #     ukf_trajectory_norm[:,:,i] = scaler.transform(ukf_trajectory[:,:,i])
    for i in range(4):
        if(i==0 or i==1):
            weight = Dist_max#np.max(np.abs(ukf_trajectory[:,:,i]))
        if(i==2 or i==3):
            weight = Velo_max  # np.max(np.abs(ukf_trajectory[:,:,i]))
        ukf_trajectory_norm[:,:,i] = ukf_trajectory[:,:,i] / weight

    # for i in range(4):
    #     weight = np.max(np.abs(gt_ukf_residual[:,:,i]))
    #     gt_ukf_residual[:,:,i] = gt_ukf_residual[:,:,i] / weight

    return (gt_trajectory, observation, ukf_trajectory, gt_ukf_residual, ukf_trajectory_norm, median)

def make_generator(batch_size, timestep_size, full=0):
    while True:
        gt_trajectory, observation, ukf_trajectory, gt_ukf_residual = create_batch(30, 3, batch_size, timestep_size)
        gt_trajectory, observation, ukf_trajectory, gt_ukf_residual, ukf_trajectory_norm, median = norm_features(gt_trajectory, observation, ukf_trajectory, gt_ukf_residual)
        if full:
            yield (gt_trajectory, observation, ukf_trajectory, gt_ukf_residual, ukf_trajectory_norm, median)
        else:
            # ukf_trajectory_norm = np.random.random(ukf_trajectory_norm.shape)
            # gt_ukf_residual = np.random.random(gt_ukf_residual.shape)
            yield (ukf_trajectory_norm, gt_ukf_residual)