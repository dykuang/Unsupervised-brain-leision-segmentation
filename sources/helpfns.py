# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 15:21:47 2019

@author: dykua

Some utility functions
"""

import numpy as np
import os
import tensorflow as tf
import keras.backend as K

def dice_coef(X,Y):
    overlap = np.multiply(X, Y)

    return 2 * overlap.sum()/(X.sum() + Y.sum() + 1e-5)

def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

def pix_target_distribution(q): # stack member distribution in a batch. 
    assert len(q.shape) == 3
    temp = []
    for i in range(q.shape[0]):
        temp.append(target_distribution(q[i]))
    return np.stack(temp, 0) 


def get_batch(path, file_list):
    output=[]
    for name in file_list:
        output.append(np.load(path+name))
    output=np.array(output)
    if len(output.shape)<4:
        return output[...,None]
    else:
        return output

def datagen(datapath, datalist, batchsize, res, num_channel, aug=False):
    x = np.zeros([batchsize, res, res,num_channel])
    size = len(datalist)
    n_batches = size//batchsize
    index = 0
    while True:
        for i in range(batchsize):
            files = datalist[(index*batchsize + i)%size]
            x[i,...,0] = np.load(datapath + files)
            if aug:
                flag = np.random.randint(0,8)
                if flag == 1:
                    x[i] = x[i,::-1,...]
                elif flag == 2:
                    x[i] = x[i,:,::-1,:]
                elif flag == 3:
                    x[i] = x[i,::-1,::-1,:]           
            
        yield x, x
        index = index + 1
        if index > n_batches:
            index = 0
            np.random.shuffle(datalist)
            
def datagen_AECL(datapath, datalist, batchsize, res, num_feature, num_channel, aug=False):
    x = np.zeros([batchsize, res, res,num_channel])
    size = len(datalist)
    n_batches = size//batchsize
    index = 0
    while True:
        for i in range(batchsize):
            files = datalist[(index*batchsize + i)%size]
            x[i,...,0] = np.load(datapath + files)
            if aug:
                flag = np.random.randint(0,8)
                if flag == 1:
                    x[i] = x[i,::-1,...]
                elif flag == 2:
                    x[i] = x[i,:,::-1,:]
                elif flag == 3:
                    x[i] = x[i,::-1,::-1,:]
                
        yield x, [x, np.empty((batchsize, res, res, num_feature))]
        index = index + 1
        if index > n_batches:
            index = 0
            np.random.shuffle(datalist)
            
def datagen_FM(datapath, datalist, batchsize, res, num_cluster, num_channel, aug=False):
    x = np.zeros([batchsize, res, res, num_channel])
    size = len(datalist)
    n_batches = size//batchsize
    index = 0
    while True:
        for i in range(batchsize):
            files = datalist[(index*batchsize + i)%size]
            x[i,...,:num_channel] = np.load(datapath + files)
            if aug:
                flag = np.random.randint(0,8)
                if flag == 1:
                    x[i] = x[i,::-1,...]
                elif flag == 2:
                    x[i] = x[i,:,::-1,:]
                elif flag == 3:
                    x[i] = x[i,::-1,::-1,:]
                
        yield x, np.empty((batchsize, res*res, num_cluster))
        index = index + 1
        if index > n_batches:
            index = 0
            np.random.shuffle(datalist)
            

            
def apply_cluster(inputs, centers, alpha=1.0):
    '''
    inputs:(None, res, res, num_features)
    '''
    res = K.int_shape(inputs)[1]
    inputs_reshaped = tf.reshape(inputs, (-1, inputs.shape[-1]))
        
    q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs_reshaped, axis=1) - centers), axis=2) /alpha))
    q **= (alpha + 1.0) / 2.0
        
    q = K.transpose(K.transpose(q) / K.sum(q, axis=1)) # Make sure each sample's 10 values add up to 1.                      
    #print(K.int_shape(inputs))  
    q_reshaped = tf.reshape(q, (-1, res, res, len(centers)))      
    #print(q_reshaped.shape)
    return q_reshaped

def get_train_data(path, index):
    files = os.listdir(path)
    data_out = []
    for idx in index:
        for f in files:
            if 'vol_{}_'.format(idx+1) in f:
                data_out.append(f)
                                                                    
    return data_out
