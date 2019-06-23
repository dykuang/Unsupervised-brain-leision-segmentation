#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 14:00:54 2019

@author: dykuang
"""

from architecture import build_model_3 as build_model
import numpy as np
from helpfns import get_batch
from sklearn.cluster import MiniBatchKMeans
from keras.optimizers import Adam
#from keras.layers import Input

output_dir = r'/home/dykuang/UMI-SEG/output/'
prefix = 'model3_'
params = np.load(output_dir+prefix+'parameters_pretrain.pkl')

import os
#params['training data list'] = []
#for f in os.listdir(params['training data path']):
#    x = np.load(params['training data path']+f)
#    if np.max(x) > 0.7:
#        params['training data list'].append(f)

params['training data list']=['vol_5_slice_{}.npy'.format(i) for i in range(12,140)]  #+ ['vol_18_slice_{}.npy'.format(i) for i in range(22,122)]
train_list = params['training data list']
params['batch size'] = 32 
params['prefix'] = prefix

AE,feature_map = build_model(input_size=(params['image res'],params['image res'], params['input channels']),
                             en_spec = params['en spec'], de_spec=params['de spec'],
                             n_features = params['n features'], n_clusters=params['n clusters'] )
AE.load_weights(output_dir+prefix+'Pretrained_weights.h5')

# check if weights are loaded correctly
file_list = ['vol_1_slice_{}.npy'.format(i) for i in range(72,82)]
img_batch = get_batch(params['training data path'], file_list)
rec_U = AE.predict(img_batch)
np.save(params['output dir'] + prefix + 'rec_pretrain.npy', rec_U)

'''
Initialize the clustering layer
'''

encoder = AE.layers[-2]
params['training size'] = len(train_list)
params['KM batch size'] = 32 
params['ite per epoch'] = len(params['training data list'])//params['KM batch size']

MB_kmean = MiniBatchKMeans(n_clusters=params['n clusters'],
                                  random_state=0,
                                  batch_size=params['KM batch size'],
                                  max_no_improvement = 20)

for ite in range(params['ite per epoch']):
    img_batch = get_batch(params['training data path'], 
                          train_list[ite * params['KM batch size']: 
                                      min((ite+1) * params['KM batch size'], params['training size'])])
    if len(img_batch) == 0:
        break
            
    X_f = encoder.predict(img_batch)
    MB_kmean.partial_fit(X_f.reshape(-1, params['n features'])) # fit with sample weights?
        
cluster_centers = MB_kmean.cluster_centers_
feature_map.get_layer(name='clustering').set_weights([cluster_centers])

# check if initialization makes sense
file_list = ['vol_1_slice_{}.npy'.format(i) for i in range(72,82)]
img_batch = get_batch(params['training data path'], file_list)

X_f = encoder.predict(img_batch)
seg_KM = MB_kmean.predict(X_f.reshape(-1, params['n features']))
seg_KM = seg_KM.reshape((-1, params['image res'], params['image res']))
np.save(params['output dir'] + prefix + 'seg_km.npy', seg_KM)

seg_U = feature_map.predict(img_batch)
seg_U = np.argmax(seg_U, -1).reshape((-1, params['image res'], params['image res']))
np.save(params['output dir'] + prefix + 'seg_ini.npy', seg_U)

# All the above can be splited into a pre-train file
# The following loads the weight and retrain.
    
from keras.models import Model   
from losses import inertia
from helpfns import get_batch,datagen_AECL

params['loss weights'] = [1.0,1.0]  # adding KL loss here?
params['whole lr'] = 1e-4
    
AECL = Model(AE.input, [AE.output, encoder.get_output_at(1)])
gen_train = datagen_AECL(params['training data path'], params['training data list'], params['batch size'], params['image res'],params['n features'], params['input channels'], aug=True)
#gen_train_2 = datagen_FM(params['training data path'], params['training data list'], params['batch size'], params['image res'], params['n clusters'], params['input channels'], aug=True)
params['loop num'] = 5 
params['AECL epochs'] = 3 
params['KM epochs'] = 2

pock_C = [cluster_centers]
CLAE_history = [MB_kmean.inertia_]

AECL_history = []

for loop_num in range(params['loop num']):
    # fix CL layer, train the encoder-decoder with inner class distance loss and reconstruction loss
    AECL.compile(loss= ['mse', inertia(centers=cluster_centers, use_weight = False)],
                    loss_weights=params['loss weights'], 
                    optimizer=Adam(lr = params['whole lr'], decay = 1e-5))
    AECL_history.append(
            AECL.fit_generator(gen_train, 
                               steps_per_epoch = len(params['training data list'])/params['batch size'], 
                               epochs = params['AECL epochs'], verbose=1) 
            )


    # fix auto encoder, update cluster center ?
    '''
    Different ideas here
    
    * getbatch --> encoder --> minibatch kmean (initialized by last time)--> set CL weights
    
    * Getbatch --> encoder --> manually update by methods such as in "k-mean friendly space" paper
    
    * Model(AE.input, CL.output) --> Freeze encoder ---> minimize weights like within-class variance
    '''
    
    # Idea 1
    MB_kmean = MiniBatchKMeans(n_clusters=params['n clusters'],
                                  random_state=0,
                                  batch_size=params['KM batch size'],
                                  max_no_improvement = 20,
                                  init = cluster_centers)  # One way to soft ensure 'center - cluster label' correspondence does not change        
    for epoch_i in range(params['KM epochs']):
        
        for ite in range(params['ite per epoch']):
            img_batch = get_batch(params['training data path'], 
                                  train_list[ite * params['KM batch size']: 
                                      min((ite+1) * params['KM batch size'], params['training size'])])
            if len(img_batch) == 0:
                np.random.shuffle(train_list)
                break
            
            X_f = encoder.predict(img_batch)
            MB_kmean.partial_fit(X_f.reshape(-1, params['n features'])) # fit with sample weights?
        # could add a boosting module here for resampling thoses with different predicted labels      
        cluster_centers = MB_kmean.cluster_centers_
        pock_C.append(cluster_centers)
        CLAE_history.append(MB_kmean.inertia_)
        
    feature_map.get_layer(name='clustering').set_weights([cluster_centers]) 
        

'''
Save weights
'''
import pickle
with open(output_dir+prefix+'parameters_train.pkl', 'wb') as f:
    pickle.dump(params, f)
with open(output_dir+prefix+'AECL_loss.pkl', 'wb') as f:
    pickle.dump(AECL_history, f)
with open(output_dir+prefix+'CLAE_loss.pkl', 'wb') as f:
    pickle.dump(CLAE_history, f)
with open(output_dir+prefix+'center_track.pkl', 'wb') as f:
    pickle.dump(pock_C, f )

decoder = AE.layers[-1]
clustering_layer = feature_map.get_layer(name='clustering')

encoder.save_weights(output_dir+prefix+'encoder_weights.h5')
decoder.save_weights(output_dir+prefix+'decoder_weights.h5')
np.save(output_dir+prefix+'CL_layer_weights.npy', clustering_layer.get_weights())

'''
check some prediction
'''
file_list = ['vol_1_slice_{}.npy'.format(i) for i in range(72,82)]
img_batch = get_batch(params['training data path'], file_list)

rec_U = AE.predict(img_batch)
seg_U = feature_map.predict(img_batch)
seg_U = np.argmax(seg_U, -1).reshape((len(file_list), params['image res'], params['image res']))
np.save(params['output dir'] +prefix +'rec_U_img_train.npy', rec_U)
np.save(params['output dir'] +prefix+ 'seg_U.npy', seg_U)
#print('*'*20 + '{}'.format( np.max(seg_U) ) + '*'*20 )
#
#'''
#check dice
#'''
#from helpfns import dice_coef
#label = np.load('tst_vol.npy')
#D = [dice_coef(seg_U==i, label) for i in range(params['n clusters'])]
#print( D )

