# -*- coding: utf-8 -*-
"""
Created on Fri May  3 09:30:44 2019

@author: dykua

pretrain an 'autoencoder' with unet
"""

#from helpfns import datagen
from architecture import build_model_3 as build_model
#from keras.layers import Input
#from keras.models import Model
import numpy as np
import os

params = dict()
params['AE batch size'] = 32 
params['AE lr'] = 1e-4
params['training data path'] = r'/home/dykuang/Flair_seg/data/train/'
#params['training data list'] = os.listdir(params['training data path'])
#params['training data list']=['vol_1_slice_{}.npy'.format(i) for i in range(12,112)]
params['training data list'] = ['vol_5_slice_{}.npy'.format(i) for i in range(12,140)] + ['vol_1_slice_{}.npy'.format(i) for i in range(12,112)]
#for f in os.listdir(params['training data path']):
#    x = np.load(params['training data path']+f)
#    if np.max(x) > 0.3:
#        params['training data list'].append(f)

params['training size'] = len(params['training data list'])
params['AE epochs'] = 30 
params['image res'] = 256
#params['kernel size'] = 3
params['n clusters'] = 3
params['n features'] = 64 
params['output dir'] = r'/home/dykuang/UMI-SEG/output/'
params['input channels'] = 1
params['en spec'] = [8, 16, 32]  #Specify layer parameters for the U-net as encoder
params['de spec'] = [8, 8, 8]    #Specify layer parameters for the decoder

#params['training data list']=['vol_1_slice_{}.npy'.format(i) for i in range(10, 121)]

AE,_ = build_model(input_size=(params['image res'],params['image res'], params['input channels']), 
                   en_spec = params['en spec'], de_spec=params['de spec'],
                   n_features= params['n features'], n_clusters=params['n clusters'])
print(AE.summary())
print( AE.layers[-2].summary() )
print( AE.layers[-1].summary() )

def datagen(datapath, datalist, batchsize):
    x = np.zeros([batchsize, params['image res'], params['image res'],params['input channels']])
    size = len(datalist)
    n_batches = size//batchsize
    index = 0
    while True:
        for i in range(batchsize):
            files = datalist[(index*batchsize + i)%size]
            x[i,...,0] = np.load(datapath + files)
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


from keras.optimizers import Adam, SGD
from keras.losses import mean_squared_error
# train the auto-encoder
AE.compile(loss = 'mse', optimizer=Adam(lr = params['AE lr'], decay = 1e-5))
gen_train = datagen(params['training data path'], params['training data list'], params['AE batch size'])

AE.fit_generator(gen_train, steps_per_epoch = len(params['training data list'])//params['AE batch size'], epochs = params['AE epochs'], verbose=1)

params['prefix'] = 'model3_'
prefix = params['prefix']
AE.save_weights(params['output dir']+params['prefix']+'Pretrained_weights.h5')

from helpfns import get_batch
file_list = params['training data list'][50:60]
img_batch = get_batch(params['training data path'], file_list)
#print(np.max(img_batch))
rec_U = AE.predict(img_batch)
np.save(params['output dir'] + prefix +'rec_U_img.npy', rec_U)
np.save(params['output dir'] + prefix + 'imgbatch.npy', img_batch)
#print(np.max(rec_U))

import pickle
with open(params['output dir']+prefix+'parameters_pretrain.pkl', 'wb') as f:
    pickle.dump(params, f)

