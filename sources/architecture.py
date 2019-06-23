# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 16:17:32 2019

@author: dykua

architectures for the network
"""
from keras.layers import Input, Conv2D, Conv2DTranspose, Reshape, Lambda, MaxPooling2D, UpSampling2D, Dropout, concatenate, multiply, add, BatchNormalization, PReLU, GaussianNoise, ZeroPadding2D
from keras.models import Model
from Mylayers import ClusteringLayer_ver2, Feature_split
import tensorflow as tf


def encoder_block(x, dim_list):
    y = Conv2D(dim_list[0], kernel_size=3, strides=1, activation='relu', padding='same')(x)
    y = Conv2D(dim_list[0], kernel_size=3, strides=2, activation='relu')(y)
    for dim in dim_list[1:-1]:
        y = Conv2D(dim, kernel_size=3, strides=1, activation='relu', padding='same')(y)
        y = Conv2D(dim, kernel_size=3, strides=2, activation='relu')(y)
    
    y = Conv2D(dim_list[-1], kernel_size=1, strides=1, activation='relu', padding='same')(y) # emebding layer
    return y

def decoder_block(x, dim_list):
    y = Conv2DTranspose(dim_list[0], kernel_size=3, strides=2,activation='relu')(x)
    y = Conv2D(dim_list[0], kernel_size=3, strides=1, activation='relu', padding='same')(y)
    for dim in dim_list[1:-1]:
        y = Conv2DTranspose(dim, kernel_size=3, strides=2,activation='relu')(y)
        y = Conv2D(dim, kernel_size=3, strides=1, activation='relu', padding='same')(y)
        
    y = Conv2D(dim_list[-1], kernel_size=1, strides=1,activation='relu', padding='same')(y) # output layer
    return y

def create_encoder(inputs, dim_list):
    output = encoder_block(inputs, dim_list)
    return Model(inputs, output)

def create_decoder(inputs, dim_list):
    output = decoder_block(inputs, dim_list)
    return Model(inputs, output)

#def make_cluster(inputs, filter_func = lambda x: 1/(1+tf.exp(-10*(x-0.5))), n_clusters, name='clustering'):       
#    clusters = ClusteringLayer_ver2(n_clusters, filter_func, name)(inputs)
#    return clusters

def build_whole_model(inputs, en_dim_list, de_dim_list, n_clusters, filter_func = lambda x: 1/(1+tf.exp(-10*(x-0.5)))):
    encoder = create_encoder(inputs, en_dim_list)
    feature = encoder(inputs) # end of encoder
    
    feature_reshaped = Reshape( (feature.shape[1] * feature.shape[2], en_dim_list[-1]) )(feature) # Did not specify batch size explicitly in Reshape layers
    CLayer = ClusteringLayer_ver2(n_clusters, filter_func, name='clustering')
    x_clusters_reshaped = CLayer(feature_reshaped)  
    x_clusters = Reshape((feature.shape[1], feature.shape[2], n_clusters))(x_clusters_reshaped) # end of clustering
    
    x_splited=Feature_split(en_dim_list[-1], n_clusters)([feature, x_clusters])  # feature splitted according to clusters
    
    decoder_input = Input((feature.shape[1], feature.shape[2], en_dim_list[-1]))
    decoder = create_decoder(decoder_input, de_dim_list)
    decoded = decoder(feature) # end of decoder
      
    Pred_label=[]
    for i in range(n_clusters):
        Pred_label.append(decoder(x_splited[i]))      
    Squeezed = Lambda(lambda x: tf.squeeze(tf.stack(x,axis=-1), axis=-2))
    
    
    AE = Model(inputs, decoded)
    feature_map = Model(inputs, x_clusters_reshaped)
    mask_map = Model(inputs, Squeezed(Pred_label))
    whole_model = Model(inputs, [AE.output, feature_map.output, mask_map.output])
    
    return AE, feature_map, mask_map, whole_model
    
def unet_CL(n_clusters, filter_func = lambda x: 1/(1+tf.exp(-10*(x-0.5))), pretrained_weights = None,input_size = (256,256,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
#    conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
#    drop4 = Dropout(0.25)(conv4)
#    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
#
#    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
#    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
#    drop5 = Dropout(0.25)(conv5)
#    
#    up6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
#    merge6 = concatenate([drop4,up6], axis = 3)
#    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
#    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv4))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
#    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'relu')(conv9)

    # segmentation branch
    feature_reshaped = Reshape( (conv4.shape[1] * conv4.shape[2], 256) )(conv4) # Did not specify batch size explicitly in Reshape layers
    CLayer = ClusteringLayer_ver2(n_clusters, filter_func, name='clustering')
    x_clusters_reshaped = CLayer(feature_reshaped)  
    x_clusters = Reshape((conv4.shape[1], conv4.shape[2], n_clusters))(x_clusters_reshaped) # end of clustering
    
    x_splited=Feature_split(conv4.shape[3], n_clusters)([conv4, x_clusters])  # feature splitted according to clusters
    
    decoder_input = Input((conv4.shape[1], conv4.shape[2], conv4.shape[3]))
    decoder = Model([inputs, decoder_input], conv10)
    
    Pred_label=[]
    for i in range(n_clusters):
        Pred_label.append(decoder([inputs, x_splited[i]]))      
    Squeezed = Lambda(lambda x: tf.squeeze(tf.stack(x,axis=-1), axis=-2))
    
    #models
    encoder = Model(inputs, conv4)
    feature = encoder(inputs)
    decoded = decoder([inputs, feature])
    AE = Model(inputs, decoded)
    feature_map = Model(inputs, x_clusters_reshaped)
    mask_map = Model(inputs, Squeezed(Pred_label))
    whole_model = Model(inputs, [AE.output, feature_map.output, mask_map.output])
    
    #model.summary()

    if(pretrained_weights):
      AE.load_weights(pretrained_weights[0])
      whole_model.get_layer(name='clustering').set_weights(pretrained_weights[1])
    return AE, encoder, feature_map, mask_map, whole_model 
    
def unet_AE(input_size = (256,256,1), pretrained_weights = None):
    inputs = Input(input_size)
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    
#    conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
#    drop4 = Dropout(0.25)(conv4)
#    drop4 = conv4
#    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
#
#    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
#    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
##    drop5 = Dropout(0.25)(conv5)
#    drop5 = conv5
    
#    up6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv4))
#    merge6 = concatenate([conv4,up6], axis = 3)
#    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
#    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv4))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
#    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'relu')(conv9)

    AE = Model(inputs, conv10)
    
    #model.summary()

    if(pretrained_weights):
      AE.load_weights(pretrained_weights[0])
      
    return AE    

def build_model_2(n_clusters, num_start = 16, pretrained_weights = None,input_size = (256,256,1)):
    inputs = Input(input_size, name='input--encoder')
    conv1 = Conv2D(num_start, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(num_start, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
                        
    conv2 = Conv2D(2*num_start, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(2*num_start, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
                                        
    conv3 = Conv2D(2*num_start, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(2*num_start, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
                                                    

    up4 = Conv2D(2*num_start, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv3))
    merge4 = concatenate([conv2,up4], axis = 3)
    conv5 = Conv2D(2*num_start, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge4)
    conv5 = Conv2D(2*num_start, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)

    up6 = Conv2D(num_start, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
    merge6 = concatenate([conv1,up6], axis = 3)
    conv7_0 = Conv2D(num_start, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
                                                                                    
    encoder = Model(inputs, conv7_0) # where to put output of the encoder? merge6, conv7_0, conv7_1
    input_de = Input( (input_size[0], input_size[1], num_start)   , name = 'input--decoder')

    conv7_1 = Conv2D(num_start, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_de)   
    conv8 = Conv2D(3, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7_1)
    conv9 = Conv2D(input_size[2], 1, activation = 'relu')(conv8)
                                                                                                                    
    decoder = Model(input_de, conv9)
                                                                                                                            
    feature = encoder(inputs)  
    decoded = decoder(feature)
                                                                                                                                            
    feature_reshaped = Reshape( (feature.shape[1] * feature.shape[2], num_start) )(feature) # Did not specify batch size explicitly in Reshape layers
    CLayer = ClusteringLayer_ver2(n_clusters, name='clustering') 
    x_clusters_reshaped = CLayer(feature_reshaped)  

    AE = Model(inputs, decoded)
    feature_map = Model(inputs, x_clusters_reshaped)
                                                                                                                                                                        
    return AE, feature_map


def build_model(input_size = (256,256,1), en_spec = [8,16,16], de_spec=[8,4], n_features = 8, n_clusters=3):
    inputs = Input(input_size, name='input--encoder')
    memo = []
    aug_input = GaussianNoise(0.05)(inputs)            
    
    conv = Conv2D(en_spec[0], 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(aug_input)
    conv = Conv2D(en_spec[0], 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv)
    pool = MaxPooling2D(pool_size=(2, 2))(conv)
    memo.append(conv)
                                    
    for num in en_spec[1:-1]:
        conv = Conv2D(num, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool)
        conv = Conv2D(num, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv)
        pool = MaxPooling2D(pool_size=(2, 2))(conv)
        memo.append(conv)
                                                                                
    conv = Conv2D(en_spec[-1], 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool)
    conv = Conv2D(en_spec[-1], 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv)
                                                                                     
    for i, num in enumerate(en_spec[-2::-1]):
        up = Conv2D(num, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv))
        merge = concatenate([memo[-i-1],up], axis = 3)
        #merge = add([memo[-i-1],up])
        conv = Conv2D(num, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge)
       # conv = multiply([conv, memo[-i-1]])
        if i== (len(en_spec) - 2 ):
            conv = Conv2D(n_features, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv)
        else:
            conv = Conv2D(num, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv)
    
 #   conv = BatchNormalization()(conv)
#    conv = PReLU(shared_axes=(1,2))(conv)

    encoder = Model(inputs, conv) # where to put output of the encoder? 
    input_de = Input( (input_size[0], input_size[1], n_features)   , name = 'input--decoder')

    conv_de = Conv2D(de_spec[0], 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_de)   
    if len(de_spec) > 1:
        for num in de_spec[1:]:
           conv_de = Conv2D(num, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv_de)
    
    conv_de = Conv2D(input_size[2], 1, activation = 'relu')(conv_de)
    decoder = Model(input_de, conv_de)
    feature = encoder(inputs)  
    decoded = decoder(feature)
    feature_reshaped = Reshape( (feature.shape[1] * feature.shape[2], n_features) )(feature) # Did not specify batch size explicitly in Reshape layers
    CLayer = ClusteringLayer_ver2(n_clusters, name='clustering') 
    x_clusters_reshaped = CLayer(feature_reshaped)  
    AE = Model(inputs, decoded)
    feature_map = Model(inputs, x_clusters_reshaped)
                                                                                                                                                                                                                                                                                                     
    return AE, feature_map   


def build_model_3(input_size = (256,256,1), en_spec = [8,16,16], de_spec=[8,4], n_features = 8, n_clusters=3):
    inputs = Input(input_size, name='input--encoder')
    memo = []
    aug_input = GaussianNoise(0.05)(inputs)            
    
    conv = Conv2D(en_spec[0], 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(aug_input)
    memo.append(conv)
    conv = Conv2D(en_spec[0], 3, strides=2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv)
    
                                    
    for num in en_spec[1:-1]:
        conv = Conv2D(num, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv)
        memo.append(conv)
        conv = Conv2D(num, 3, strides=2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv)
        
                                                                                
    conv = Conv2D(en_spec[-1], 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv)
                                                                                     
    for i, num in enumerate(en_spec[-2::-1]):
        up = Conv2DTranspose(num, 3, strides=2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv)
#        if up.shape[1] == memo[-i-1].shape[1]:   # shape is (?,?,?,int) for up?
        merge = concatenate([memo[-i-1],up], axis = 3)

        #merge = add([memo[-i-1],up])
        conv = Conv2D(num, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge)
       # conv = multiply([conv, memo[-i-1]])
        if i== (len(en_spec) - 2 ):
            conv = Conv2D(n_features, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv)
        else:
            conv = Conv2D(num, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv)
    
 #   conv = BatchNormalization()(conv)
#    conv = PReLU(shared_axes=(1,2))(conv)

    encoder = Model(inputs, conv) # where to put output of the encoder? 
    input_de = Input( (input_size[0], input_size[1], n_features)   , name = 'input--decoder')

    conv_de = Conv2D(de_spec[0], 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_de)   
#    if len(de_spec) > 1:
#        for num in de_spec[1:]:
#           conv_de = Conv2D(num, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv_de)
    
    for num in de_spec:
        conv_de_1 = Conv2D(num, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv_de)   
        conv_de_1 = Conv2D(num, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv_de_1)   
        conv_de = add([conv_de, conv_de_1])
        
    conv_de = Conv2D(input_size[2], 1, activation = 'relu')(conv_de)
    decoder = Model(input_de, conv_de)
    
    feature = encoder(inputs)  

    decoded = decoder(feature)
    feature_reshaped = Reshape( (encoder.output_shape[1] * encoder.output_shape[2], n_features) )(feature) # Did not specify batch size explicitly in Reshape layers
    CLayer = ClusteringLayer_ver2(n_clusters, name='clustering') 
    x_clusters_reshaped = CLayer(feature_reshaped)  
    AE = Model(inputs, decoded)
    feature_map = Model(inputs, x_clusters_reshaped)
                                                                                                                                                                                                                                                                                                     
    return AE, feature_map   
