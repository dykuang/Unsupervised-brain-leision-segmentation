# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 15:44:21 2019

@author: dykua

Loss functions
"""
import tensorflow as tf
from helpfns import apply_cluster
from keras.losses import mean_squared_error, kullback_leibler_divergence
#import keras.backend as K

def seg_loss(N=3, thres = 0.1):
    def Loss(y_true, y_pred):
#        cond = tf.greater(y_pred, tf.ones(tf.shape(y_pred))*thres)
#        mask = tf.where(cond, tf.ones(tf.shape(y_pred)), tf.zeros(tf.shape(y_pred))) # use tf.clip_by_value to make a softer cut?
#        loss = mean_squared_error(tf.multiply(y_true,mask), y_pred)
        rec = tf.reduce_sum(y_pred, axis=-1)
        loss = mean_squared_error(y_true[...,0], rec)
        return loss
    return Loss


'''
the inner class loss with centers as extra parameter
'''

def inertia(centers, use_weight=False):
    '''
    centers: (n clusters, n features)
    '''
    def loss(yTrue, yPred):
        '''
        yPred: (None, res, res, feature_num)
        '''
        score = apply_cluster(yPred, centers)
        label_img = tf.argmax(score, -1) #(None, res, res)

        label_OH = tf.one_hot(label_img, depth=len(centers)) #(None, res, res, cluster_num)
        C_tensor = tf.constant(centers) # (cluster_num, feature_num)
        C_pix = tf.einsum('bhwc,cf->bhwf', label_OH, C_tensor) #(None, res, res, feature_num)
        
        dist_img = tf.reduce_sum(tf.square(yPred - C_pix), axis=-1)/centers.shape[-1] #(None, res, res)
        return tf.reduce_mean( dist_img ) #use weighted sum here
    
    def loss_weighted(yTrue, yPred):
        '''
        yPred: (None, res, res, feature_num)
        '''
        score = apply_cluster(yPred, centers)
        label_img = tf.argmax(score, -1) #(None, res, res)
        label_count = [tf.reduce_sum( tf.cast(
                                        tf.equal(label_img, i*tf.ones_like(label_img)), tf.float32
                                        ) ) for i in range(len(centers))]
        
        label_OH = tf.one_hot(label_img, depth=len(centers)) #(None, res, res, cluster_num)
        C_tensor = tf.constant(centers) #(cluster_num, feature_num)
        C_pix = tf.einsum('bhwc,cf->bhwf', label_OH, C_tensor) #(None, res, res, feature_num)
        
        weight = tf.zeros_like(label_img)
#        is_zero = [tf.greater(label_count[i], 0) for i in range(len(centers))]
        
        for i in range( len(label_count) ):
            weight = tf.cast(weight, tf.float32) + (1/ tf.maximum(label_count[i], tf.constant(1.)) ) * tf.cast(label_OH[...,i], tf.float32) 
        
        dist_img = tf.reduce_sum(tf.square(yPred - C_pix), axis=-1)/centers.shape[-1] #(None, res, res)
        return tf.reduce_mean(tf.multiply(dist_img, weight)) #use weighted sum here
    
    if use_weight:
        return loss_weighted
    else:
        return loss

'''
softmax loss with labels as extra parameter
'''
#from keras.losses import categorical_crossentropy
def CL_loss(n_clusters): # need to carefully consider that shape, check doc for tf.one_hot
    def loss(yT, yP):
        '''
        yT, yP: (None, pix_num, 3)
        '''
        yP_flat = tf.reshape(yP, (-1, n_clusters) )# (None, 3)
        yP_label =tf.argmax(yP_flat, axis=-1) #(None,)
        yP_OH = tf.one_hot(yP_label, depth=n_clusters) #(None, 3)
               
        return tf.losses.softmax_cross_entropy(yP_OH, yP_flat)  
    return loss

def tf_target_distribution(q):
    '''
    q: (None, 3)
    '''
    weight = q ** 2 / tf.reduce_sum(q, 0)
    return tf.transpose(tf.transpose(weight) / tf.reduce_sum(weight, 1))

def KL_loss(n_clusters):
    def loss(yT, yP):
        '''
        yT, yP: (None, pix_num, 3)
        '''
        yP_flat = tf.reshape(yP, (-1, n_clusters))
        p = tf_target_distribution(yP_flat)
        
        return tf.reduce_mean(kullback_leibler_divergence(yP_flat, p))
    return loss
    
    
