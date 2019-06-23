# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 15:36:55 2019

@author: dykua

Contains custom layers
"""

import keras.backend as K
import tensorflow as tf
from keras.engine.topology import Layer, InputSpec

'''
ClusteringLayer from https://github.com/Tony607/Keras_Deep_Clustering/blob/master/Keras-DEC.ipynb
'''
class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: degrees of freedom parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight((self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True
    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
         Measure the similarity between embedded point z_i and centroid µ_j.
                 q_ij = 1/(1+dist(x_i, µ_j)^2), then normalize it.
                 q_ij can be interpreted as the probability of assigning sample i to cluster j.
                 (i.e., a soft assignment)
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1)) # Make sure each sample's 10 values add up to 1. 
#        q = K.transpose(K.transpose(K.exp(q)) / K.sum(K.exp(q), axis=1)) #Use softmax here?
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], self.n_clusters)

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

'''
PixClayer applies Clusterlayer to each member of in a batch
Input: a batch of feature map
'''
class PixCLayer(Layer):
    def __init__(self, CLayer, n_clusters, **kwargs):
#        self.output_dim = output_dim
        self.cluster_layer = CLayer
        self.n_clusters = n_clusters
        super(PixCLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
#        assert isinstance(input_shape, list)
#        # Create a trainable weight variable for this layer.
#        self.kernel = self.add_weight(name='kernel',
#                                      shape=(input_shape[0][1], self.output_dim),
#                                      initializer='uniform',
#                                      trainable=True)
        super(PixCLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        batch_size = x.shape[0]
        clusters = []
        for i in range(batch_size):
            clusters.append(self.cluster_layer(x[i]))
#        for xx in x:
#            clusters.append(self.cluster_layer(xx))  # suggest using tf.map_fn ?
        return K.stack(clusters, axis=0)

    def compute_output_shape(self, input_shape):
#        assert isinstance(input_shape, list)
#        shape_a, shape_b = input_shape
        return (input_shape[0], input_shape[1], self.n_clusters)
    
# mask layer
# input: mask block (none. pixel_num, ), feature block
# output: a list of features selected from the mask.
class Feature_split(Layer):

    def __init__(self, n_features, n_clusters, **kwargs):
        self.n_features = n_features
        self.n_clusters = n_clusters
        #self.filter_func = filter_func
        super(Feature_split, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        assert isinstance(input_shape, list)
        super(Feature_split, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        assert isinstance(x, list)
        feature, mask = x
        #mask = self.filter_func(mask)
        splited_features = []
        temp = []
        for i in range(self.n_clusters):
            start_idx = i * self.n_features
            end_idx = (i+1) * self.n_features
            for j in range(self.n_features):
                temp.append(tf.multiply(feature[...,j], mask[...,i])) # need a threshold to control the mask?
            splited_features.append(tf.stack(temp[start_idx:end_idx], axis=-1))
        return splited_features

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [shape_a for i in range(self.n_clusters)]
    

'''
A different clustering layer that works on a batch of image features by combining them into
one batch of pixel feature vectors.
'''
class ClusteringLayer_ver2(Layer):

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer_ver2, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=3)
       # self.filter_func = filter_func

    def build(self, input_shape):
        assert len(input_shape) == 3
        input_dim_pix = input_shape[1]
        input_dim_fts = input_shape[2]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim_pix, input_dim_fts))
        self.clusters = self.add_weight((self.n_clusters, input_dim_fts), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True
    def call(self, inputs, **kwargs):

        inputs_reshaped = tf.reshape(inputs, (-1, inputs.shape[2]))
        
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs_reshaped, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1)) # Make sure each sample's 10 values add up to 1. 
        #q = self.filter_func(q)
        
        q_reshaped = tf.reshape(q, (-1, inputs.shape[1], self.n_clusters))
        return q_reshaped

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 3
        return (input_shape[0], input_shape[1], self.n_clusters)

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer_ver2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
