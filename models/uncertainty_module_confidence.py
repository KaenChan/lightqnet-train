from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import tensorflow.contrib.slim as slim


batch_norm_params = {
    'decay': 0.995,
    'epsilon': 0.001,
    'center': True,
    'scale': True,
    'updates_collections': None,
    'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
}

batch_norm_params_sigma = {
    'decay': 0.995,
    'epsilon': 0.001,
    'center': False,
    'scale': False,
    'updates_collections': None,
    'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],}


def scale_and_shift(x, gamma_init=1.0, beta_init=0.0):
    num_channels = x.shape[-1].value
    with tf.variable_scope('scale_and_shift'):
        gamma = tf.get_variable('alpha', (),
                        initializer=tf.constant_initializer(gamma_init),
                        regularizer=slim.l2_regularizer(0.0),
                        dtype=tf.float32)
        beta = tf.get_variable('gamma', (),
                        initializer=tf.constant_initializer(beta_init),
                        dtype=tf.float32)
        # x = tf.exp(gamma) * x +  beta
        x = gamma * x +  beta

        return x   
    

def inference(inputs, embedding_size, phase_train, 
        weight_decay=5e-4, reuse=None, scope='UncertaintyModule'):
    with slim.arg_scope([slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        activation_fn=tf.nn.relu):
        with tf.variable_scope(scope, [inputs], reuse=reuse):
            with slim.arg_scope([slim.batch_norm, slim.dropout],
                                is_training=phase_train):
                print('UncertaintyModule input shape:', [dim.value for dim in inputs.shape])

                net = slim.flatten(inputs)

                if embedding_size < 256:
                    fc1_size = 256
                else:
                    fc1_size = embedding_size
                net = slim.fully_connected(net, fc1_size, scope='fc1',
                    normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params, 
                    activation_fn=tf.nn.relu)

                net = slim.fully_connected(net, embedding_size, scope='fc_log_sigma_sq',
                    normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params_sigma,
                    activation_fn=None)

                id_quality = tf.sigmoid(net)
    return id_quality, net
