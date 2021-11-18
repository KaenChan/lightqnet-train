# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Mobilenet V3 conv defs and helper functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import functools
import numpy as np

import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim

from models.mobilenet import conv_blocks as ops
from models.mobilenet import mobilenet as lib

slim = contrib_slim
op = lib.op
expand_input = ops.expand_input_by_factor
from models.mobilenet import mobilenet_v3

def inference(images,
              bottleneck_layer_size=128,
              phase_train=True,
              keep_probability=1.0,
              weight_decay=1e-5,
              scope='UncertaintyModule'):
    # conv_defs = mobilenet_v3.V3_LARGE_FACE
    conv_defs = mobilenet_v3.V3_LARGE_MINIMALISTIC_FACE_OP1S1
    # with slim.arg_scope(mobilenet_v3.training_scope(weight_decay=weight_decay)):
    arg_scope = mobilenet_v3.mobilenet_v2_arg_scope(is_training=phase_train, weight_decay=weight_decay)
    with slim.arg_scope(arg_scope):
        with tf.variable_scope(scope):
            prelogit, end_points = mobilenet_v3.mobilenet(images, num_classes=bottleneck_layer_size,
                                             depth_multiplier=1.0,
                                             conv_defs=conv_defs,
                                             is_training=phase_train)
            net = end_points['layer_18']
            net = tf.reshape(net, [-1, net.get_shape()[1]*net.get_shape()[2]*net.get_shape()[3]])
            end_points['last_conv'] = net
            print('last_conv', net.get_shape(), net)
            end_points['last_conv'] = net
            # net = slim.dropout(net, keep_probability, is_training=phase_train, scope='Dropout')
            net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None, normalizer_fn=None,
                                       scope='Bottleneck')
            print('Bottleneck', net.get_shape(), net)

            id_quality = tf.sigmoid(net)
            return id_quality, net
