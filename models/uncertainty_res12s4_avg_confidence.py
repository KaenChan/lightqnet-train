import tensorflow as tf
import tensorflow.contrib.slim as slim

'''
Resface20 and Resface36 proposed in sphereface and applied in Additive Margin Softmax paper
Notice:
batch norm is used in line 111. to cancel batch norm, simply commend out line 111 and use line 112
'''

def prelu(x):
    with tf.variable_scope('PRelu'):
        pos = tf.nn.relu(x)
        return pos

def resface_block(lower_input,output_channels,scope=None):
    with tf.variable_scope(scope):
        net = slim.conv2d(lower_input, output_channels,weights_initializer=tf.truncated_normal_initializer(stddev=0.01))
        net = slim.conv2d(net, output_channels,weights_initializer=tf.truncated_normal_initializer(stddev=0.01))
        return lower_input + net

def resface_pre(lower_input,output_channels,scope=None):
    net = slim.conv2d(lower_input, output_channels, stride=2, scope=scope)
    return net

def resface18(images, keep_probability,
              phase_train=True, bottleneck_layer_size=512,
              weight_decay=0.0, reuse=None):
    '''
    conv name
    conv[conv_layer]_[block_index]_[block_layer_index]
    '''
    end_points = {}
    with tf.variable_scope('Conv1'):
        net = resface_pre(images,64//4,scope='Conv1_pre') # 48x48
        net = slim.repeat(net,1,resface_block,64//4,scope='Conv1')
    end_points['Conv1'] = net
    with tf.variable_scope('Conv2'):
        net = resface_pre(net,128//4,scope='Conv2_pre') # 24x24
        net = slim.repeat(net,1,resface_block,128//4,scope='Conv2')
    end_points['Conv2'] = net
    with tf.variable_scope('Conv3'):
        net = resface_pre(net,256//4,scope='Conv3_pre') # 12x12
        net = slim.repeat(net,1,resface_block,256//4,scope='Conv3')
    end_points['Conv3'] = net
    with tf.variable_scope('Conv4'):
        net = resface_pre(net,512//4,scope='Conv4_pre') # 6x6
        net = slim.repeat(net,1,resface_block,512//4,scope='Conv4')
    end_points['Conv4'] = net
    with tf.variable_scope('Logits'):
        #pylint: disable=no-member
        net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
                              scope='AvgPool')
        net = slim.flatten(net)
        net = slim.dropout(net, keep_probability, is_training=phase_train,
                           scope='Dropout')
    net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None,
                               scope='Bottleneck', reuse=False)
    id_quality = tf.sigmoid(net)
    return id_quality, net


def inference(image_batch,
              embedding_size=128,
              phase_train=True,
              keep_probability=1.0,
              weight_decay=1e-5,
              scope='UncertaintyModule'):
    batch_norm_params = {
        'decay': 0.995,
        'epsilon': 0.001,
        'scale':True,
        'is_training': phase_train,
        'updates_collections': None,
        'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
    }
    # image_batch = tf.image.resize_images(image_batch, (64, 64))
    with tf.variable_scope(scope):
        with slim.arg_scope([slim.conv2d, slim.fully_connected], 
                             weights_initializer=tf.contrib.layers.xavier_initializer(),
                             weights_regularizer=slim.l2_regularizer(weight_decay), 
                             activation_fn=prelu,
                             normalizer_fn=slim.batch_norm,
                             #normalizer_fn=None,
                             normalizer_params=batch_norm_params):
            with slim.arg_scope([slim.conv2d], kernel_size=3):
                return resface18(images=image_batch,
                                 keep_probability=keep_probability,
                                 phase_train=phase_train,
                                 bottleneck_layer_size=embedding_size,
                                 reuse=None)
