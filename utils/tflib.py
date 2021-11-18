''' Functions for tensorflow '''
# MIT License
# 
# Copyright (c) 2019 Yichun Shi
# Copyright (c) 2021 Kaen Chan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import tensorflow as tf
import numpy as np

def negative_MLS(X, Y, sigma_sq_X, sigma_sq_Y, mean=False):
    print('sigma_sq_X.shape[1] =', sigma_sq_X.shape[1])
    with tf.name_scope('negative_MLS'):
        if sigma_sq_X.shape[1] == 1:
            # cosin distance sigma
            D = X.shape[1].value
            sigma_sq_fuse = sigma_sq_X + tf.transpose(sigma_sq_Y)
            # sigma_sq_fuse = tf.maximum(sigma_sq_X, sigma_sq_Y)
            X = tf.stop_gradient(X)
            Y = tf.stop_gradient(Y)
            sigma_sq_fuse_ = tf.stop_gradient(sigma_sq_fuse)
            # cos_theta = tf.matmul(X_, tf.transpose(Y_))
            cos_theta = tf.matmul(X, tf.transpose(Y))
            diffs = 2*(1-cos_theta) / (1e-10 + sigma_sq_fuse) + tf.log(sigma_sq_fuse)
            diffs_neg = 2*(1-cos_theta) / (1e-10 + sigma_sq_fuse) + tf.log(sigma_sq_fuse)
            attention = 2*(1-cos_theta) / (1e-10 + sigma_sq_fuse)
            return diffs, attention, diffs_neg
        else:
            D = X.shape[1].value
            X = tf.reshape(X, [-1, 1, D])
            Y = tf.reshape(Y, [1, -1, D])
            sigma_sq_X = tf.reshape(sigma_sq_X, [-1, 1, D])
            sigma_sq_Y = tf.reshape(sigma_sq_Y, [1, -1, D])
            sigma_sq_fuse = sigma_sq_X + sigma_sq_Y
            X_ = tf.stop_gradient(X)
            Y_ = tf.stop_gradient(Y)
            # sigma_sq_fuse_ = tf.stop_gradient(sigma_sq_fuse)
            diffs = tf.square(X_-Y_) / (1e-10 + sigma_sq_fuse) + tf.log(sigma_sq_fuse)
            total = tf.reduce_mean(diffs, axis=2)
            # diffs = D * tf.square(X-Y) / (1e-10 + sigma_sq_fuse) + tf.log(sigma_sq_fuse)
            # return tf.reduce_sum(diffs, axis=2)
            attention = tf.reduce_mean(tf.square(X-Y) / (1e-10 + sigma_sq_fuse), axis=2)
            return total, attention, total


def mutual_likelihood_score_loss(labels, mu, log_sigma_sq):
    with tf.name_scope('MLS_Loss'):
        batch_size = tf.shape(mu)[0]

        diag_mask = tf.eye(batch_size, dtype=tf.bool)
        non_diag_mask = tf.logical_not(diag_mask)

        sigma_sq = tf.exp(log_sigma_sq)
        print(mu)
        print(sigma_sq)
        loss_mat, attention_mat, loss_mat_neg = negative_MLS(mu, mu, sigma_sq, sigma_sq)
        
        label_mat = tf.equal(labels[:,None], labels[None,:])
        label_mask_pos = tf.logical_and(non_diag_mask, label_mat)
        label_mask_neg = tf.logical_and(non_diag_mask, tf.logical_not(label_mat))

        loss_mls = tf.boolean_mask(loss_mat, label_mask_pos)
        loss_mls_neg = tf.boolean_mask(loss_mat_neg, label_mask_neg)
        attention_pos = tf.boolean_mask(attention_mat, label_mask_pos)
        attention_neg = tf.boolean_mask(attention_mat, label_mask_neg)
        # loss_attention = tf.reduce_mean(attention_pos) - tf.reduce_mean(attention_neg)

        # metric_type = 'lifted_struct'
        # metric_type = 'binomial'
        # metric_type = 'contrastive'
        # metric_type = 'triplet_semihard'
        mean_pos = tf.reduce_mean(attention_pos)
        mean_neg = tf.reduce_mean(attention_neg)
        loss_mls = tf.reduce_mean(loss_mls)
        # loss_mls += tf.reduce_mean(loss_mls_neg)
        # loss_mls = tf.reduce_sum(loss_mls) / tf.reduce_sum(tf.cast(loss_mls_neg>0, tf.float32))

        return loss_mls, mean_pos, mean_neg


def idq_loss_pairs(X, Y, p_X, p_Y, label_mask_pos, params):
    print('sigma_sq_X.shape[1] =', p_X.shape[1])
    with tf.name_scope('negative_MLS'):
        # cosin distance sigma
        # p_fuse = (p_X + tf.transpose(p_Y)) / 2
        # p_fuse = tf.maximum(p_X, p_Y)
        p_fuse = tf.minimum(p_X, p_Y)
        # p_fuse = tf.minimum(p_X, p_Y) * 0.8 + tf.maximum(p_X, p_Y) * 0.2
        # p_fuse = (p_X + tf.transpose(p_Y)) / 2 * 0.1 + tf.maximum(p_X, p_Y) * 0.9
        # p_fuse = 2*p_X*tf.transpose(p_Y)/(p_X + tf.transpose(p_Y))
        X = tf.stop_gradient(X)
        Y = tf.stop_gradient(Y)
        sigma_sq_fuse_ = tf.stop_gradient(p_fuse)
        # cos_theta = tf.matmul(X_, tf.transpose(Y_))
        cos_theta = tf.matmul(X, tf.transpose(Y))
        cos_theta_pos = tf.boolean_mask(cos_theta, label_mask_pos)
        # diffs = 2*(1-cos_theta) / (1e-10 + 256*sigma_sq_fuse) + tf.log(sigma_sq_fuse)
        # s = 8
        # m = 0.5
        s = params['s']
        m = params['m']
        loss_type = params['loss_type']
        if loss_type == 'idqnet':
            t_soft = tf.sigmoid(s*(cos_theta-m))
        else:
            t_soft = tf.cast(cos_theta>m, tf.float32)  # hard target

        t_soft_pos = tf.boolean_mask(t_soft, label_mask_pos)
        # p_fuse_pos = tf.boolean_mask(p_fuse, label_mask_pos)
        # ce_loss = - t_soft_pos * tf.log(p_fuse_pos) - (1-t_soft_pos) * tf.log(1-p_fuse_pos)
        if loss_type == 'idqnet' or loss_type == 'idqnet-hard' or loss_type == 'idq':
            ce_loss = - t_soft * tf.log(p_fuse) - (1-t_soft) * tf.log(1-p_fuse)
        elif loss_type == 'pcnet':
            # PCNet
            ce_loss = (cos_theta - p_fuse)**2
        elif loss_type == 'pcnet-l1':
            # PCNet-L1
            ce_loss = tf.abs(cos_theta - p_fuse)
        else:
            raise ('error', loss_type)
        ce_loss_pos = tf.boolean_mask(ce_loss, label_mask_pos)

        return ce_loss_pos, cos_theta_pos, t_soft_pos


def soft_idq_loss(labels, mu, confidence, params):
    with tf.name_scope('MLS_Loss'):
        batch_size = tf.shape(mu)[0]

        diag_mask = tf.eye(batch_size, dtype=tf.bool)
        non_diag_mask = tf.logical_not(diag_mask)

        label_mat = tf.equal(labels[:, None], labels[None, :])
        label_mask_pos = tf.logical_and(non_diag_mask, label_mat)
        label_mask_neg = tf.logical_and(non_diag_mask, tf.logical_not(label_mat))

        print(mu)
        print(confidence)
        loss_mls, cos_theta, t_soft = idq_loss_pairs(mu, mu, confidence, confidence, label_mask_pos, params)

        loss_mls = tf.reduce_mean(loss_mls)
        # loss_mls += tf.reduce_mean(loss_mls_neg)
        # loss_mls = tf.reduce_sum(loss_mls) / tf.reduce_sum(tf.cast(loss_mls_neg>0, tf.float32))
        return loss_mls, cos_theta, t_soft
