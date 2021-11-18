"""Main implementation class of PFE
"""
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

import os
import sys
import imp
import time

import numpy as np
import tensorflow as tf

from utils.tflib import soft_idq_loss, mutual_likelihood_score_loss


class Network:
    def __init__(self):
        self.graph = tf.Graph()
        gpu_options = tf.GPUOptions(allow_growth=True)
        tf_config = tf.ConfigProto(gpu_options=gpu_options,
                allow_soft_placement=True, log_device_placement=False)
        self.sess = tf.Session(graph=self.graph, config=tf_config)
            
    def initialize(self, config, num_classes=None, for_freeze=False):
        '''
            Initialize the graph from scratch according to config.
        '''
        self.config = config
        if 'loss_weights' not in dir(config):
            self.weight_uncertainty_loss = 1.0
            self.weight_student_distilling = 0.0
        else:
            self.weight_uncertainty_loss = config.loss_weights['uncertainty_loss']
            self.weight_student_distilling = config.loss_weights['student_distilling']

        self.use_mls_loss = self.weight_uncertainty_loss > 0
        self.use_student_distilling = self.weight_student_distilling > 0

        with self.graph.as_default():
            with self.sess.as_default():
                # Set up placeholders
                h, w = config.image_size
                channels = config.channels

                if for_freeze:
                    self.images = tf.placeholder(tf.float32, shape=[None, h, w, channels], name='input')
                    self.labels = None
                    self.phase_train = False
                    self.learning_rate = None
                    self.keep_prob = 1.
                    self.global_step = 0
                else:
                    self.images = tf.placeholder(tf.float32, shape=[None, h, w, channels], name='images')
                    self.labels = tf.placeholder(tf.int32, shape=[None], name='labels')
                    self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
                    self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
                    self.phase_train = tf.placeholder(tf.bool, name='phase_train')
                    self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32, name='global_step')

                # Initialialize the backbone network
                network = imp.load_source('embedding_network', config.embedding_network)
                # mu, conv_final = network.inference(self.images, config.embedding_size, keep_probability=self.keep_prob, phase_train=self.phase_train)
                mu, conv_final = network.inference(self.images, config.embedding_size)

                # Initialize the uncertainty module
                uncertainty_module = imp.load_source('uncertainty_module', config.uncertainty_module)
                if 'uncertainty_module_input' not in dir(config):
                    uncertainty_module_input = conv_final
                elif config.uncertainty_module_input == "images":
                    uncertainty_module_input = self.images
                else:
                    uncertainty_module_input = conv_final

                if 'uncertainty_module_output_size' in dir(config):
                    uncertainty_module_output_size = config.uncertainty_module_output_size
                else:
                    uncertainty_module_output_size = config.embedding_size
                    uncertainty_module_output_size = 1
                print('uncertainty_module_output_size', uncertainty_module_output_size)
                scoepname = 'UncertaintyModuleTeacher'
                confidence, logit = uncertainty_module.inference(uncertainty_module_input, uncertainty_module_output_size,
                                        phase_train = self.phase_train, weight_decay = config.weight_decay,
                                        scope=scoepname)

                self.mu = tf.identity(mu, name='mu')
                self.confidence = tf.identity(confidence, name='confidence')
                self.sigma_sq = tf.identity(1-confidence, name='sigma_sq')

                if self.use_student_distilling:
                    # uncertainty_module_name = "models/uncertainty_res12_avg.py"
                    # uncertainty_module_name = "models/mobilenet/mobilenet_v3_mini_face_dm100_op1s1.py"
                    uncertainty_module_name = config.uncertainty_module_student
                    if 'image_size_student' in dir(config) and config.image_size_student[0] != config.image_size[0]:
                        images_st = tf.image.resize_images(self.images, config.image_size_student)
                    else:
                        images_st = self.images
                    uncertainty_module = imp.load_source('uncertainty_module', uncertainty_module_name)
                    confidence_st, logit_st = uncertainty_module.inference(images_st, uncertainty_module_output_size,
                                                                phase_train = self.phase_train, weight_decay = config.weight_decay,
                                                                scope='UncertaintyModule')
                    self.confidence_st = tf.identity(confidence_st, name='confidence_st')
                    self.sigma_sq_st = tf.identity(1 - confidence_st, name='sigma_sq_st')
                else:
                    self.sigma_sq_st = None

                if for_freeze:
                    # res = tf.concat(values=(self.mu, self.sigma_sq), axis=1, name='embeddings_with_sigma')
                    return

                # Build all losses
                loss_list = []
                self.watch_list = {}

                # sigma_sq_wd = tf.reduce_mean(self.sigma_sq) * 10
                # loss_list.append(sigma_sq_wd)
                # self.watch_list['s_wd'] = sigma_sq_wd

                params = {'s': config.t_soft_s, 'm': config.t_soft_m, 'loss_type': config.t_soft_loss_type}
                idq_loss, cos_theta, t_soft = soft_idq_loss(self.labels, mu, confidence, params)
                if self.use_mls_loss:
                    loss_list.append(self.weight_uncertainty_loss * idq_loss)
                    self.watch_list['idq'] = idq_loss
                    self.watch_list['cos'] = cos_theta
                    self.watch_list['t_soft'] = t_soft

                self.watch_list['p'] = confidence

                #st
                if self.use_student_distilling:
                    # MLS_loss_st = mutual_likelihood_score_loss(self.labels, mu, log_sigma_sq_student)
                    idq_loss_st, cos_theta_st, t_soft_st = soft_idq_loss(self.labels, mu, confidence_st, params)
                    loss_list.append(self.weight_uncertainty_loss * idq_loss_st)
                    self.watch_list['idq_st'] = idq_loss_st

                    def kl_for_log_probs(p, q):
                        log_p = tf.log(p)
                        log_q = tf.log(q)
                        p = tf.exp(log_p)
                        neg_ent = p * tf.log(p) + (1-p) * tf.log(1-p)
                        neg_cross_ent = p * tf.log(q) + (1-p) * tf.log(1-q)
                        kl = neg_ent - neg_cross_ent
                        kl = tf.reduce_mean(kl)
                        return kl
                    # print(id_quality_stop)
                    # print(id_quality_student)
                    logit_stop = tf.stop_gradient(logit)
                    p = tf.nn.sigmoid(logit_stop/config.idq_tau)
                    q = tf.nn.sigmoid(logit_st/config.idq_tau)
                    kl_loss = kl_for_log_probs(p, q) * (config.idq_tau**2)
                    kl_loss = kl_loss * self.weight_student_distilling
                    # print(kl_loss)
                    # exit(0)
                    loss_list.append(kl_loss)
                    self.watch_list['kl_loss_st'] = kl_loss
                    self.watch_list['p_st'] = confidence_st

                # Collect all losses
                reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='reg_loss')
                loss_list.append(reg_loss)
                self.watch_list['regu'] = reg_loss

                total_loss = tf.add_n(loss_list, name='total_loss')
                self.watch_list['a'] = total_loss
                grads = tf.gradients(total_loss, self.trainable_variables)

                # Training Operaters
                train_ops = []

                opt = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9)
                # opt = tf.train.AdamOptimizer(self.learning_rate)
                # opt = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-1)
                apply_gradient_op = opt.apply_gradients(list(zip(grads, self.trainable_variables)))

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                train_ops.extend([apply_gradient_op] + update_ops)

                train_ops.append(tf.assign_add(self.global_step, 1))
                self.train_op = tf.group(*train_ops)

                # Initialize variables
                self.sess.run(tf.local_variables_initializer())
                self.sess.run(tf.global_variables_initializer())
                self.saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=1)

    @property
    def trainable_variables(self):
        trainable_scopes = 'UncertaintyModule,StudentModule'
        trainable_scopes = trainable_scopes.split(',')
        variables_to_train = []
        for scope in trainable_scopes:
            variables_to_train += [k for k in tf.global_variables() if k.op.name.startswith(scope)]
        print('variables_to_train', len(variables_to_train))
        return variables_to_train

    def save_model(self, model_dir, global_step):
        with self.sess.graph.as_default():
            checkpoint_path = os.path.join(model_dir, 'ckpt')
            metagraph_path = os.path.join(model_dir, 'graph.meta')

            print('Saving variables...', model_dir, global_step)
            self.saver.save(self.sess, checkpoint_path, global_step=global_step, write_meta_graph=False)
            if not os.path.exists(metagraph_path):
                print('Saving metagraph...')
                self.saver.export_meta_graph(metagraph_path)

    def get_model_filenames(self, model_dir):
        files = os.listdir(model_dir)
        meta_files = [s for s in files if s.endswith('.meta')]
        if len(meta_files) == 0:
            raise ValueError('No meta file found in the model directory (%s)' % model_dir)
        elif len(meta_files) > 1:
            raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
        meta_file = meta_files[0]
        meta_files = [s for s in files if '.ckpt' in s]
        max_step = -1
        ckpt_file = None
        import re
        for f in files:
            step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
            if step_str is not None and len(step_str.groups()) >= 2:
                step = int(step_str.groups()[1])
                if step > max_step:
                    max_step = step
                    ckpt_file = step_str.groups()[0]
        if ckpt_file is None:
            for f in files:
                if 'index' in f:
                    ckpt_file = f[:-len('.index')]
        meta_file = os.path.join(model_dir, meta_file)
        ckpt_file = os.path.join(model_dir, ckpt_file)
        return meta_file, ckpt_file

    def restore_model(self, model_dir, restore_scopes=None, exclude_restore_scopes=None):
        with self.sess.graph.as_default():
            # var_list = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            var_list = tf.trainable_variables()
            print(restore_scopes)
            if exclude_restore_scopes is not None:
                for exclude_restore_scope in exclude_restore_scopes:
                    var_list = [var for var in var_list  if exclude_restore_scope not in var.op.name]
            print(len(var_list))
            meta_file, ckpt_file = self.get_model_filenames(model_dir)

            print('Restoring {} variables from {} ...'.format(len(var_list), ckpt_file))
            saver = tf.train.Saver(var_list)
            saver.restore(self.sess, ckpt_file)

    def load_model(self, model_path, scope=None):
        with self.sess.graph.as_default():
            model_path = os.path.expanduser(model_path)

            # Load grapha and variables separatedly.
            meta_files = [file for file in os.listdir(model_path) if file.endswith('.meta')]
            assert len(meta_files) == 1
            meta_file = os.path.join(model_path, meta_files[0])
            ckpt_file = tf.train.latest_checkpoint(model_path)
            
            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)
            saver = tf.train.import_meta_graph(meta_file, clear_devices=True, import_scope=scope)
            saver.restore(self.sess, ckpt_file)

            # Setup the I/O Tensors
            try:
                self.images = self.graph.get_tensor_by_name('images:0')
                self.mu = self.graph.get_tensor_by_name('mu:0')
                self.sigma_sq = self.graph.get_tensor_by_name('sigma_sq:0')
            except:
                self.images = self.graph.get_tensor_by_name('input:0')
                self.mu = self.graph.get_tensor_by_name('embeddings:0')
                self.sigma_sq = self.graph.get_tensor_by_name('sigma_sq:0')
            try:
                self.sigma_sq_st = self.graph.get_tensor_by_name('sigma_sq_st:0')
                self.use_student_distilling = True
            except:
                self.sigma_sq_st = None
                self.use_student_distilling = False

            self.phase_train = self.graph.get_tensor_by_name('phase_train:0')
            try:
                self.keep_prob = self.graph.get_tensor_by_name('keep_prob:0')
            except:
                print('no keep_prob in load model')
                exit(0)
                self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.config = imp.load_source('network_config', os.path.join(model_path, 'config.py'))
            print('mu', self.mu.get_shape())
            print('sigma_sq', self.sigma_sq.get_shape())

    def train(self, images_batch, labels_batch, learning_rate, keep_prob):
        feed_dict = {   self.images: images_batch,
                        self.labels: labels_batch,
                        self.learning_rate: learning_rate,
                        self.keep_prob: keep_prob,
                        self.phase_train: True,}
        _, wl = self.sess.run([self.train_op, self.watch_list], feed_dict = feed_dict)

        step = self.sess.run(self.global_step)

        return wl, step

    def extract_feature(self, images, batch_size, proc_func=None, verbose=False):
        num_images = len(images)
        num_features = self.mu.shape[1]
        mu = np.ndarray((num_images, num_features), dtype=np.float32)
        num_features_sq = self.sigma_sq.shape[1]
        if 'use_student_distilling' in dir(self) and self.use_student_distilling:
            sigma_sq = np.ndarray((num_images, 2), dtype=np.float32)
        else:
            sigma_sq = np.ndarray((num_images, num_features_sq), dtype=np.float32)
        start_time = time.time()
        for start_idx in range(0, num_images, batch_size):
            if verbose:
                elapsed_time = time.strftime('%H:%M:%S', time.gmtime(time.time()-start_time))
                sys.stdout.write('# of images: %d Current image: %d Elapsed time: %s \t\r' 
                    % (num_images, start_idx, elapsed_time))
            end_idx = min(num_images, start_idx + batch_size)
            images_batch = images[start_idx:end_idx]
            if proc_func:
                images_batch = proc_func(images_batch)
            feed_dict = {self.images: images_batch,
                        self.phase_train: False,
                    self.keep_prob: 1.0}
            if 'use_student_distilling' in dir(self) and self.use_student_distilling:
                # mu[start_idx:end_idx], sigma_sq[start_idx:end_idx] = self.sess.run([self.mu, self.sigma_sq_st], feed_dict=feed_dict)
                mu[start_idx:end_idx], sigma_sq1, sigma_sq2 = self.sess.run([self.mu, self.sigma_sq, self.sigma_sq_st], feed_dict=feed_dict)
                sigma_sq[start_idx:end_idx] = np.concatenate([sigma_sq1, sigma_sq2], axis=-1)
            else:
                mu[start_idx:end_idx], sigma_sq[start_idx:end_idx] = self.sess.run([self.mu, self.sigma_sq], feed_dict=feed_dict)
            # lprint(mu[0, :10])
            # print(sigma_sq[0, :10])
            # exit(0)
        if verbose:
            print('')
        return mu, sigma_sq

    def freeze_model(self, model_dir):
        self.config = imp.load_source('network_config', os.path.join(model_dir, 'config.py'))
        self.initialize(self.config, for_freeze=True)
        with self.sess.graph.as_default():
            var_list = tf.trainable_variables()
            print(len(var_list))
            # model_dir = os.path.expanduser(model_dir)
            # ckpt_file = tf.train.latest_checkpoint(model_dir)
            meta_file, ckpt_file = self.get_model_filenames(model_dir)
            print('Restoring {} variables from {} ...'.format(len(var_list), ckpt_file))
            saver = tf.train.Saver(var_list)
            saver.restore(self.sess, ckpt_file)

            # Retrieve the protobuf graph definition and fix the batch norm nodes
            gd = self.sess.graph.as_graph_def()
            for node in gd.node:
                if node.op == 'RefSwitch':
                    node.op = 'Switch'
                    for index in range(len(node.input)):
                        if 'moving_' in node.input[index]:
                            node.input[index] = node.input[index] + '/read'
                elif node.op == 'AssignSub':
                    node.op = 'Sub'
                    if 'use_locking' in node.attr: del node.attr['use_locking']
                elif node.op == 'AssignAdd':
                    node.op = 'Add'
                    if 'use_locking' in node.attr: del node.attr['use_locking']

            # Get the list of important nodes
            # output_node_names = 'embeddings_with_sigma'
            # output_node_names = 'mu,sigma_sq'
            # output_node_names = 'sigma_sq_st'
            output_node_names = 'confidence_st'
            whitelist_names = []
            for node in gd.node:
                # if node.name.startswith('InceptionResnetV1') or node.name.startswith('embeddings') or node.name.startswith('phase_train') or node.name.startswith('Bottleneck'):
                print(node.name)
                if not node.name.startswith('Logits'):
                    whitelist_names.append(node.name)

            from tensorflow.python.framework import graph_util
            # Replace all the variables in the graph with constants of the same values
            output_graph_def = graph_util.convert_variables_to_constants(
                self.sess, gd, output_node_names.split(","),
                variable_names_whitelist=whitelist_names)

            # Serialize and dump the output graph to the filesystem
            output_file = os.path.join(model_dir, 'freeze_model.pb')
            with tf.gfile.GFile(output_file, 'wb') as f:
                f.write(output_graph_def.SerializeToString())
            print("%d ops in the final graph." % len(output_graph_def.node))
            print(model_dir)
            print(output_file)
