"""Helper for evaluation on the Labeled Faces in the Wild dataset 
"""

# MIT License
# 
# Copyright (c) 2016 David Sandberg
# Copyright (c) 2016 Kaen Chan
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

from easydict import EasyDict

from utils import utils
from utils.imageprocessing import preprocess
from utils.dataset import Dataset
from network_idq import Network

import os
import argparse
import sys
import numpy as np
from scipy import misc
from scipy import interpolate
import sklearn
import cv2
import math
import datetime
import pickle
from sklearn.decomposition import PCA
import mxnet as mx
from mxnet import ndarray as nd
import _pickle as cPickle

from evaluation.pyeer.eer_info import get_eer_stats


def calculate_eer(embeddings1, embeddings2, actual_issame, compare_func, nrof_folds=10):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    dist = compare_func(embeddings1, embeddings2)
    gscores_a = dist[actual_issame == 1]
    iscores_a = dist[actual_issame == 0]
    stats_a = get_eer_stats(gscores_a, iscores_a)
    return stats_a


def evaluate(embeddings, actual_issame, compare_func, nrof_folds=10, keep_idxes=None):
    # Calculate evaluation metrics
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    actual_issame = np.asarray(actual_issame)
    if keep_idxes is not None:
        embeddings1 = embeddings1[keep_idxes]
        embeddings2 = embeddings2[keep_idxes]
        actual_issame = actual_issame[keep_idxes]
    return calculate_eer(embeddings1, embeddings2,
                         actual_issame, compare_func, nrof_folds=nrof_folds)


def load_bin(path, image_size):
    print(path, image_size)
    with open(path, 'rb') as f:
        if 'lfw_all' in path:
            bins, issame_list = pickle.load(f)
        else:
            bins, issame_list = pickle.load(f, encoding='latin1')
    data_list = []
    for flip in [0]:
        data = nd.empty((len(issame_list)*2, image_size[0], image_size[1], 3))
        data_list.append(data)
    print(len(bins))
    for i in range(len(issame_list)*2):
        _bin = bins[i]
        # print(type(_bin))
        img = mx.image.imdecode(_bin)
        # img = nd.transpose(img, axes=(2, 0, 1))
        for flip in [0]:
            if flip==1:
                img = mx.ndarray.flip(data=img, axis=2)
            data_list[flip][i][:] = img
        # if i%5000==0:
        #   print('loading bin', i)
    print(data_list[0].shape)
    return (data_list, issame_list)


def extract_features(images_preprocessed, issame_list, network, batch_size, name='', result_dir='',
                     re_extract_feature=True):
    print('testing verification..')
    if name:
        save_name_pkl_feature = result_dir + '/%s_feature.pkl' % name
    if re_extract_feature or not os.path.exists(save_name_pkl_feature):
        images = images_preprocessed
        print(images.shape)
        mu, sigma_sq = network.extract_feature(images, batch_size, verbose=True)
        save_data = (mu, sigma_sq, issame_list)
        if name:
            with open(save_name_pkl_feature, 'wb') as f:
                cPickle.dump(save_data, f)
            print('save', save_name_pkl_feature)
    else:
        with open(save_name_pkl_feature, 'rb') as f:
            data = cPickle.load(f)
        if len(data) == 3:
            mu, sigma_sq, issame_list = data
        else:
            mu, sigma_sq = data
        print('load', save_name_pkl_feature)
    return mu, sigma_sq, issame_list


def eval_images_with_sigma(mu, sigma_sq, issame_list, nfolds=10, name='', sigma_sizes=1, result_dir=''):
    print('sigma_sq', sigma_sq.shape)
    s = 'sigma_sq ' + str(np.percentile(sigma_sq.ravel(), [0, 10, 30, 50, 70, 90, 100])) + \
        ' percentile [0, 10, 30, 50, 70, 90, 100]\n'
    # print(mu.shape)

    # print('sigma_sq', sigma_sq.shape)
    if sigma_sq.shape[1] == 2:
        sigma_sq_c = np.copy(sigma_sq)
        sigma_sq_list = [sigma_sq_c[:,:1], sigma_sq_c[:,1:]]
    elif type(sigma_sizes) == list:
        sigma_sq_list = []
        idx = 0
        for si in sigma_sizes:
            sigma = sigma_sq[:, idx:idx + si]
            if si > 1:
                sigma = 1/np.mean(1/(sigma+1e-6), axis=-1)
            sigma_sq_list += [sigma]
            idx += si
    elif sigma_sq.shape[1] > 2:
        sigma_sq_list = [1/np.mean(1/(sigma_sq+1e-6), axis=-1)]
    else:
        sigma_sq_list = [sigma_sq]
    for sigma_sq in sigma_sq_list:
        sigma_sq1 = sigma_sq[0::2]
        sigma_sq2 = sigma_sq[1::2]
        sigma_fuse = np.maximum(sigma_sq1, sigma_sq2)
        # reject_factor = 0.1
        error_list = []
        # reject_factors = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        # reject_factors = np.arange(50) / 100.
        # reject_factors = np.arange(30) / 100.
        # reject_factors = [0.0, 0.1, 0.2, 0.3]
        reject_factors_points = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        reject_factors = np.arange(0, 1.0, 0.01)
        fmr100_th_fixed = 0
        fmr1000_th_fixed = 0
        for reject_factor in reject_factors:
            risk_threshold = np.percentile(sigma_fuse.ravel(), (1-reject_factor)*100)
            keep_idxes = np.where(sigma_fuse <= risk_threshold)[0]
            if len(keep_idxes) == 0:
                keep_idxes = None

            stats = evaluate(mu, issame_list, utils.pair_cosin_score, nrof_folds=nfolds, keep_idxes=keep_idxes)
            # get fnmr by fixed recognition threshold T
            if reject_factor == 0:
                fmr100_th_fixed = stats.fmr100_th
                fmr1000_th_fixed = stats.fmr1000_th
                fnmr_fmr1000_fixT = stats.fmr1000
                fnmr_fmr100_fixT = stats.fmr100
            else:
                index = np.argmin(abs(stats.thrs - fmr1000_th_fixed))
                fnmr_fmr1000_fixT = stats.fnmr[index]
                index = np.argmin(abs(stats.thrs - fmr100_th_fixed))
                fnmr_fmr100_fixT = stats.fnmr[index]

            if reject_factor in reject_factors_points:
                s += 'reject_factor {:.4f} '.format(reject_factor)
                s += 'risk_threshold {:.6f} '.format(risk_threshold)
                s += 'keep_idxes {} / {} '.format(len(keep_idxes), len(sigma_fuse))
                s += 'Cosine score eer %f fmr100 %f fmr1000 %f\n' % (stats.eer, fnmr_fmr100_fixT, fnmr_fmr1000_fixT)
            error_list += [fnmr_fmr1000_fixT]
            if keep_idxes is None:
                break
        # s_avg = 'reject_factor 0.5 risk_threshold 0.585041 keep_idxes 3500 / 7000 '
        s_avg = 'reject_factor mean --------------------------------------------- '
        s_avg += 'Cosine score fmr1000 %f\n' % (np.mean(error_list))
        s += s_avg
        tpr = error_list
        fpr = reject_factors
        auerc = sklearn.metrics.auc(fpr, tpr)
        l = int(len(tpr)*0.3)
        auc30 = sklearn.metrics.auc(fpr[:l], tpr[:l])
        s += 'AUERC: %1.4f\n' % auerc
        s += 'AUERC30: %1.4f\n' % auc30
        best = error_list[0]**2/2
        auc = auerc-best
        s += 'AUC: %1.4f\n' % (auerc-best)
        best30 = (error_list[0] * min(error_list[0], 0.3))/2
        s += 'AUC30: %1.4f\n' % (auc30-best30)
        s += '\n'
        print(s)
        if name != '':
            save_name_pkl_fnmr = result_dir + '/fixT-fnmr-{}.pkl'.format(name)
            save_data = (reject_factors, error_list)
            if name:
                with open(save_name_pkl_fnmr, 'wb') as f:
                    cPickle.dump(save_data, f)
                print('save', save_name_pkl_fnmr)
    # print(s)
    return s[:-1], auc


def eval_images(images_preprocessed, issame_list, network, batch_size, nfolds=10, name='', result_dir='',
                re_extract_feature=True):
    mu, sigma_sq, issame_list = extract_features(images_preprocessed, issame_list, network, batch_size,
                                                 name=name, result_dir=result_dir,
                                                 re_extract_feature=re_extract_feature)
    s_reject, auc = eval_images_with_sigma(mu, sigma_sq, issame_list, nfolds=nfolds, name='')
    return s_reject, auc


def eval(data_set, network, batch_size, nfolds=10, name='', result_dir='', re_extract_feature=True):
  print('testing verification..')
  data_list = data_set[0]
  issame_list = data_set[1]
  data_list = data_list[0].asnumpy()
  images = preprocess(data_list, network.config, False)
  del data_set
  for i in range(1):
      # name1 = name + '_keep0.9_%03d' % i
      name1 = name
      ret, _ = eval_images(images, issame_list, network, batch_size, nfolds=10, name=name1, result_dir=result_dir,
                         re_extract_feature=re_extract_feature)
      print(ret)
      # ret = eval_images_cmp(images, issame_list, network, batch_size, nfolds=10, name=name, result_dir=result_dir,
      #                       re_extract_feature=re_extract_feature)
  return ret


def main(args):
    data_dir = args.dataset_path
    # data_dir = r'F:\data\face-recognition\MS-Celeb-1M\faces_emore'
    # data_dir = r'F:\data\face-recognition\trillion-pairs\challenge\ms1m-retinaface-t1'

    re_extract_feature = False

    noise_dataset_name = ''
    # noise_dataset_name = 'cutout-r0.5n2-p0.05'
    if noise_dataset_name != '':
        data_dir = os.path.join(data_dir, noise_dataset_name)

    network = Network()
    network.load_model(args.model_dir)

    # # images = np.random.random([1, 128, 128, 3])
    # images = np.random.random([1, 96, 96, 3])
    # for _ in range(5):
    #     mu, sigma_sq = network.extract_feature(images, 1, verbose=True)
    #     print(mu[0, :5])
    # exit(0)

    for namec in args.target.split(','):
        path = os.path.join(data_dir,namec+".bin")
        if os.path.exists(path):
            image_size = [112, 112]
            data_set = load_bin(path, image_size)
            name = namec
            print('ver', name)
            info = eval(data_set, network, args.batch_size, 10, name=name, result_dir=args.model_dir,
                        re_extract_feature=re_extract_feature)
            # print(info)
            info_result = '--- ' + name + ' ---\n'
            info_result += info + "\n"
            print("")
            print(info_result)
            log_path = os.path.join(
                args.model_dir,
                'testing-log-fnmr-{}.txt'.format(name))
            if noise_dataset_name != '':
                log_dir = os.path.join(args.model_dir, noise_dataset_name)
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                log_path = os.path.join(
                    log_dir,
                    'testing-log-fnmr-{}.txt'.format(name, noise_dataset_name))
            with open(log_path, 'a') as f:
                f.write(info_result + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", help="The path to the pre-trained model directory",
                        type=str,
                        default=r'')
    parser.add_argument("--dataset_path", help="The path to the LFW dataset directory",
                        type=str, default=r'F:\data\face-recognition\trillion-pairs\challenge\ms1m-retinaface-t1')
    parser.add_argument("--batch_size", help="Number of images per mini batch",
                        type=int, default=16)
    parser.add_argument('--target', type=str, default='lfw,cfp_fp,agedb_30', help='verification targets')
    args = parser.parse_args()
    # args.target = 'lfw,calfw,cplfw,cfp_ff,cfp_fp,agedb_30,vgg2_fp'
    # args.target = 'cfp_fp,agedb_30'
    # args.target = 'lfw,calfw,cplfw,cfp_ff,vgg2_fp'
    # args.target = 'lfw,sllfw,calfw,cplfw,cfp_ff,cfp_fp,agedb_30,vgg2_fp'
    # args.target = 'sllfw'
    args.dataset_path = r'F:\data\metric-learning\face\ms1m-retinaface-t1'
    # args.dataset_path = r'F:\data\face-recognition\trillion-pairs\challenge\ms1m-retinaface-t1'
    args.target = 'cfp_fp'
    args.model_dir = r'E:\chenkai\lightqnet-train\log\resface64_mbv3\20210128-150935-s16-m0.4+'
    main(args)
