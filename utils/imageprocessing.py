"""Functions for image processing
"""
# MIT License
# 
# Copyright (c) 2017 Yichun Shi
# Copyright (c) 2020 Kaen Chan
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

import sys
import os
import math
import random
import numpy as np
from scipy import misc
import cv2
from PIL import Image, ImageFilter

# Calulate the shape for creating new array given (h,w)
def get_new_shape(images, size=None, n=None):
    shape = list(images.shape)
    if size is not None:
        h, w = tuple(size)
        shape[1] = h
        shape[2] = w
    if n is not None:
        shape[0] = n
    shape = tuple(shape)
    return shape

def random_crop(images, size):
    n, _h, _w = images.shape[:3]
    h, w = tuple(size)
    shape_new = get_new_shape(images, size)
    assert (_h>=h and _w>=w)

    images_new = np.ndarray(shape_new, dtype=images.dtype)

    y = np.random.randint(low=0, high=_h-h+1, size=(n))
    x = np.random.randint(low=0, high=_w-w+1, size=(n))

    for i in range(n):
        images_new[i] = images[i, y[i]:y[i]+h, x[i]:x[i]+w]

    return images_new

def center_crop(images, size):
    n, _h, _w = images.shape[:3]
    h, w = tuple(size)
    assert (_h>=h and _w>=w)

    y = int(round(0.5 * (_h - h)))
    x = int(round(0.5 * (_w - w)))

    images_new = images[:, y:y+h, x:x+w]

    return images_new

def random_flip(images):
    images_new = images.copy()
    flips = np.random.rand(images_new.shape[0])>=0.5
    
    for i in range(images_new.shape[0]):
        if flips[i]:
            images_new[i] = np.fliplr(images[i])

    return images_new

def flip(images):
    images_new = images.copy()
    for i in range(images_new.shape[0]):
        images_new[i] = np.fliplr(images[i])

    return images_new

def resize(images, size):
    n, _h, _w = images.shape[:3]
    h, w = tuple(size)
    shape_new = get_new_shape(images, size)

    images_new = np.ndarray(shape_new, dtype=images.dtype)

    for i in range(n):
        images_new[i] = misc.imresize(images[i], (h,w))

    return images_new

def padding(images, padding):
    n, _h, _w = images.shape[:3]
    if len(padding) == 2:
        pad_t = pad_b = padding[0]
        pad_l = pad_r = padding[1]
    else:
        pad_t, pad_b, pad_l, pad_r = tuple(padding)
       
    size_new = (_h + pad_t + pad_b, _w + pad_l + pad_b)
    shape_new = get_new_shape(images, size_new)
    images_new = np.zeros(shape_new, dtype=images.dtype)
    images_new[:, pad_t:pad_t+_h, pad_l:pad_l+_w] = images

    return images_new

def standardize_images(images, standard):
    if standard=='mean_scale':
        mean = 128.0
        std = 128.0
    elif standard=='scale':
        mean = 0.0
        std = 255.0
    # images_new = images.astype(np.float32)
    # images_new = images.copy()
    images_new = images
    images_new = (images_new - mean) / std
    return images_new



def random_shift(images, max_ratio):
    n, _h, _w = images.shape[:3]
    pad_x = int(_w * max_ratio) + 1
    pad_y = int(_h * max_ratio) + 1
    images_temp = padding(images, (pad_x, pad_y))
    images_new = images.copy()

    shift_x = (_w * max_ratio * np.random.rand(n)).astype(np.int32)
    shift_y = (_h * max_ratio * np.random.rand(n)).astype(np.int32)

    for i in range(n):
        images_new[i] = images_temp[i, pad_y+shift_y[i]:pad_y+shift_y[i]+_h, 
                            pad_x+shift_x[i]:pad_x+shift_x[i]+_w]

    return images_new    
    

def random_downsample(images, min_ratio):
    n, _h, _w = images.shape[:3]
    images_new = images.copy()
    ratios = min_ratio + (1-min_ratio) * np.random.rand(n)

    for i in range(n):
        w = int(round(ratios[i] * _w))
        h = int(round(ratios[i] * _h))
        images_new[i,:h,:w] = misc.imresize(images[i], (h,w))
        images_new[i] = misc.imresize(images_new[i,:h,:w], (_h,_w))
        
    return images_new

def random_interpolate(images):
    _n, _h, _w = images.shape[:3]
    nd = images.ndim - 1
    assert _n % 2 == 0
    n = int(_n / 2)

    ratios = np.random.rand(n,*([1]*nd))
    images_left, images_right = (images[np.arange(n)*2], images[np.arange(n)*2+1])
    images_new = ratios * images_left + (1-ratios) * images_right
    images_new = images_new.astype(np.uint8)

    return images_new
    
def expand_flip(images):
    '''Flip each image in the array and insert it after the original image.'''
    _n, _h, _w = images.shape[:3]
    shape_new = get_new_shape(images, n=2*_n)
    images_new = np.stack([images, flip(images)], axis=1)
    images_new = images_new.reshape(shape_new)
    return images_new

def five_crop(images, size):
    _n, _h, _w = images.shape[:3]
    h, w = tuple(size)
    assert h <= _h and w <= _w

    shape_new = get_new_shape(images, size, n=5*_n)
    images_new = []
    images_new.append(images[:,:h,:w])
    images_new.append(images[:,:h,-w:])
    images_new.append(images[:,-h:,:w])
    images_new.append(images[:,-h:,-w:])
    images_new.append(center_crop(images, size))
    images_new = np.stack(images_new, axis=1).reshape(shape_new)
    return images_new

def ten_crop(images, size):
    _n, _h, _w = images.shape[:3]
    shape_new = get_new_shape(images, size, n=10*_n)
    images_ = five_crop(images, size)
    images_flip_ = five_crop(flip(images), size)
    images_new = np.stack([images_, images_flip_], axis=1)
    images_new = images_new.reshape(shape_new)
    return images_new

def cutout(img, length_ratio=0.3, n_holes=1):
    """
    Args:
        img (Tensor): Tensor image of size (C, H, W).
    Returns:
        Tensor: Image with n_holes of dimension length x length cut out of it.
    """
    h = img.shape[0]
    w = img.shape[1]
    length = int(h*length_ratio)
    mask = np.ones((h, w), np.float32)
    for n in range(n_holes):
        y = np.random.randint(h)
        x = np.random.randint(w)
        # y = np.random.randint(15, h - length + 1 - 10)
        # x = np.random.randint(10, w - length + 1 - 10)
        y1 = np.clip(y - length // 2, 0, h)
        y2 = np.clip(y + length // 2, 0, h)
        x1 = np.clip(x - length // 2, 0, w)
        x2 = np.clip(x + length // 2, 0, w)
        mask[y1: y2, x1: x2] = 0.
    img = img * np.expand_dims(mask, -1)
    return img

def rotate(img, level=3):
    pil_img = Image.fromarray(img.astype(np.uint8))  # Convert to PIL.Image
    degrees = int(level * 30 / 10)
    # print('rotate', degrees)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    pil_img = pil_img.rotate(degrees, resample=Image.BILINEAR)
    img = np.asarray(pil_img)
    return img.astype(np.uint8)

def gaussian_blur(img, radius=3.0):
    pil_img = Image.fromarray(img.astype(np.uint8))  # Convert to PIL.Image
    pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=radius))
    img = np.asarray(pil_img)
    return img.astype(np.uint8)

def cutout_images(images, length_ratio=0.3, n_holes=1):
    for i in range(len(images)):
        images[i] = cutout(images[i], length_ratio, n_holes)
    return images

def rotate_images(images, level=3, p=0.5):
    for i in range(len(images)):
        if np.random.random() < p:
            images[i] = rotate(images[i], level)
    return images

def gaussian_blur_images(images, radius=5.0, p=0.5):
    for i in range(len(images)):
        if np.random.random() < p:
            images[i] = gaussian_blur(images[i], radius)
    return images

register = {
    'resize': resize,
    'padding': padding,
    'random_crop': random_crop,
    'center_crop': center_crop,
    'random_flip': random_flip,
    'standardize': standardize_images,
    'random_shift': random_shift,
    'random_interpolate': random_interpolate,
    'random_downsample': random_downsample,
    'expand_flip': expand_flip,
    'five_crop': five_crop,
    'ten_crop': ten_crop,
    'cutout': cutout_images,
    'rotate': rotate_images,
    'gaussian_blur': gaussian_blur,
}

def preprocess(images, config, is_training=False):
    # Load images first if they are file paths
    if type(images[0]) == str:
        image_paths = images
        images = []
        assert (config.channels==1 or config.channels==3)
        mode = 'RGB' if config.channels==3 else 'I'
        for image_path in image_paths:
            # images.append(misc.imread(image_path, mode=mode))
            img = misc.imread(image_path, mode='RGB')
            # img = cv2.resize(img, (112, 112))
            images.append(img)
        images = np.stack(images, axis=0)
    else:
        assert type(images) == np.ndarray
        assert images.ndim == 4

    # Process images
    proc_funcs = config.preprocess_train if is_training else config.preprocess_test
    # print(proc_funcs)

    images = images.copy()
    images_noaug = images.copy()

    for proc in proc_funcs:
        proc_name, proc_args = proc[0], proc[1:]
        assert proc_name in register, \
            "Not a registered preprocessing function: {}".format(proc_name)
        images = register[proc_name](images, *proc_args)
        # print(proc_name, images.shape)
    # if is_training:
    #     for proc in config.preprocess_test:
    #         proc_name, proc_args = proc[0], proc[1:]
    #         assert proc_name in register, \
    #             "Not a registered preprocessing function: {}".format(proc_name)
    #         images_noaug = register[proc_name](images_noaug, *proc_args)
    #     images = np.concatenate([images, images_noaug], axis=0)

    if len(images.shape) == 3:
        images = images[:,:,:,None]
    return images
        

