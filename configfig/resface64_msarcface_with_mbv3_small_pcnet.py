''' Config Proto '''

import sys
import os


####### INPUT OUTPUT #######

# The name of the current model for output
name = 'resface64_mbv3'

# The folder to save log and model
log_base_dir = './log/'

# The interval between writing summary
summary_interval = 100

# Training dataset path
train_dataset_path = r"F:\data\metric-learning\face\ms1m-retinaface-t1-img"

# Target image size for the input of network
image_size = [96, 96]

# 3 channels means RGB, 1 channel for grayscale
channels = 3

# Preprocess for training
preprocess_train = [
    # ['center_crop', (image_size[0], image_size[1])],
    ['resize', (image_size[0], image_size[1])],
    ['random_flip'],
    ['standardize', 'mean_scale'],
]

# Preprocess for testing
preprocess_test = [
    ['resize', (image_size[0], image_size[1])],
    ['standardize', 'mean_scale'],
]

####### NETWORK #######

# The network architecture
embedding_network = "models/resface64_relu.py"

# The network architecture
uncertainty_module = "models/uncertainty_module_confidence.py"
uncertainty_module_input = "conv_final"

uncertainty_module_student = "models/mobilenet/mobilenet_v3_small_mini_face_dm100_op1s1_confidence.py"

# Number of dimensions in the embedding space
embedding_size = 256

# uncertainty_module_output_size = embedding_size
uncertainty_module_output_size = 1


####### TRAINING STRATEGY #######

# Base Random Seed
base_random_seed = 11

# Number of samples per batch
batch_size = 100
samples_per_class = 2
batch_format = { 'size': batch_size,
    'num_classes': batch_size // samples_per_class,
    # 'num_classes': 32,
}

# Number of batches per epoch
epoch_size = 1000

# Number of epochs
# num_epochs = 64
num_epochs = 12

# learning rate strategy
learning_rate_strategy = 'step'

# learning rate schedule
# lr = 3e-2
lr = 1e-2
learning_rate_schedule = {
    0:      1 * lr,
    num_epochs/4*2000:   0.1 * lr,
    num_epochs/4*3000:   0.01 * lr,
}

# Restore model
restore_model = r'log\resface-arc\20191211-1058-resface64-96-arc-aug0.01-retina-99.80-99.50-mom'

# Keywords to filter restore variables, set None for all
# restore_scopes = ['Resface', 'UncertaintyModule']
restore_scopes = None
exclude_restore_scopes = ['UncertaintyModule']

# Weight decay for model variables
weight_decay = 1e-5

# Keep probability for dropouts
keep_prob = 1.0

loss_weights = {
    'uncertainty_loss': 1.0,
    'student_distilling': 20.,
}

t_soft_s = 16
t_soft_m = 0.4
t_soft_loss_type = 'pcnet'

idq_tau = 1
