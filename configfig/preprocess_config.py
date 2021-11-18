''' Config Proto '''

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
