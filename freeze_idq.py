import os
import sys
import imp
import argparse
import time
import math
import numpy as np

from utils import utils
from utils.imageprocessing import preprocess
from utils.dataset import Dataset
from network_idq import Network


def main(args):
    # Load model files and config file
    network = Network()
    args.batch_size = None
    # args.model_dir = r'log\resface64_mbv3\20210128-150935-s16-m0.4+'
    network.freeze_model(args.model_dir)
    # network.config.preprocess_train = []
    # network.config.preprocess_test = []


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", help="The path to the pre-trained model directory",
                        type=str)
    args = parser.parse_args()
    main(args)
