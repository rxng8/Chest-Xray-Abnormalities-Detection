#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
   Author: Alex Nguyen
   Gettysburg College
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import os
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split

from .utils import show_img, preprocess_image


class BaseDataset:
    def __init__(self, root: str, img_shape: Tuple, batch_size: int=16, steps_per_epoch: int=20):
        self.root = root
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.img_shape = img_shape

    def set_batch_size(self, batch_size: int):
        self.batch_size = batch_size

    def set_steps_per_epoch(self, steps_per_epoch: int):
        self.steps_per_epoch = steps_per_epoch

class Dataset256(BaseDataset):
    def __init__(self, root: str, img_shape: Tuple, batch_size: int=16, steps_per_epoch: int=20):
        super().__init__(root, img_shape, batch_size=batch_size, steps_per_epoch=steps_per_epoch)
        self.test_file = os.path.join(root, "test.csv")
        self.train_file = os.path.join(root, "train.csv") 
        self.test_data = os.path.join(root, "test")
        self.train_data = os.path.join(root, "train")
        self.sample_submisison = os.path.join(root, "sample_submission.csv")
    
    