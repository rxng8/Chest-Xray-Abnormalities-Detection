#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
   Author: Alex Nguyen
   Gettysburg College
"""

# %%

import os
import time

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from core.data import Dataset256, show_img

# CONFIG

IMG_SHAPE = (256, 256)

root_data_folder = "./dataset/vinbigdata-256-image-dataset/vinbigdata"

ds = Dataset256(root_data_folder, img_shape=IMG_SHAPE)

# %%

# See sample
sample = ds.sample(preprocess=True, shape=IMG_SHAPE)
show_img(sample[:, :, 0])

# %%





