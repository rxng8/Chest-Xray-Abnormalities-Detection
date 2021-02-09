#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
   Author: Alex Nguyen
   Gettysburg College
"""

import sys
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import math
import os
import re

import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import cv2

def k_fold():
    """
    Reference to here:
        https://www.kaggle.com/backtracking/smart-data-split-train-eval-for-object-detection/comments
    """
    pass

def preprocess_image(
        src_img: np.ndarray, 
        shape,
        resize_method=tf.image.ResizeMethod.BILINEAR,
        range='tanh'
    ) -> tf.Tensor:
    # Expect image value range 0 - 255

    img = src_img
    if len(src_img.shape) == 2:
        img = tf.expand_dims(src_img, axis=-1)

    resized = tf.image.resize(
        img, 
        shape,
        method=resize_method
    )

    rescaled = None
    if range == 'tanh':
        rescaled = tf.cast(resized, dtype=float) / 255.0
        rescaled = (rescaled - 0.5) * 2 # range [-1, 1]
    elif range == 'sigmoid':
        rescaled = tf.cast(resized, dtype=float) / 255.0
    elif range == None:
        rescaled = tf.cast(resized, dtype=float)
    else:
        print("Wrong type!")
        sys.exit(1)

    # Convert to BGR
    bgr = rescaled[..., ::-1]
    return bgr

def deprocess_img(img):
    # Expect img range [-1, 1]
    # Do the rescale back to 0, 1 range, and convert from bgr back to rgb
    return (img / 2.0 + 0.5)[..., ::-1]

def show_img(img):
    if len(img.shape) == 3:
        plt.figure()
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    elif len(img.shape) == 2:
        plt.figure()
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.show()

def load_image(path):
    return np.asarray(Image.open(path))
