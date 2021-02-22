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
import pandas as pd

import json

from core.data import DatasetCOCO, show_img

# CONFIG

IMG_SHAPE = (256, 256)

root_data_folder = "./dataset/vinbigdata-coco-dataset-with-wbf-3x-downscaled/vinbigdata-coco-dataset-with-wbf-3x-downscaled"

ds = DatasetCOCO(root_data_folder, img_shape=IMG_SHAPE)



# column_name = ['filename', 'width', 'height',
#                 'class', 'xmin', 'ymin', 'xmax', 'ymax']
# json_df = pd.DataFrame(a, columns=column_name)

# %%

ds.export_csv("./test.csv")



# %%

print(ds)


# %%

df = json_to_csv(ds.test_label)

# %%
cnt = 0

# %%

# See sample
# sample = ds.sample(preprocess=True, shape=IMG_SHAPE)
sample = ds.sample()
# show_img(sample[:, :, 0])

# %%

sample.shape

# %%




