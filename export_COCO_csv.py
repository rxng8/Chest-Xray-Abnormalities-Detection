#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
   Author: Alex Nguyen
   Gettysburg College
"""

import os
import argparse

from core.data import DatasetCOCO

# CONFIG

IMG_SHAPE = (256, 256)

if __name__ == '__main__':
   root_data_folder = "./dataset/vinbigdata-coco-dataset-with-wbf-3x-downscaled/vinbigdata-coco-dataset-with-wbf-3x-downscaled"
   ds = DatasetCOCO(root_data_folder, img_shape=IMG_SHAPE)
   if not os.path.exists("./tmp"):
      os.mkdir("./tmp/")
   train_df, val_df = ds.export_csv("./tmp/train.csv", "./tmp/val.csv")
   print("Exported! Head(5):", train_df.head())