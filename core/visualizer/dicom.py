#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
   Author: Alex Nguyen
   Gettysburg College
"""

import matplotlib.pyplot as plt
from pydicom.dataset import FileDataset
import numpy as np

def show_dcm_info(dataset: FileDataset):
    print("Patient's Gender :", dataset.PatientSex)

    if 'PixelData' in dataset:
        rows = int(dataset.Rows)
        cols = int(dataset.Columns)
        print("Image size : {rows:d} x {cols:d}, {size:d} bytes".format(
            rows=rows, cols=cols, size=len(dataset.PixelData)))
        if 'PixelSpacing' in dataset:
            print("Pixel spacing :", dataset.PixelSpacing)

    

def plot_pixel_array(dataset: FileDataset, figsize=(10,10)):
    plt.figure(figsize=figsize)
    plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)
    plt.show()

def get_array(dataset: FileDataset) -> np.ndarray:
    return dataset.pixel_array