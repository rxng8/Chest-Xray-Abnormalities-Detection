#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
   Author: Alex Nguyen
   Gettysburg College
"""

import sys
from pathlib import Path
from typing import Collection, List, Dict, Tuple
import numpy as np
from numpy.lib.type_check import imag
import pandas as pd
from PIL import Image
import os
import random
import tensorflow as tf
import collections
import json
from sklearn.model_selection import train_test_split

from .utils import preprocess_img, load_image


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
    
    def sample(self, preprocess: bool=False, **kwargs) -> np.ndarray:
        """ Return an image that is in the train dataset

        Returns:
            np.ndarray: A numpy repr of the image. If preprocess is False,
                an image with original shape is returned. Otherwise, it will
                preprocess the image
        """
        dir = os.listdir(self.train_data)
        r = random.randint(0, len(dir) - 1)
        img = load_image(os.path.join(self.train_data, dir[r]))
        if preprocess:
            img = preprocess_img(img, **kwargs)
        return img

class DatasetCOCO(BaseDataset):
    
    class COCOInstance():
        def __init__(
            self, 
            filename: str, 
            width: float, 
            height: float
        ) -> None:
            self.filename = filename
            self.width = width
            self.height = height
            self.boxes = []

        def __str__(self):
            return f"File name: {self.filename}, class: {self.clss}"
        
        def add_box(self, xmin, ymin, xmax, ymax, clss: int) -> None:
            self.boxes.append({"class": clss, "box":[xmin, ymin, xmax, ymax]})

    def __init__(self, root: str, img_shape: Tuple, batch_size: int=16, steps_per_epoch: int=20):
        super().__init__(root, img_shape, batch_size=batch_size, steps_per_epoch=steps_per_epoch)
        self.val_label = os.path.join(root, "val_annotations.json")
        self.train_label = os.path.join(root, "train_annotations.json")
        self.val_data = os.path.join(root, "val_images")
        self.train_data = os.path.join(root, "train_images")
        self.dataset: Dict[str, collections.defaultdict] = \
            {"train": collections.defaultdict(DatasetCOCO.COCOInstance),
            "val": collections.defaultdict(DatasetCOCO.COCOInstance)}
        self._load_data(self.train_label, "train")
        self._load_data(self.val_label, "val")

    def _load_data(self, annotation_path, key) -> None:
        """ Load the data from the annotation files into own dataset.

        Args:
            annotation_path (path, str, etc.): Path of the annotation json file.
            key (str): The key in the dataset dictionary. Either "train" or "val".
        """
        with open(annotation_path) as f:
            json_data = json.load(f)

        # Load data and creting instances
        for image in json_data["images"]:
            image_id = image["id"]
            if image_id not in self.dataset[key]:
                self.dataset[key][image_id] = DatasetCOCO.COCOInstance(
                    filename = image["file_name"], 
                    width = image["width"], 
                    height = image["height"]
                )

        # Load bounding boxes and add to the instances.
        for annotation in json_data["annotations"]:
            image_id = annotation["image_id"]
            self.dataset[key][image_id].add_box(
                *annotation["bbox"],
                annotation["category_id"]
            )


    def sample(self, preprocess: bool=False, **kwargs) -> np.ndarray:
        """ Return an image that is in the train dataset

        Returns:
            np.ndarray: A numpy repr of the image. If preprocess is False,
                an image with original shape is returned. Otherwise, it will
                preprocess the image
        """
        dir = os.listdir(self.train_data)
        r = random.randint(0, len(dir) - 1)
        img = load_image(os.path.join(self.train_data, dir[r]))
        if preprocess:
            img = preprocess_img(img, **kwargs)
        else:
            assert len(img.shape) == 2, "Wrong expected image shape."
            img = tf.expand_dims(img, axis=-1)
        return img

    def __str__(self):
        len_train = len(self.dataset["train"])
        len_val = len(self.dataset["val"])
        return f"Dataset:\nTrain: {len_train} items.\nVal: {len_val} items."