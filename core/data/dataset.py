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
import re
from numpy.lib.type_check import imag
import pandas as pd
from PIL import Image
import os
import random
import tensorflow as tf
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import collections
import json
from sklearn.model_selection import train_test_split

from .utils import preprocess_img, load_image

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

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
    INT2LABEL = {
        0: "Aortic enlargement",
        1: "Atelectasis",
        2: "Calcification",
        3: "Cardiomegaly",
        4: "Consolidation",
        5: "ILD",
        6: "Infiltration",
        7: "Lung Opacity",
        8: "Nodule/Mass",
        9: "Other lesion",
        10: "Pleural effusion",
        11: "Pleural thickening",
        12: "Pneumothorax",
        13: "Pulmonary fibrosis",
        14: "No finding",
    }
    class COCOInstance():
        def __init__(
            self, 
            filename: str, 
            width: float, 
            height: float
        ) -> None:
            if "/" in filename or "\\" in filename:
                self.filename = re.findall(r"[\\|\/].*jpg$", filename)[0][1:]
            else:
                self.filename = filename
            self.width = width
            self.height = height
            self.boxes = []

        def __str__(self):
            return f"File name: {self.filename}, class: {self.clss}"
        
        def add_box(self, xmin, ymin, xmax, ymax, clss: int, tf_record_mode=False) -> None:
            if tf_record_mode:
                # TODO: Be careful!
                # With this mode, 0 means no diseases, does it also means background? no!
                # self.boxes.append({"class": (clss + 1) % 16, "box":[xmin, ymin, xmax, ymax]})
                self.boxes.append({
                    "class": DatasetCOCO.INT2LABEL[clss], 
                    "box":[
                        min(xmin, xmax), 
                        min(ymin, ymax), 
                        max(xmin, xmax), 
                        max(ymin, ymax)
                    ]
                })
            else:
                self.boxes.append({
                    "class": clss, 
                    "box":[
                        min(xmin, xmax), 
                        min(ymin, ymax), 
                        max(xmin, xmax), 
                        max(ymin, ymax)
                    ]
                })

    def __init__(self, root: str, img_shape: Tuple, batch_size: int=16, steps_per_epoch: int=20, tf_record_mode=False):
        """ Initialization.
        self.dataset: 
        {
            "train": {id: COCOInstance},
            "val": {id: COCOInstance}
        }
        Args:
            root (str): [description]
            img_shape (Tuple): [description]
            batch_size (int, optional): [description]. Defaults to 16.
            steps_per_epoch (int, optional): [description]. Defaults to 20.
        """
        super().__init__(root, img_shape, batch_size=batch_size, steps_per_epoch=steps_per_epoch)
        self.tf_record_mode = tf_record_mode
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
                int(annotation["category_id"]),
                tf_record_mode=self.tf_record_mode
            )

    def export_csv(self, target_train_path: str, target_val_path: str):
        data_train = []

        # Code to append to data list here!
        for image_id, obj in self.dataset["train"].items():
            for _dict in obj.boxes:
                value = (
                    obj.filename,
                    str(obj.width),
                    str(obj.height),
                    str(_dict["class"]),
                    str(_dict["box"][0]),
                    str(_dict["box"][1]),
                    str(_dict["box"][2]),
                    str(_dict["box"][3])
                )
                data_train.append(value)

        data_val = []
        for image_id, obj in self.dataset["val"].items():
            for _dict in obj.boxes:
                value = (
                    obj.filename,
                    str(obj.width),
                    str(obj.height),
                    str(_dict["class"]),
                    str(_dict["box"][0]),
                    str(_dict["box"][1]),
                    str(_dict["box"][2]),
                    str(_dict["box"][3])
                )
                data_val.append(value)

        column_name = ['filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']
        data_train_df = pd.DataFrame(data_train, columns=column_name)
        data_train_df.to_csv(target_train_path, index=False)
        data_val_df = pd.DataFrame(data_val, columns=column_name)
        data_val_df.to_csv(target_val_path, index=False)
        return data_train_df, data_val_df

    def sample(self, preprocess: bool=False, **kwargs) -> np.ndarray:
        """ Return an image that is in the train dataset
        Params:
            preprocess (bool): 
            (preprocess_img **kwargs): 
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


class DatasetCOCOPytorch(DatasetCOCO, Dataset):
    def __init__(self, root: str, img_shape: Tuple, train_set=True, transforms=None, **kwargs):
        super().__init__(root, img_shape, **kwargs)
        if train_set:
            self.dataset = self.dataset["train"]
        else:
            self.dataset = self.dataset["val"]
        # self.indices = collections.defaultdict(int)
        self.transforms = transforms
        self.train_set = train_set
    
    def __build_path(self, img_name):
        if self.train_set:
            return os.path.join(self.root, "train_images", img_name)
        else:
            return os.path.join(self.root, "val_images", img_name)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        # print("id: ", idx)

        # Since the image index in the dataset start from 0, we
        # can assume that the index for the internal dataset be
        # the same as the id of the image.
        img_name = self.dataset[idx].filename
        img_path = self.__build_path(img_name)
        image = np.asarray(Image.open(img_path))

        assert len(image.shape) == 2, "Wrong shape length!"

        image = np.stack([image, image, image])
        image = image.astype('float32')
        image = image - image.min()
        image = image / image.max()
        image = image * 255.0
        image = image.transpose(1,2,0)

        boxes = []
        labels = []
        for dict_datum in self.dataset[idx].boxes:
            boxes.append(dict_datum["box"])
            labels.append(dict_datum["class"])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        # suppose all instances are not crowd
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.uint8)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([idx])
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']

            target['boxes'] = torch.tensor(sample['bboxes'])

        if target["boxes"].shape[0] == 0:
            # Albumentation cuts the target (class 14, 1x1px in the corner)
            target["boxes"] = torch.from_numpy(np.array([[0.0, 0.0, 1.0, 1.0]]))
            target["area"] = torch.tensor([1.0], dtype=torch.float32)
            target["labels"] = torch.tensor([14], dtype=torch.int64)

        # return sample
        return image, target, idx