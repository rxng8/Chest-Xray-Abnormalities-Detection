# %%

import pandas as pd
import numpy as np
import cv2
import os
import re
import pydicom
import warnings

import time

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd

import json

from PIL import Image
from torch.distributions import transforms

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform

import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler

from matplotlib import pyplot as plt
import random

from core.data import DatasetCOCO, DatasetCOCOPytorch, show_img

paddingSize= 0

DIR_INPUT = './dataset/vinbigdata-coco-dataset-with-wbf-3x-downscaled/vinbigdata-coco-dataset-with-wbf-3x-downscaled'
DIR_TRAIN = f'{DIR_INPUT}/train'
DIR_TEST = f'{DIR_INPUT}/test'

# %%

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

num_classes = 15 # 14 Classes + 1 background

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# %%

IMG_SHAPE = (256, 256)

root_data_folder = "./dataset/vinbigdata-coco-dataset-with-wbf-3x-downscaled/vinbigdata-coco-dataset-with-wbf-3x-downscaled"

ds = DatasetCOCO(root_data_folder, img_shape=IMG_SHAPE)

# %%

len(ds.dataset["train"].values())

# %%

def get_train_transform():
    return A.Compose([
        A.Flip(0.5),
        A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=45, p=0.25),
        A.LongestMaxSize(max_size=800, p=1.0),
        # Dilation(),
        # FasterRCNN will normalize.
        A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


# train_dataset = DatasetCOCOPytorch(root=root_data_folder, img_shape=IMG_SHAPE, transform=get_train_transform())

# %%

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def collate_fn(batch):
    return tuple(zip(*batch))
train_dataset = DatasetCOCOPytorch(root=root_data_folder, img_shape=IMG_SHAPE, transforms=get_train_transform())
# train_dataset = DatasetCOCOPytorch(train_df, DIR_TRAIN, get_train_transform())
valid_dataset = DatasetCOCOPytorch(root=root_data_folder, img_shape=IMG_SHAPE, train_set=False, transforms=get_train_transform())

# split the dataset in train and test set
indices = torch.randperm(len(train_dataset)).tolist()
# Create train and validate data loader
train_data_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn
)

valid_data_loader = DataLoader(
    valid_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
)

# %%

# Train dataset sample
images, targets, image_ids = next(iter(train_data_loader))
# images = list(image.to(device) for image in images)
# targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

# %%


train_dataset[0]
# %%
