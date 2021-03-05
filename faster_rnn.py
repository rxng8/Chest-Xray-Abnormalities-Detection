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

