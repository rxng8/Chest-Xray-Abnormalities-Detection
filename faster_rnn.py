# %%

import pandas as pd
import numpy as np
import cv2
import os
import re
import pydicom
import warnings

import sys
import time

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd

from tqdm import tqdm

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

IMG_SHAPE = (256, 256)
BATCH_SIZE = 2

DIR_INPUT = './dataset/vinbigdata-coco-dataset-with-wbf-3x-downscaled/vinbigdata-coco-dataset-with-wbf-3x-downscaled'
DIR_TRAIN = f'{DIR_INPUT}/train'
DIR_TEST = f'{DIR_INPUT}/test'

MODEL_DIR = "./models/"

root_data_folder = "./dataset/vinbigdata-coco-dataset-with-wbf-3x-downscaled/vinbigdata-coco-dataset-with-wbf-3x-downscaled"

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# %%

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

num_classes = 15 # 14 Classes + 1 background

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# %%

# UNCOMMENT TO LOAD WEIGHT
# MODEL_NAME = "model.pth"
# model.load_state_dict(torch.load(os.path.join(MODEL_DIR, MODEL_NAME)))

# %%

def dilation(img): # custom image processing function
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple(np.random.randint(1, 6, 2)))
    img = cv2.dilate(img, kernel, iterations=1)
    return img

class Dilation(ImageOnlyTransform):
    def apply(self, img, **params):
        return dilation(img)

def collate_fn(batch):
    return tuple(zip(*batch))

def get_train_transform():
    # return A.Compose([
    #     A.Flip(0.5),
    #     A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=45, p=0.25),
    #     A.LongestMaxSize(max_size=800, p=1.0),
    #     # Dilation(),
    #     # FasterRCNN will normalize.
    #     A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0, p=1.0),
    #     ToTensorV2(p=1.0)
    # ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
    return A.Compose(
        [
            # A.Flip(0.5),
            # A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=45, p=0.25),
            # A.LongestMaxSize(max_size=800, p=1.0),
            # Dilation(),
            A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0)
        ],
        bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']}
    )


# train_dataset = DatasetCOCOPytorch(root=root_data_folder, img_shape=IMG_SHAPE, transform=get_train_transform())

# %%

train_dataset = DatasetCOCOPytorch(root=root_data_folder, img_shape=IMG_SHAPE, transforms=get_train_transform())
# train_dataset = DatasetCOCOPytorch(train_df, DIR_TRAIN, get_train_transform())
valid_dataset = DatasetCOCOPytorch(root=root_data_folder, img_shape=IMG_SHAPE, train_set=False, transforms=get_train_transform())

# split the dataset in train and test set
# indices = torch.randperm(len(train_dataset)).tolist()

# Create train and validate data loader
train_data_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    # num_workers=4,
    collate_fn=collate_fn
)

valid_data_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    # num_workers=4,
    collate_fn=collate_fn
)

# %%

# Train dataset sample
images, targets, image_ids = next(iter(train_data_loader))
# images = list(image.to(device) for image in images)
# targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

# %%

images, targets, image_ids = train_dataset[2]
# %%

images.shape

# %%

targets


# %%

model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

num_epochs =  1 #Low epoch to save GPU time


# %%
itr = 1
for epoch in range(num_epochs):
    # loss_hist.reset()
    print("Epochs:", epoch)
    time.sleep(1)
    with tqdm(total=len(train_dataset)) as pbar:
        for images, targets, image_ids in train_data_loader:
            try:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)  
                
                losses = sum(loss for loss in loss_dict.values())
                loss_value = losses.item()

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                if itr % 50 == 0:
                    print(f"Iteration #{itr} loss: {loss_value}")

                itr += 1
                pbar.update(1)
            except:
                continue
            # break
        # pass
    # update the learning rate
    # if lr_scheduler is not None:
    #     lr_scheduler.step()
    # lossHistoryepoch.append(loss_hist.value)
    # print(f"Epoch #{epoch} loss: {loss_hist.value}")  
    # pass

# %%
MODEL_NAME = "model1.pth"
torch.save(model.state_dict(), os.path.join(MODEL_DIR, MODEL_NAME))


# %%

# Eval

def format_prediction_string(labels, boxes, scores):
    pred_strings = []
    for j in zip(labels, scores, boxes):
        pred_strings.append("{0} {1:.4f} {2} {3} {4} {5}".format(
            j[0], j[1], j[2][0], j[2][1], j[2][2], j[2][3]))

    return " ".join(pred_strings)


# %%

detection_threshold = 0.5
results = []

with torch.no_grad():

    for images, image_ids in valid_data_loader:

        images = list(image.to(device) for image in images)
        outputs = model(images)

        for i, image in enumerate(images):

            image_id = image_ids[i]

            result = {
                'image_id': image_id,
                'PredictionString': '14 1.0 0 0 1 1'
            }

            boxes = outputs[i]['boxes'].data.cpu().numpy()
            labels = outputs[i]['labels'].data.cpu().numpy()
            scores = outputs[i]['scores'].data.cpu().numpy()

            if len(boxes) > 0:

                labels = labels - 1
                labels[labels == -1] = 14

                selected = scores >= detection_threshold

                boxes = boxes[selected].astype(np.int32)
                scores = scores[selected]
                labels = labels[selected]

                if len(boxes) > 0:
                    result = {
                        'image_id': image_id,
                        'PredictionString': format_prediction_string(labels, boxes, scores)
                    }


            results.append(result)
        

# %%

results[0:2]

# %%

test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
test_df.head()

# %%


sample = images[1].permute(1,2,0).cpu().numpy()
boxes = outputs[1]['boxes'].data.cpu().numpy()
scores = outputs[1]['scores'].data.cpu().numpy()

boxes = boxes[scores >= detection_threshold].astype(np.int32)

# %%

fig, ax = plt.subplots(1, 1, figsize=(16, 8))

for box in boxes:
    cv2.rectangle(sample,
                  (box[0], box[1]),
                  (box[2], box[3]),
                  (220, 0, 0), 2)
    
ax.set_axis_off()
ax.imshow(sample)

# %%

test_df.to_csv('submission.csv', index=False)




