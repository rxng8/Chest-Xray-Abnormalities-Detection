#!/bin/bash

# Author: Alex Nguyen
# Gettysburg College

mkdir tmp
cd ./tmp

for name in "$@" 
do
    if [ $name == coco ];
    then
        kaggle datasets download -d sreevishnudamodaran/vinbigdata-coco-dataset-with-wbf-3x-downscaled

    elif [ $name == orig ];
    then
        kaggle datasets download -d awsaf49/vinbigdata-original-image-dataset

    elif [ $name == yolo ];
    then
        kaggle datasets download -d awsaf49/vinbigdata-yolo-labels-dataset

    elif [ $name == 1024 ];
    then
        kaggle datasets download -d awsaf49/vinbigdata-1024-image-dataset

    elif [ $name == 512 ];
    then
        kaggle datasets download -d awsaf49/vinbigdata-512-image-dataset

    elif [ $name == 256 ];
    then
        kaggle datasets download -d awsaf49/vinbigdata-256-image-dataset

    else
        echo "Please choose the exact dataset name."
    fi
done

cd ..

python ./preprocess_data.py "./tmp"
