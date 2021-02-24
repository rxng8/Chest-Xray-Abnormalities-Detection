# Chest X RAY Abnormalities Detection | VinBigData's Kaggle Competition

## About the project:
This project is created for academic purpose and also available publicly. More information about the competition can be found [here](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/).

# Prerequisites
* [Anaconda](https://www.anaconda.com/products/individual)
* Kaggle API:
  * [Set up kaggle](https://www.kaggle.com/docs/api). 
  * [Create api key and move the key file to ./kaggle folder.](https://github.com/Kaggle/kaggle-api/issues/15#issuecomment-500713264).
* If you are using window, you will have to use window's linux subsystem to run the bash script which download the dataset. Or you can download directly from kaggle.
* Clone this repository recursively (to download the submodules):
```
git clone --recursive https://github.com/rxng8/Chest-Xray-Abnormalities-Detection.git
```

<!-- # Process of development
* Install dataset.
* Visualize data and Data Analysis.
* Write up training pipeline:
  * Data/Dataset Object Structure
  * Data Loader and Data Generator
  * Deep Learning Model
  * Train it!
* Evaluation
* Visualization -->

# Run the project

## Run the project with Tensorflow Object Detection API
**Note:** The object detection api is followed directly from [Tensorflow Object Detection API's tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html).

**Note** Currently, the detection api use [COCO Dataset](https://www.kaggle.com/sreevishnudamodaran/vinbigdata-coco-dataset-with-wbf-3x-downscaled/). Please make sure to download that dataset (from Kaggle), unzip and put it in the dataset directory. Or you can use the [script](/download_dataset.sh) in the root directory that downloads the dataset for you (Linux-like machine required, kaggle installed required):

```
bash download_dataset.sh coco
```

1. Installation according to [this tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html), briefly:
   * Create conda virtual environment.
   * Install tensorflow (If using GPU version, install CUDA toolkit, cuDNN).
   * Download the Tensorflow Model Garden. (**downloaded in this git repo**)
   * Install Protobuf.
   * Install COCO API.
   * Install the Object detection API from the Tensorflow Model Garden repo.

2. Training custom object detector.
   * Prepare the workspace as mentioned in [this tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#preparing-the-workspace).
   * Export csv files for the current dataset:
    ```
    # At the root directory
    python export_COCO_csv.py
    ```
   * Copy all images from the `train_images` and `val_images` directories in the `COCO Dataset` into the `train` and `test` directories in the tensorflow workspace setup in the first bullet point.
   * Generate the `tfrecord` files for the dataset in order to call the training API:
    ```
    # In the root directory. Generate tfrecord for training.
    python ./Tensorflow/scripts/generate_tf_records.py -j ./tmp/train.csv  -i ./Tensorflow/workspace/training-demo/images/train -l ./Tensorflow/workspace/training-demo/annotations/label_map.pbtxt -o ./Tensorflow/workspace/training-demo/annotations/train.record

    # In the root directory. Generate tfrecord for validation.
    python ./Tensorflow/scripts/generate_tf_records.py -j ./tmp/val.csv  -i ./Tensorflow/workspace/training-demo/images/test -l ./Tensorflow/workspace/training-demo/annotations/label_map.pbtxt -o ./Tensorflow/workspace/training-demo/annotations/val.record
    ```
   * Download any of the pre-trained model in this [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) (I used [`SSD ResNet50 V1 FPN 640x640` model](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz)). Please refer to [this part](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#download-pre-trained-model) of the tutorial for more information.
   * Create directory and config the model according to [this part](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#configure-the-training-pipeline) of the tutorial.
   *  in the 
   * Copy the file `model_main_tf2.py` to the working directory and train the model by following the command, according to [this part](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#training-the-model):
    ```
    # cd inside training-demo directory.
    python model_main_tf2.py --model_dir=models/mymodel --pipeline_config_path=models/mymodel/pipeline.config
    ```
3. Evaluating the detector.

[**To be editted**]

1. Visualizing data and training performance with Tensorboard.

[**To be editted**]

## Run the project whole pipeline
[**To be editted**]

---------------
# Project Backlog:

## Week 1: Feb 1 - Feb 5:
* **[3 hours]** Created project pipeline template. More information can be found in [issue #2](https://github.com/rxng8/VINDR-Competition/issues/2)! 
* **[6 hours]** Research about R-CNN. Here is the list of articles about object detection (R-CNN techniques) I have read and gain an insight to it:
  * **[6 hours]** [R-CNN, Fast R-CNN, Faster R-CNN, YOLO — Object Detection Algorithms](https://towardsdatascience.com/r-cnn-fast-r-cnn-faster-r-cnn-yolo-object-detection-algorithms-36d53571365e):
    * **[1.5 hours]** [R-CNN](https://arxiv.org/abs/1311.2524): This paper is the original technique where the images is segmented into more than 1000 bounding boxes and then classified by the network.
    * **[1.5 hours]** [Fast R-CNN](https://arxiv.org/abs/1504.08083): In this paper, instead of proposing 1000 bounding boxes and feed the region proposal to the convolutional neural network as the original R-CNN technique, the image is passed through a convolutional neural network to predict the RoI feature vector. Then the RoI vectors is passed through the fully connected network.
    * **[1.5 hours]** [Faster R-CNN](https://arxiv.org/abs/1506.01497): This paper improves the Fast R-CNN by upgrading the region proposal process. Instead of using search algorithm to propse regions, this paper has a separate network to predict the region of the object, then the region is passed to the RoI pooling layer as usual.
    * **[1.5 hours]** [YOLO paper](https://arxiv.org/abs/1506.02640): This algorithm is different from above algorithms by dividing the whole image into grids of probability. Then it uses a convolutional neural networks to predict each group of bounding boxes to be the bounding boxes of the object.
  * After reading those articles, I come to the conclusion that for the Chest X-ray data, there are a lot of small-scale object/abnormalities, so the large frame/grid in YOLO algorithm will have a hard time predicting. Therefore I choose to follow the R-CNN path.

## Week 2: Feb 8 - Feb 12:
* **[5 hours]** Implement data preprocessing pipeline, in particular:
  * [`data`](./core/data): Contains data preprocessing utilities and dataset models.
* **[4 hours]** Implements the R-CNN neural network with high-level library, in particular:
  * [`models`](./core/models): Contains model building utilities, and initial code of Faster R-CNN model.

## Week 3: Feb 15 - Feb 19:

* [`API`](./Tensorflow): Integrate the use of `Tensorflow Object Detection API`.
  * **[3 hours]** Install the tensorflow model library and setup protobuf.
  * [more reference](https://towardsdatascience.com/how-to-build-a-custom-object-detector-classifier-using-tensorflow-object-detection-api-811b7bcd31c4)
  * **[2 hours]** Write [COCO Dataset Wrapper](./core/data/dataset.py) for further processing, especially export to `tfrecord`.
  * **[2 hours]** Generate image annotation and `tfrecord` annotation for tensorflow model usage.

## Week 4: Feb 22 - Feb 26:
* [**7 hours**] Successfully implement the data pipeline and export to tfrecord.
  * [**4 hours**] Implement the data wrapper class [`DatasetCOCO`](./core/data/dataset.py). Sucessfully generate csv file as needed.
  * [**3 hours**] Copy and re-implement the scripts [`generate_tf_records.py`](./Tensorflow/scripts/generate_tf_records.py). This file takes inputs csv containing paths, bounding boxes, classes, etc, to generate `tfrecord` files for api training. (I spend almost 2 hours on fixing the bugs: [sanity error](https://stackoverflow.com/questions/62075321/tensorflow-python-framework-errors-impl-invalidargumenterror-invalid-argument))
* [**2 hours**] Writing documentation for methods, installation, and requirements.
* [**4 hours**] Train and export models to be evaluated in future.
  * (Do not counted this) [**10 hours**] Wait for the model to be trained on GPU.
  * [**2 hours**] Exporting models and evaluate on the validation set. (To be added!)
  * [**2 hours**] Draws tensorflow graph, here is the result: (To be added!)

---------------

# Tentative:

## Week 5: March 1 - March 5:
* Finish the whole pipeline and train models.
* Improve and refactor library to the application.
  * Implement according to [this tutorial](https://towardsdatascience.com/training-a-tensorflow-faster-r-cnn-object-detection-model-on-your-own-dataset-b3b175708d6d) and this [kaggle notebook](https://www.kaggle.com/leighplt/starter-faster-r-cnn)
* Implement on-scratch Faster R-CNN Model. (9 hours)

## Week 6: March 8 - March 12:

## Week 7: March 15 - March 19:

## Week 8: March 22 - March 26:

## Week 9: March 29 - April 2:

## Week 10: April 5 - April 9:

## Week 11: April 12 - April 15:

## Week 12: April 18 - April 22:

## Week 13: April 25 - April 28:

## Week 14: May 1 - May 5:

## Week 15: May 8 - May 12:

## Week 16: May 15 - May 19:

-----------------

# Resources:
1. [https://missinglink.ai/guides/tensorflow/building-faster-r-cnn-on-tensorflow-introduction-and-examples/](https://missinglink.ai/guides/tensorflow/building-faster-r-cnn-on-tensorflow-introduction-and-examples/): 
2. [Tensorflow Object Detection Model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md): 
3. [Faster R-CNN layer information](https://towardsdatascience.com/faster-r-cnn-object-detection-implemented-by-keras-for-custom-data-from-googles-open-images-125f62b9141a): 
4. [Tensorflow object detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md): 