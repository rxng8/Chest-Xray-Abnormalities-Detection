# Vin Big Data Chest X-RAY Abnormalies Detection - Kaggle Competition

## About the project:
This project is created for academic purpose and also available publicly. More information about the competition can be found [here](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/).

## Prerequisites:
* [Anaconda]()
* Kaggle API:
  * [Set up kaggle](https://www.kaggle.com/docs/api). 
  * [Create api key and move the key file to ./kaggle folder.](https://github.com/Kaggle/kaggle-api/issues/15#issuecomment-500713264).
* If you are using window, you will have to use window's linux subsystem to run the bash script which download the dataset. Or you can download directly from kaggle.

## Process of development:
* Install dataset.
* Visualize data and Data Analysis.
* Write up training pipeline:
  * Data/Dataset Object Structure
  * Data Loader and Data Generator
  * Deep Learning Model
  * Train it!
* Evaluation
* Visualization

## Run the project:

```
#To be editted
```
---------------
# Project Backlog:

## Week 1: Feb 1 - Feb 5:
* Created project pipeline template. More information can be found in [issue #2](https://github.com/rxng8/VINDR-Competition/issues/2)! (3 hours)
* Research about R-CNN. Here is the list of articles about object detection (R-CNN techniques) I have read and gain an insight to it:
  * [R-CNN, Fast R-CNN, Faster R-CNN, YOLO — Object Detection Algorithms](https://towardsdatascience.com/r-cnn-fast-r-cnn-faster-r-cnn-yolo-object-detection-algorithms-36d53571365e):
    * [R-CNN](https://arxiv.org/abs/1311.2524): This paper is the original technique where the images is segmented into more than 1000 bounding boxes and then classified by the network. (Gain general insight in 1.5 hours)
    * [Fast R-CNN](https://arxiv.org/abs/1504.08083): In this paper, instead of proposing 1000 bounding boxes and feed the region proposal to the convolutional neural network as the original R-CNN technique, the image is passed through a convolutional neural network to predict the RoI feature vector. Then the RoI vectors is passed through the fully connected network. (Gain general insight in 1.5 hours)
    * [Faster R-CNN](https://arxiv.org/abs/1506.01497): This paper improves the Fast R-CNN by upgrading the region proposal process. Instead of using search algorithm to propse regions, this paper has a separate network to predict the region of the object, then the region is passed to the RoI pooling layer as usual. (Gain general insight in 1.5 hours)
    * [YOLO paper](https://arxiv.org/abs/1506.02640): This algorithm is different from above algorithms by dividing the whole image into grids of probability. Then it uses a convolutional neural networks to predict each group of bounding boxes to be the bounding boxes of the object. (Gain general insight in 1.5 hours)
  * After reading those articles, I come to the conclusion that for the Chest X-ray data, there are a lot of small-scale object/abnormalities, so the large frame/grid in YOLO algorithm will have a hard time predicting. Therefore I choose to follow the R-CNN path.

---------------
# Tentative:

## Week 2: Feb 8 - Feb 12:
* Implement data preprocessing pipeline. (3 hours)
* Implements the R-CNN neural network with high-level library. (6 hours)

## Week 3: Feb 15 - Feb 19:
* Finish the whole pipeline and train models. (3 hours)
* Improve and refactor library to the application. (6 hours)

## Week 4: Feb 22 - Feb 26:
* Implement on-scratch Faster R-CNN Model. (9 hours)

## Week 4: March 1 - March 5:
* \<to be eddited\>