# Semantic Segmentation
### Introduction

This repository contains the project solution to Udacities Self-Driving Car ND semester 3 Project 2: Semantic Segmentation. The Udacity repository can be found [here](https://github.com/udacity/CarND-Semantic-Segmentation).

The project takes image data and performs Semantic Segmentation to label the pixels of a road in images using a Fully Convolutional Network (FCN).

![Test set 3](runs/10 epoch .007 loss/um_000005.png)

### Setup
##### Install Dependencies
The code requires the following to be installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)

##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training test images.

##### Build Instructions
1. Clone this repo.

  ```sh
  $ git clone https://github.com/Heych88/udacity-sdcnd-Semantic-Segmentation.git
  ```

### Start
##### Run
1. Navigate to the cloned repositories location and open the `code/` folder.

 ```sh
 $ cd <path to repository>/udacity-sdcnd-Semantic-Segmentation/
 ```

2. Run the project script
 ```sh
 $ python3 main.py
 ```

If the above worked correctly, the Semantic Segmentation program will process the kitti dataset and train the model for three epochs and will display the training progress of the network.

Upon completion of the training,  the network will test the model and save the output images in the `run/` folder.  
