import cv2
import _pickle as cPickle
import os
import numpy as np
from collections import namedtuple
import random

project_dir = "/media/haidyn/Self Driving Car/GIT/my_github/SDC/udacity-sdcnd-Semantic-Segmentation/"
data_dir = project_dir + "data/"
output_dir = data_dir + "processed_data/"
train_imgs_dir = data_dir + "training/image_2/"
train_gt_dir = data_dir + "training/gt_image_2/"

new_img_height = 160 #512 # (the height all images fed to the model will be resized to)
new_img_width = 576 #1024 # (the width all images fed to the model will be resized to)
no_of_classes = 20 # (number of object classes (road, sidewalk, car etc.))

image_shape = (160, 576)

# get the path to all training images and their corresponding label image:
train_img_list = []
train_gt_img_list = []

# create a function mapping id to trainId:
def map_label(data):
    # palette must be given in sorted order
    palette = [0, 76, 105]
    # key gives the new values you wish palette to be mapped to.
    key = np.array([0, 128, 255])
    index = np.digitize(data.ravel(), palette, right=True)

    return key[index].reshape(data.shape)

def translate_image(img, gt_img, file_name, distance=0, step=20):
    # Translation of image data to create more training data
    # img : 3D image data
    # y_value : float label data for the image
    # distance : max distance of transformed images
    # step : steps between transformed images
    # y_value_gain : the gain of the label data over the transform distance
    # return : list of new images and corresponding label data

    rows, cols, _ = img.shape

    name_split = file_name.split(".")
    img_name = data_dir + "processed_data/" + name_split[0] + ".png"
    cv2.imwrite(img_name, img)
    train_img_list.append(img_name)

    gt_img_name = data_dir + "processed_data/" + 'gt_' + name_split[0] + ".png"
    cv2.imwrite(gt_img_name, map_label(gt_img))
    train_gt_img_list.append(gt_img_name)

    # add step to include the distance in the image transform
    '''for offset in range(step, distance+step, step):
        M_right = np.float32([[1, 0, offset], [0, 1, 0]]) # move image right
        M_left = np.float32([[1, 0, -offset], [0, 1, 0]]) # move image left

        # shift the image to the right and append the process image to the list
        img_list.append(cv2.warpAffine(img, M_right, (cols, rows)))
        # shift the image to the left and append the process image to the list
        img_list.append(cv2.warpAffine(img, M_left, (cols, rows)))'''





gt_file_names = os.listdir(train_gt_dir)

for file_num, gt_file_name in enumerate(gt_file_names):
    if file_num % 10 == 0:
        print("file %d/%d" % (file_num, len(gt_file_names)-1))

    # open and resize the input images:
    img_path = train_gt_dir + gt_file_name
    gt_img = cv2.resize(cv2.imread(img_path, 0), (image_shape[1], image_shape[0]))

    name_split = gt_file_name.split("_")
    if(len(name_split) == 3):
        file_name = name_split[0] + '_' + name_split[2]
        img_path = train_imgs_dir + file_name
        img = cv2.resize(cv2.imread(img_path, -1), (image_shape[1], image_shape[0]))
    else:
        print("Incorrect file name. Label file name must be in format *_abcd_* with no other preceeding or post '_'.")
        exit(1)

    translate_image(img, gt_img, file_name)

    # convert the label image from id to trainId pixel values:
    #id_label = gt_img
    #trainId_label = id_to_trainId_map_func(gt_img)

    #print("len ", len(name_split), "  file_name ", file_name)
    #cv2.imshow("image", trainId_label)



'''
    

    # save the label image to project_dir/data:
    trainId_label_path = data_dir + "data/" + img_id + "_trainId_label.png"
    cv2.imwrite(trainId_label_path, trainId_label)
    train_trainId_label_paths.append(trainId_label_path)'''
