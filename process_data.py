import cv2
import _pickle as cPickle
import os
import numpy as np
from collections import namedtuple
import random

project_dir = "/media/haidyn/Self Driving Car/GIT/my_github/SDC/udacity-sdcnd-Semantic-Segmentation/"
data_dir = project_dir + "data/"
output_dir = data_dir + "processed_data/"

# (NOTE! this is taken from the official Cityscapes scripts:)
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )

# (NOTE! this is taken from the official Cityscapes scripts:)
labels = [
    #       name          id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label('unlabeled' ,  0 ,      19 , 'void'            , 0       , False        , True         , (  0,  0,255)),
    Label('road'      ,  1 ,      10 , 'void'            , 0       , False        , True         , (255,  0,255)),
    Label('other'     ,  2 ,      1  , 'void'            , 0       , False        , True         , (  0,  0,  0)),
]

# create a function mapping id to trainId:
id_to_trainId = {label.color: label.trainId for label in labels}
id_to_trainId_map_func = np.vectorize(id_to_trainId.get)

new_img_height = 160 #512 # (the height all images fed to the model will be resized to)
new_img_width = 576 #1024 # (the width all images fed to the model will be resized to)
no_of_classes = 20 # (number of object classes (road, sidewalk, car etc.))


train_imgs_dir = data_dir + "training/image_2/"
train_gt_dir = data_dir + "training/gt_image_2/"


image_shape = (160, 576)


# get the path to all training images and their corresponding label image:
train_img_list = []
train_gt_img_list = []


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
    cv2.imwrite(gt_img_name, gt_img)
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
    gt_img = cv2.resize(cv2.imread(img_path, -1), (image_shape[1], image_shape[0]))

    name_split = gt_file_name.split("_")
    if(len(name_split) == 3):
        file_name = name_split[0] + '_' + name_split[2]
        img_path = train_imgs_dir + file_name
        img = cv2.resize(cv2.imread(img_path, -1), (image_shape[1], image_shape[0]))
    else:
        print("Incorrect file name. Label file name must be in format *_abcd_* with no other preceeding or post '_'.")
        exit(1)

    translate_image(img, gt_img, file_name)

    #print("len ", len(name_split), "  file_name ", file_name)
    #cv2.imshow("image", img)
    #cv2.imshow("gt_image", gt_img)

    #cv2.waitKey(0)


'''
    # convert the label image from id to trainId pixel values:
    id_label = gt_img_small
    trainId_label = id_to_trainId_map_func(id_label)

    # save the label image to project_dir/data:
    trainId_label_path = data_dir + "data/" + img_id + "_trainId_label.png"
    cv2.imwrite(trainId_label_path, trainId_label)
    train_trainId_label_paths.append(trainId_label_path)'''
