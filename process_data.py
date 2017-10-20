import cv2
import os
import numpy as np
from pathlib import Path
import pickle

data_dir = r"./data/"  # location of all the data relative to the programs directory
output_dir = data_dir + "processed_data/"
train_imgs_dir = data_dir + "data_road/training/image_2/"
train_gt_dir = data_dir + "data_road/training/gt_image_2/"

# get the path to all training images and their corresponding label image:
train_img_list = []

no_road_val = 76
road_val = 105
other_road_val = 0
image_pad_val = 255

dist = 0

# create a function mapping id to trainId
# https://stackoverflow.com/questions/13572448/change-values-in-a-numpy-array
def mapLabel(data):
    # palette must be given in sorted order
    palette = [other_road_val, no_road_val, road_val, image_pad_val]
    # key gives the new values you wish palette to be mapped to.
    key = np.array([2, 1, 2, 0])
    index = np.digitize(data.ravel(), palette, right=True)

    return np.array(key[index].reshape(data.shape))

def translateImage(image, gt_img, file_name, new_file_name, distance=0, step=20):
    # Translation of image data to create more training data.
    # The new image names will include the offset pixel count used.
    # img : 3D training image
    # gt_img : 3D training label image
    # file_name: name of the file being processed
    # new_file_name: additional file name to be added to each processed image
    # distance : max distance of transformed images
    # step : steps between transformed images
    # return : list of new images and corresponding label data

    rows, cols, _ = image.shape

    for brighter in range(5, 20):

        # the following is copied from https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
        invGamma = 10.0 / brighter
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")

        # apply gamma correction using the lookup table
        img = cv2.LUT(image, table)

        name_split = file_name.split(".")
        img_name = output_dir + name_split[0] + new_file_name + '_bright_' + str(brighter) +".png"
        cv2.imwrite(img_name, img)

        gt_img_name = output_dir + 'gt_' + name_split[0] + new_file_name + '_bright_' + str(brighter) + ".png"
        new_gt_img = mapLabel(gt_img)
        cv2.imwrite(gt_img_name, new_gt_img)
        train_img_list.append([img_name, gt_img_name])

        # add step to include the distance in the image transform
        for offset in range(step, distance+step, step):
            M_right = np.float32([[1, 0, offset], [0, 1, 0]]) # move image right
            M_left = np.float32([[1, 0, -offset], [0, 1, 0]]) # move image left

            img_name = output_dir + name_split[0] + '_' + new_file_name + '_bright_' + str(brighter) + '_' + str(offset) + ".png"
            gt_img_name = output_dir + 'gt_' + name_split[0] + '_' + new_file_name + '_bright_' + str(brighter) + '_' + str(offset) + ".png"

            # shift the image to the right and append the process image to the list
            new_img = cv2.warpAffine(img, M_right, (cols, rows))
            cv2.imwrite(img_name, new_img)

            new_gt_img = cv2.warpAffine(gt_img, M_right, (cols, rows), borderValue=image_pad_val)
            new_gt_img = mapLabel(new_gt_img)
            cv2.imwrite(gt_img_name, new_gt_img)
            train_img_list.append([img_name, gt_img_name])

            # shift the image to the left and append the process image to the list
            new_img = cv2.warpAffine(img, M_left, (cols, rows))
            cv2.imwrite(img_name, new_img)

            new_gt_img = cv2.warpAffine(gt_img, M_left, (cols, rows), borderValue=image_pad_val)
            new_gt_img = mapLabel(new_gt_img)
            cv2.imwrite(gt_img_name, new_gt_img)
            train_img_list.append([img_name, gt_img_name])


def getData(image_shape):

    saved_file_dir = output_dir + "kitti_list.p"
    if Path(saved_file_dir).exists():
        print("Loading data from ", saved_file_dir)
        global train_img_list
        train_img_list = pickle.load(open(saved_file_dir, "rb"))

    else:

        gt_file_names = os.listdir(train_gt_dir)

        for file_num, gt_file_name in enumerate(gt_file_names):
            print("\rprocessing file %d of %d" % (file_num+1, len(gt_file_names)), end=' ')

            # open and resize the input images:
            img_path = train_gt_dir + gt_file_name
            gt_img = cv2.resize(cv2.imread(img_path, 0), (image_shape[1], image_shape[0]))

            name_split = gt_file_name.split("_")
            file_name = name_split[0] + '_' + name_split[2]
            img_path = train_imgs_dir + file_name
            img = cv2.resize(cv2.imread(img_path, -1), (image_shape[1], image_shape[0]))

            # translate the input images into the data folder
            translateImage(img, gt_img, file_name, "normal", distance=dist, step=20)
            # translate the horizontally flipped input images
            translateImage(cv2.flip(img, 1), cv2.flip(gt_img, 1), file_name, "horz_flip", distance=dist, step=20)
            # translate the vertically flipped input images
            # translateImage(cv2.flip(img, 0), cv2.flip(gt_img, 0), file_name, "vert_flip", use_cityscape, distance=dist, step=20)
            # translate the horizontally and vertically flipped input images
            # translateImage(cv2.flip(img, -1), cv2.flip(gt_img, -1), file_name, "vert_horz_flip", use_cityscape, distance=dist, step=20)

        pickle.dump(train_img_list, open(saved_file_dir, "wb+"))

    print("\nTotal data size ", len(train_img_list))
    return train_img_list

