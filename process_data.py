import cv2
import os
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm

data_dir = r"./data/"  # location of all the data relative to the programs directory
output_dir = data_dir + "processed_data/"
train_imgs_dir = data_dir + "data_road/training/image_2/" # path to training images
train_gt_dir = data_dir + "data_road/training/gt_image_2/" # path to ground truth images

no_road_val = 76    # ground truth gt_image value for the non-road area
road_val = 105      # ground truth gt_image value for the road area
other_road_val = 0  # ground truth gt_image value for the other-road area
image_pad_val = 255 # value given to the un-imaged area of the shifted pre-processed image and ground truth

max_dist = 0    # max pixel distance to shift pre-processed image and ground truth

def mapLabel(gt_image):
    '''
    Maps ground truth to training label ground truth. Original from
    https://stackoverflow.com/questions/13572448/change-values-in-a-numpy-array
    :param gt_image: ground truth to be mapped
    :return: the mapped ground truth
    '''
    # palette must be given in sorted order
    palette = [other_road_val, no_road_val, road_val, image_pad_val]
    # key gives the new values you wish palette to be mapped to.
    key = np.array([2, 1, 2, 0])
    index = np.digitize(gt_image.ravel(), palette, right=True)

    return np.array(key[index].reshape(gt_image.shape))

def translate_image(image, gt_img, file_name, new_file_name, image_shape, gamma_min=10, gamma_max=10,
                    gamma_step=2, distance=0, distance_step=20):
    '''
    Translation of image data to create more training data.
    The new image names will include the offset pixel count used.
    :param image: training image
    :param gt_img: ground truth of the training image
    :param file_name: name of the file being processed
    :param new_file_name: additional file name string to be added to each processed image
    :param image_shape: tupple of the image shape
    :param gamma_min: darkest adjustment value for the image to be processed
    :param gamma_max: brightest adjustment value for the image to be processed
    :param gamma_step: adjustment step between gamma_min to gamma_max
    :param distance: max distance of transformed images
    :param distance_step: steps between transformed images
    :return: the address list of the new training and ground truth images
    '''
    rows = image_shape[0]
    cols = image_shape[1]
    address_list = [] # address list of the new training images
    first_image = True
    first_gt_img_file_name = None

    gamma_min = int(np.minimum(np.maximum(gamma_min, 1), 100)) # clip value between 1 and 100
    gamma_max = int(np.minimum(np.maximum(gamma_max, gamma_min), 100)) # clip value between gamma_min and 100

    distance = int(np.minimum(np.maximum(distance, 0), cols))  # clip value between 0 and image width

    for gamma in range(gamma_min, gamma_max+gamma_step, gamma_step):

        # Adjust the training image brightness. Copied from
        # https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
        invGamma = 10.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction using the lookup table
        img = cv2.LUT(image, table)

        name_split = file_name.split(".")
        img_file_name = output_dir + name_split[0] + new_file_name + '_bright_' + str(gamma) +".png"
        cv2.imwrite(img_file_name, img)

        if first_image:
            # only create one ground truth image for every gamma adjustment to save memory
            first_gt_img_file_name = output_dir + 'gt_' + name_split[0] + new_file_name + '_bright_' + str(gamma) + ".png"
            new_gt_img = mapLabel(gt_img) # map ground truths to training labels
            cv2.imwrite(first_gt_img_file_name, new_gt_img)

            first_image = False

        address_list.append([img_file_name, first_gt_img_file_name])

        # add step to include the distance in the image transform
        # below has been copied from my 'behavioural cloning project'
        # https://github.com/Heych88/udacity-sdcnd-behavioral-cloning
        for offset in range(distance_step, distance+distance_step, distance_step):
            # TODO: Shift vertically, not just horizontally
            M_right = np.float32([[1, 0, offset], [0, 1, 0]]) # move image right
            M_left = np.float32([[1, 0, -offset], [0, 1, 0]]) # move image left

            img_file_name = output_dir + name_split[0] + '_' + new_file_name + '_bright_' + str(gamma) \
                            + '_' + str(offset) + ".png"
            gt_img_file_name = output_dir + 'gt_' + name_split[0] + '_' + new_file_name + '_bright_' \
                               + str(gamma) + '_' + str(offset) + ".png"

            # shift the image to the right
            new_img = cv2.warpAffine(img, M_right, (cols, rows))
            cv2.imwrite(img_file_name, new_img)

            # shift the ground truth to match the training image
            new_gt_img = cv2.warpAffine(gt_img, M_right, (cols, rows), borderValue=image_pad_val)
            new_gt_img = mapLabel(new_gt_img)
            cv2.imwrite(gt_img_file_name, new_gt_img)

            address_list.append([img_file_name, gt_img_file_name]) # save the image addresses to the address list

            # shift the image to the left
            new_img = cv2.warpAffine(img, M_left, (cols, rows))
            cv2.imwrite(img_file_name, new_img)

            # shift the ground truth to match the training image
            new_gt_img = cv2.warpAffine(gt_img, M_left, (cols, rows), borderValue=image_pad_val)
            new_gt_img = mapLabel(new_gt_img)
            cv2.imwrite(gt_img_file_name, new_gt_img)

            address_list.append([img_file_name, gt_img_file_name])
    return address_list


def getData(image_shape):
    '''
    Loads the training data or expands the training data with image manipulation
    :param image_shape: size of the output image
    :return: the address list of the training and ground truth data
    '''
    saved_file_dir = output_dir + "kitti_list.p"
    address_list = [] # holds the addresses of the processed images and corresponding ground truth's

    # check if there is an address list of the pre-processed images
    if Path(saved_file_dir).exists():
        print("Loading image data from ", saved_file_dir)
        address_list = pickle.load(open(saved_file_dir, "rb"))
    else:
        print("Processing image data from ", saved_file_dir)
        gt_file_names = os.listdir(train_gt_dir)
        rows = image_shape[0]
        cols = image_shape[1]

        for file_num, gt_file_name in enumerate(tqdm(gt_file_names)):
            # open and resize the ground truth label images
            img_path = train_gt_dir + gt_file_name
            gt_img = cv2.resize(cv2.imread(img_path, 0), (cols, rows))

            # use the ground truth label images file name to locate the corresponding training image
            name_split = gt_file_name.split("_")
            file_name = name_split[0] + '_' + name_split[2]
            img_path = train_imgs_dir + file_name
            img = cv2.resize(cv2.imread(img_path, -1), (cols, rows))

            # translate the input images into the data folder
            address_list.extend(translate_image(img, gt_img, file_name, "normal", image_shape,
                                                gamma_min=4, gamma_max=20, distance=max_dist))
            # translate the  input images horizontally and flipped
            address_list.extend(translate_image(cv2.flip(img, 1), cv2.flip(gt_img, 1), file_name, "horz_flip",
                                                image_shape, gamma_min=4, gamma_max=20, distance=max_dist))
            # translate vertically flipped
            #address_list.extend(translate_image(cv2.flip(img, 0), cv2.flip(gt_img, 0), file_name, "vert_flip",
            #                                    image_shape, gamma_min=4, gamma_max=20, distance=max_dist))
            # translate the horizontally and vertically flipped input images
            #address_list.extend(translate_image(cv2.flip(img, -1), cv2.flip(gt_img, -1), file_name, "vert_horz_flip",
            #                                    image_shape, gamma_min=4, gamma_max=20, distance=max_dist))

        # save the address list of each image to prevent remaking the same images
        # delete the 'xxxx_list.p' file, in saved_file_dir to rebuild new images
        pickle.dump(address_list, open(saved_file_dir, "wb+"))

    print("\nTotal data size ", len(address_list))
    return address_list

