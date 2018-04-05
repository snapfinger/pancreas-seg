"""
details
-------
1. Get name list of images and labels (/masks/ground truth)
2. Slice each case according to X/Y/Z dim
3. Save a list of [case number, slice number, slicename, labelname, average pixel value of the slice,
                  total pixel number of ground truth, bounding box (minA, maxA, minB, maxB)]

This code is adapted from https://github.com/198808xc/OrganSegC2F/blob/master/OrganSegC2F/init.py
"""

import numpy as np
import os
import sys
import time
from utils import *


# read input arguments
data_path = sys.argv[1] + "/"
organ_number = int(sys.argv[2])
folds = int(sys.argv[3])
low_range = int(sys.argv[4])
high_range = int(sys.argv[5])

if __name__=="__main__":
    # get image list
    image_list = []
    image_filename = []
    keyword = ''
    for directory, _, file_ in os.walk(image_path):
	for filename in sorted(file_):
	    if keyword in filename:
		image_list.append(os.path.join(directory, filename))
		image_filename.append(os.path.splitext(filename)[0])

    # get label list
    label_list = []
    label_filename = []
    for directory, _, file_ in os.walk(label_path):
	for filename in sorted(file_):
	    if keyword in filename:
		label_list.append(os.path.join(directory, filename))
		label_filename.append(os.path.splitext(filename)[0])

    # check if #image equals #labels
    if len(image_list) != len(label_list):
	exit('Error: the number of labels and the number of images are not equal!')


    total_samples = len(image_list)

    for plane in ['Z']:
    	output = open(list_training[plane], 'w')
    	output.close()

    print 'Initialization starts.'

# iterate through all samples
for i in range(total_samples):
    start_time = time.time()
    print 'Processing ' + str(i + 1) + ' out of ' + str(len(image_list)) + ' files.'

    image = np.load(image_list[i])
    label = np.load(label_list[i])

    # only z for now
    for plane in ['Z']:
        # slice_number is the number of slices of corresponding dimension (X/Y/Z)
        slice_number = label.shape[2]

        image_directory_ = os.path.join(image_path_[plane], image_filename[i])
        if not os.path.exists(image_directory_):
            os.makedirs(image_directory_)

        label_directory_ = os.path.join(label_path_[plane], label_filename[i])
        if not os.path.exists(label_directory_):
            os.makedirs(label_directory_)

        print '    Slicing data: ' + str(time.time() - start_time) + ' second(s) elapsed.'
        # for storing the total number of pixels of ground truth mask
        sum_ = np.zeros((slice_number, organ_number + 1), dtype = np.int)
        # for storing bounding boxes of ground truth masks (A_min, A_max, B_min, B_max)
        minA = np.zeros((slice_number, organ_number + 1), dtype = np.int)
        maxA = np.zeros((slice_number, organ_number + 1), dtype = np.int)
        minB = np.zeros((slice_number, organ_number + 1), dtype = np.int)
        maxB = np.zeros((slice_number, organ_number + 1), dtype = np.int)
        # for storing mean pixel value of each slice
        average = np.zeros((slice_number), dtype = np.float)

        # iterate through all slices of current case i and current plane
        for j in range(0, slice_number):
            # image_filename_ sample dir: image_X /  0001  / 0001.npy
            #                              plane/ case num / slice num
            image_filename_ = os.path.join( \
                image_path_[plane], image_filename[i], '{:0>4}'.format(j) + '.npy')

            label_filename_ = os.path.join( \
                label_path_[plane], label_filename[i], '{:0>4}'.format(j) + '.npy')

            image_ = image[:, :, j]
            label_ = label[:, :, j]

            # threshold image to specified range ([-100, 240] for pancreas)
            image_[image_ < low_range] = low_range
            image_[image_ > high_range] = high_range

            # save sliced image and label
            if not os.path.isfile(image_filename_) or not os.path.isfile(label_filename_):
                np.save(image_filename_, image_)
                np.save(label_filename_, label_)

            # compute the mean value of the slice
            average[j] = float(image_.sum()) / (image_.shape[0] * image_.shape[1])

            for o in range(1, organ_number + 1):
                # this is the sum of pixel numbers of a ground truth mask
                sum_[j, o] = (is_organ(label_, o)).sum()
                # record the coordinates of ground truth mask pixels
                arr = np.nonzero(is_organ(label_, o))

                # save the bounding box of ground truth mask (A_min, A_max, B_min, B_max)
                minA[j, o] = 0 if not len(arr[0]) else min(arr[0])
                maxA[j, o] = 0 if not len(arr[0]) else max(arr[0])
                minB[j, o] = 0 if not len(arr[1]) else min(arr[1])
                maxB[j, o] = 0 if not len(arr[1]) else max(arr[1])

        # iterate each slice of current case i
        for j in range(0, slice_number):
            image_filename_ = os.path.join( \
                image_path_[plane], image_filename[i], '{:0>4}'.format(j) + '.npy')

            label_filename_ = os.path.join( \
                label_path_[plane], label_filename[i], '{:0>4}'.format(j) + '.npy')

            # append the following output to training_X/Y/Z.txt
            output = open(list_training[plane], 'a+')
            # case number, slice number
            output.write(str(i) + ' ' + str(j))
            # image file name, label file name
            output.write(' ' + image_filename_ + ' ' + label_filename_)
            # average pixel value of slice j, case i, and current plane
            output.write(' ' + str(average[j]))
            # sum of ground truth pixels, and bounding box of gt mask (A_min, A_max, B_min, B_max)
            for o in range(1, organ_number + 1):
                output.write(' ' + str(sum_[j, o]) + ' ' + str(minA[j, o]) + \
                    ' ' + str(maxA[j, o]) + ' ' + str(minB[j, o]) + ' ' + str(maxB[j, o]))

            output.write('\n')

        output.close()

        print '  ' + plane + ' plane is done: ' + \
            str(time.time() - start_time) + ' second(s) elapsed.'

    print 'Processed ' + str(i + 1) + ' out of ' + str(len(image_list)) + ' files: ' + \
        str(time.time() - start_time) + ' second(s) elapsed.'


# create the 4 training image lists
print 'Writing training image list.'
for f in range(folds):
    list_training_ = training_set_filename(f)
    output = open(list_training_, 'w')
    for i in range(total_samples):
        if in_training_set(total_samples, i, folds, f):
            output.write(str(i) + ' ' + image_list[i] + ' ' + label_list[i] + '\n')
    output.close()

# create the 4 test image lists
print 'Writing testing image list.'
for f in range(folds):
    list_testing_ = testing_set_filename(f)
    output = open(list_testing_, 'w')
    for i in range(total_samples):
        if not in_training_set(total_samples, i, folds, f):
            output.write(str(i) + ' ' + image_list[i] + ' ' + label_list[i] + '\n')
    output.close()

print 'Initialization is done.'
