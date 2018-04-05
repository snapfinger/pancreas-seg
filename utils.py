# -*- coding: utf-8 -*-
import numpy as np
import os
import sys
import math

data_path = sys.argv[1]

print "data_path: ", data_path

def preprocess(imgs):
    """add one more axis as tf require"""
    imgs = imgs[..., np.newaxis]
    return imgs

def preprocess_front(imgs):
    imgs = imgs[np.newaxis, ...]
    return imgs

# returning the binary label map by the organ ID (especially useful under overlapping cases)
#   label: the label matrix
#   organ_ID: the organ ID
def is_organ(label, organ_ID):
    return label == organ_ID

def pad_2d(image, plane, padval, xmax, ymax, zmax):
    """pad image with zeros to reach dimension as (row_max, col_max)

    Params
    -----
    image : 2D numpy array
        image to pad
    dim : char
        X / Y / Z
    padval : int
        value to pad around
    xmax, ymax, zmax : int
        dimension to reach in x/y/z axis
    """

    if plane == 'X':
        npad = ((0, ymax - image.shape[1]), (0, zmax - image.shape[2]))
        padded = np.pad(image, pad_width=npad, mode='constant', constant_values = padval)
    elif plane =='Z':
        npad = ((0, xmax - image.shape[0]), (0, ymax - image.shape[1]))
        padded = np.pad(image, pad_width=npad, mode='constant', constant_values = padval)

    return padded


#   determining if a sample belongs to the training set by the fold number
#   total_samples: the total number of samples
#   i: sample ID, an integer in [0, total_samples - 1]
#   folds: the total number of folds
#   current_fold: the current fold ID, an integer in [0, folds - 1]
def in_training_set(total_samples, i, folds, current_fold):
    fold_remainder = folds - total_samples % folds
    fold_size = (total_samples - total_samples % folds) / folds
    start_index = fold_size * current_fold + max(0, current_fold - fold_remainder)
    end_index = fold_size * (current_fold + 1) + max(0, current_fold + 1 - fold_remainder)
    return not (i >= start_index and i < end_index)


# returning the filename of the training set according to the current fold ID
def training_set_filename(current_fold):
    return os.path.join(list_path, 'training_' + 'FD' + str(current_fold) + '.txt')


# returning the filename of the testing set according to the current fold ID
def testing_set_filename(current_fold):
    return os.path.join(list_path, 'testing_' + 'FD' + str(current_fold) + '.txt')


# computing the DSC together with other values based on the label and prediction volumes
def DSC_computation(label, pred):
    pred_sum = pred.sum()
    label_sum = label.sum()
    inter_sum = np.logical_and(pred, label).sum()
    return 2 * float(inter_sum) / (pred_sum + label_sum), inter_sum, pred_sum, label_sum


# ------ defining the common variables used throughout the entire flowchart ------
image_path = os.path.join(data_path, 'images')
image_path_ = {}
for plane in ['Z']:
    image_path_[plane] = os.path.join(data_path, 'images_' + plane)
    if not os.path.exists(image_path_[plane]):
        os.makedirs(image_path_[plane])

label_path = os.path.join(data_path, 'labels')
label_path_ = {}
for plane in ['Z']:
    label_path_[plane] = os.path.join(data_path, 'labels_' + plane)
    if not os.path.exists(label_path_[plane]):
        os.makedirs(label_path_[plane])

list_path = os.path.join(data_path, 'lists')
if not os.path.exists(list_path):
    os.makedirs(list_path)

list_training = {}
for plane in ['Z']:
    list_training[plane] = os.path.join(list_path, 'training_' + plane + '.txt')

model_path = os.path.join(data_path, 'models')
if not os.path.exists(model_path):
    os.makedirs(model_path)

log_path = os.path.join(data_path, 'logs')
if not os.path.exists(log_path):
    os.makedirs(log_path)
