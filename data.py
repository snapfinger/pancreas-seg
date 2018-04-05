"""
This code is to
1. Create train & test input to Network as numpy arrays
2. Load the train & test numpy arrays
"""

import numpy as np
from utils import *

# data type to save as np array
npdtype = np.float32


def create_train_data(current_fold, plane):
    """
    Crop each slice by its ground truth bounding box,
    then pad zeros to form uniform dimension,
    rescale pixel intensities to [0,1]
    """
    # get the list of image and label number of current_fold
    imlb_list = open(training_set_filename(current_fold), 'r').read().splitlines()
    current_fold = current_fold
    training_image_set = np.zeros((len(imlb_list)), dtype = np.int)

    for i in range(len(imlb_list)):
        s = imlb_list[i].split(' ')
        training_image_set[i] = int(s[0])

    slice_list = open(list_training[plane], 'r').read().splitlines()
    slices = len( slice_list)
    image_ID = np.zeros(( slices), dtype = np.int)
    slice_ID = np.zeros(( slices), dtype = np.int)
    image_filename = ['' for l in range( slices)]
    label_filename = ['' for l in range( slices)]
    pixels = np.zeros((slices), dtype = np.int)

    for l in range(slices):
        s =  slice_list[l].split(' ')
        image_ID[l] = s[0]
        slice_ID[l] = s[1]
        image_filename[l] = s[2]
        label_filename[l] = s[3]
        pixels[l] = int(s[organ_ID * 5])

    create_slice_list = []
    create_label_list = []

    for l in range(slices):
        if image_ID[l] in training_image_set and pixels[l] >= 100:
            create_slice_list.append(image_filename[l])
            create_label_list.append(label_filename[l])
    if len(create_slice_list)!= len(create_label_list):
        raise ValueError('slice number does not equal label number!')

    total = len(create_slice_list)

    img_rows = XMAX
    img_cols = YMAX

    imgs = np.ndarray((total, img_rows, img_cols), dtype = npdtype)
    imgs_mask = np.ndarray((total, img_rows, img_cols), dtype = npdtype)

    print('-'*30)
    print('  Creating training data...')
    print('-'*30)

    for i in range(len(create_slice_list)):
        cur_im = np.load(create_slice_list[i])
        cur_mask = np.load(create_label_list[i])

        cur_im = (cur_im - low_range) / float(high_range - low_range)
        arr = np.nonzero(cur_mask)

        width = cur_mask.shape[0]
        height = cur_mask.shape[1]

        minA = min(arr[0])
        maxA = max(arr[0])
        minB = min(arr[1])
        maxB = max(arr[1])

        # with margin
        cropped_im = cur_im[max(minA - margin, 0): min(maxA + margin + 1, width), \
                                    max(minB - margin, 0): min(maxB + margin + 1, height)]
        cropped_mask = cur_mask[max(minA - margin, 0): min(maxA + margin + 1, width), \
                                    max(minB - margin, 0): min(maxB + margin + 1, height)]

        imgs[i] = pad_2d(cropped_im, plane, 0, XMAX, YMAX, ZMAX)
        imgs_mask[i] = pad_2d(cropped_mask, plane, 0, XMAX, YMAX, ZMAX)

        if i % 100 == 0:
            print('Done: {0}/{1} slices'.format(i, total))

    np.save('imgs_train_%s_%s.npy'%(current_fold, plane), imgs)
    np.save('masks_train_%s_%s.npy'%(current_fold, plane), imgs_mask)
    print('Training data created for fold %s, plane %s'%(current_fold, plane))


def load_train_data(current_fold, plane):
    imgs_train = np.load('imgs_train_%s_%s.npy'%(current_fold, plane))
    mask_train = np.load('masks_train_%s_%s.npy'%(current_fold, plane))
    return imgs_train, mask_train


def load_test_data(current_fold, plane):
    imgs_test = np.load('imgs_test_%s_%s.npy'%(current_fold, plane))
    mask_test = np.load('masks_test_$s_%s.npy'%(current_fold, plane))
    return imgs_test, mask_test


if __name__ == '__main__':

    data_path = sys.argv[1]
    current_fold = int(sys.argv[2])
    plane = sys.argv[3]

    # dim of each case (after padding zeors to max gt bounding box)
    ZMAX = int(sys.argv[4])
    YMAX = int(sys.argv[5])
    XMAX = int(sys.argv[6])

    margin = int(sys.argv[7])
    organ_ID = int(sys.argv[8])
    low_range = int(sys.argv[9])
    high_range = int(sys.argv[10])

    create_train_data(current_fold, plane)
