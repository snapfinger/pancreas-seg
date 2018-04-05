"""
This code is to test NN model and visualize output
"""
import numpy as np
import sys
import time
import matplotlib.pyplot as plt

from keras.models import Model, load_model
from keras.layers import Input, Activation, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, ZeroPadding2D, BatchNormalization
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import tensorflow as tf

from data import load_train_data, load_test_data
from utils import *

K.set_image_data_format('channels_last')  # Tensorflow dimension ordering

data_path  = sys.argv[1] + "/"
model_path = data_path + "models/"

# dir for storing results that contains
rst_path = data_path + "test-records/"
if not os.path.exists(rst_path):
    os.makedirs(rst_path)

model_to_test = sys.argv[2]
cur_fold = sys.argv[3]
plane = sys.argv[4]
im_z = int(sys.argv[5])
im_y = int(sys.argv[6])
im_x = int(sys.argv[7])
high_range = float(sys.argv[8])
low_range = float(sys.argv[9])
margin = int(sys.argv[10])
vis = sys.argv[11]

# prediction of trained model
pred_path = os.path.join(rst_path, "pred-%s/"%cur_fold)
if not os.path.exists(pred_path):
    os.makedirs(pred_path)

"""
Dice Ceofficient and Cost functions for training
"""
smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return  -dice_coef(y_true, y_pred)


def test(model_to_test, current_fold, plane, rst_dir, vis):
    print "-"*50
    print "loading model ", model_to_test
    print "-"*50

    model = load_model(model_path + model_to_test + '.h5', custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef':dice_coef})
    volume_list = open(testing_set_filename(current_fold), 'r').read().splitlines()
    total = len(volume_list)

    dsc = np.zeros((total, 2))

    # iterate all test cases
    for i in range(total):
        s = volume_list[i].split(' ')
        image = np.load(s[1])
        label = np.load(s[2])

        case_num = s[1].split("00")[1].split(".")[0]
        print "testing case: ", case_num

        image_ = np.transpose(image, (2, 0, 1))
        label_ = np.transpose(label, (2, 0, 1))

	    # standardize test data
        image_[image_ < low_range] = low_range
        image_[image_ > high_range] = high_range
        image_ = (image_ - low_range) / float(high_range - low_range)

        # for creating final prediction visualization
        pred = np.zeros_like(image_)

        for sli in range(label_.shape[0]):
            try:
                # crop each slice according to smallest bounding box of each slice
                width = label_[sli].shape[0]
                height = label_[sli].shape[1]

                arr = np.nonzero(label_[sli])

                if len(arr[0]) == 0:
        	        continue

                minA = min(arr[0])
                maxA = max(arr[0])
                minB = min(arr[1])
                maxB = max(arr[1])

                minAdiff = margin
                maxAdiff = margin
                minBdiff = margin
                maxBdiff = margin

                cropped = image_[sli, max(minA - minAdiff, 0): min(maxA + maxAdiff + 1, width), \
                        max(minB - minBdiff, 0): min(maxB + maxBdiff + 1, height)]
                cropped_mask = label_[sli, max(minA - minAdiff, 0): min(maxA + maxAdiff + 1, width), \
                        max(minB - minBdiff, 0): min(maxB + maxBdiff + 1, height)]

                image_padded_ = pad_2d(cropped, plane, 0, im_x, im_y, im_z)
                mask_padded_ = pad_2d(cropped_mask, plane, 0, im_x, im_y, im_z)

                image_padded_prep = preprocess_front(preprocess(image_padded_))

                out_ori = (model.predict(image_padded_prep) > 0.5).astype(np.uint8)

                out = out_ori[:,0:cropped.shape[0], 0:cropped.shape[1],:].reshape(cropped.shape)
                pred[sli, max(minA - minAdiff, 0): min(maxA + maxAdiff + 1, width), max(minB - minBdiff, 0): min(maxB + maxBdiff+ 1, height)] = out
                pred_vis = pred[sli, max(minA - minAdiff, 0): min(maxA + maxAdiff + 1, width), max(minB - minBdiff, 0): min(maxB + maxBdiff+ 1, height)]

                if vis == "true":
                    fig = plt.figure()
                    ax = fig.add_subplot(1, 3, 1)
                    ax.set_title("input test image")
                    ax.imshow(cropped, cmap=plt.cm.gray)

                    ax = fig.add_subplot(1, 3, 2)
                    ax.set_title("prediction")
                    ax.imshow(pred_vis, cmap=plt.cm.gray)

                    ax = fig.add_subplot(1, 3, 3)
                    ax.set_title("ground truth")
                    ax.imshow(cropped_mask, cmap=plt.cm.gray)

                    # plt.suptitle("slice %s"%sli)
                    fig.canvas.set_window_title("slice %s"%sli)
                    plt.axis('off')
                    plt.show()

            except KeyboardInterrupt:
                print 'KeyboardInterrupt caught'
                raise ValueError("terminate because of keyboard interruption")

        # ------------ write out for visualization ---------------
        np.save(pred_path + case_num + ".npy", pred) # prediction made by the trained model

	    # compute DSC
        cur_dsc, _, _, _ = DSC_computation(label_, pred)
        print cur_dsc

        dsc[i][0] = case_num
        dsc[i][1] = cur_dsc

    dsc_mean = np.mean(dsc[:,1])
    dsc_std = np.std(dsc[:,1])

    # record test dsc mean and standard deviation for each fold in the one file
    fd = open(rst_path + 'test_stats.csv','a+')
    fd.write("%s,%s,%s,%s\n"%(cur_fold, model_to_test, dsc_mean, dsc_std))
    fd.close()

    print "---------------------------------"
    print "mean: ", dsc_mean
    print "std: ", dsc_std

    dir1 = model_to_test.split("/")[0]
    if not os.path.exists(rst_path + dir1):
        os.mkdir(rst_path + dir1)

    # record test result case by case
    np.savetxt(rst_path + model_to_test + ".csv", dsc, fmt = "%i, %.5f", delimiter=",", header="case_num,DSC")


if __name__ == "__main__":

    start_time = time.time()

    test(model_to_test, cur_fold, plane, rst_path, vis)

    print "-----------test done, total time used: %s ------------"% (time.time() - start_time)
