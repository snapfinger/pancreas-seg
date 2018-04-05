#!/bin/sh

# directory of project folder
DIR="/Users/yijun/panc"
DATADIR="${DIR}/data"
CODEDIR="${DIR}/repo"


# which fold to experiment, set it to 0 / 1/ 2/ 3
cur_fold=0

# settings same as in fixed-point
FOLDS=4
LOW_RANGE=-100
HIGH_RANGE=240
ORGAN_NUMBER=1
MARGIN=20
# to build a uniform dimension for input of NN
ZMAX=160
YMAX=256
XMAX=192
# parameters of training
epoch=10
init_lr=1e-5
# model to test
model_test=unet_fd${cur_fold}_Z_ep${epoch}_lr${init_lr}
# if want to view visualization of model's prediction slice by slice, set vis to true
vis=false

# ---------------- programs -------------------
# slice the 3d volumn to 2d slices
python slice.py ${DATADIR} ${ORGAN_NUMBER} ${FOLDS} ${LOW_RANGE} ${HIGH_RANGE}

# create data for training
python data.py ${DATADIR} ${cur_fold} Z ${ZMAX} ${YMAX} ${XMAX} ${MARGIN} ${ORGAN_NUMBER} ${LOW_RANGE} ${HIGH_RANGE}

# train the model
python unet.py ${DATADIR} ${cur_fold} Z ${epoch} ${init_lr}

# test the model
python testvis.py ${DATADIR} ${model_test} ${cur_fold} Z ${ZMAX} ${YMAX} ${XMAX} ${HIGH_RANGE} ${LOW_RANGE} ${MARGIN} ${vis}
