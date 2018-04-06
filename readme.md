# Pancreas Segmentation in Abdominal CT Scans
### Introduction
---------
This is the code repository for the abstract [Pancreas Segmentation in Abdominal CT Scans](http://perfectroc.com/publication/Yijun_ISBI181page_final.pdf) presented at [IEEE International Symposium on Biomedical Imaging (ISBI) 2018](http://biomedicalimaging.org/2018/). The code for data preparation, test and utilities is largely from [OrganSegC2F](https://github.com/198808xc/OrganSegC2F). Please follow their requirements if you want to use the code in your work. There are no restrictions other than this.

We propose a U-Net based approach for pancreas segmentation. Under the same setting where bounding boxes are provided, this method outperforms previously reported results with a mean Dice Coefficient of 86.70 for the NIH dataset with 4-fold cross validation. Results show that a network designed specifically for and trained from scratch with biomedical images can achieve a better performance with much less training time compared to fine-tuning the models that are designed for and pre-trained on natural images.

### Main Dependencies
----------
- tensorflow-gpu (1.3.0)

- Keras (2.0.8)

- numpy (1.13.1)

- pandas (0.20.3)

- matplotlib (used for test output visualization)

### To run the experiment
--------
Step 1. Navigate to your project root directory, download the [pancreas segmentation dataset](https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT), use the [code](https://github.com/198808xc/OrganSegC2F/tree/master/DATA2NPY) to convert the images and annotations to numpy arrays.

Step 2. Clone this repo in your project root directory.

Step 3. Modify the path variables in `pipeline` to fit your own settings.

Step 4. Execute script

    chmod +x pipeline
    ./pipeline

Step 5. Modify the `cur_fold` variable in script `pipeline` to run in different fold. 

After each round, there should be 

1. A `test_stats.csv` in `/project-root-dir/data/test-records/` which records DSC mean and standard deviation for each fold
2. A `/project-root-dir/data/test-records/{test_model_name}.csv` which records DSC for each test case
3. Output prediction segmentation in `/project-root-dir/data/test-records/pred-{current_fold}` for each test case

Note: since the code is not well tested after clean-up, there may be some caveats when running the code. Issues and PRs are welcome.

### References
-----------
[1] Y. Zhou, L. Xie, W. Shen, Y. Wang, E. Fishman and A. Yuille, "A Fixed-Point Model for Pancreas Segmentation in Abdominal CT Scans", Proc. MICCAI, 2017

[2] H. Roth, L. Lu, A Farag, H-C Shin, J Liu, E. Turkbey, and R. M. Summers, "DeepOrgan: Multi-level deep convolutional networks for automated pancreas segmentation", Proc. MICCAI, 2015.

[3] O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation", Proc. MICCAI, 2015.
